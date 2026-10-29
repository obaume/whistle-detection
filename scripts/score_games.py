# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
"""Score the pitch-tonality detector against hockey-analyzer's tagged games.

Ground truth is the app's tagged events, read-only from its database:
- every `stoppage` and every `shot_attempt` with outcome goal is a stop whistle;
- a stop is a goal if the next faceoff is at the centre dot (checked against
  the league's period scores, which are printed alongside);
- the stretch between a stop and the next faceoff is dead time, where
  line-change whistles live and count neither way.

Usage: uv run scripts/score_games.py [GAME_ID ...] [--threshold 8]
"""

import argparse
import json
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg
import numpy as np

DB = Path(os.environ["APPDATA"]) / "Hockey Analyzer" / "hockey_analyzer.db"
SR = 16000
N, HOP = 1024, 256  # 64 ms window, 16 ms hop
BAND = (2200, 4600)
MIN_GAP_S = 3.0
LOOSE_S = 4.0  # earlier runs counted a hit within ±4 s of the tag
BAR_S = 1.0  # the accuracy bar: within 1 s
PERIOD_END_S = 6.0  # detections this close to a period end are the end signal


@dataclass
class Stop:
    ms: int
    period: int
    goal: bool = False


@dataclass
class Period:
    number: int
    start_ms: int
    end_ms: int  # tagged period end, or the last tagged event if tagging stops early
    complete: bool
    league_goals: int | None


@dataclass
class Game:
    id: int
    name: str
    venue: str
    video: str
    periods: list[Period] = field(default_factory=list)
    stops: list[Stop] = field(default_factory=list)
    faceoffs: list[int] = field(default_factory=list)
    period_ends: list[int] = field(default_factory=list)


def load_game(db: sqlite3.Connection, game_id: int) -> Game:
    row = db.execute(
        "select g.venue, th.name, ta.name, g.period_scores, g.video_path from games g "
        "left join teams th on th.id = g.home_team_id "
        "left join teams ta on ta.id = g.away_team_id where g.id = ?",
        (game_id,),
    ).fetchone()
    venue, home, away, scores, video = row
    scores = json.loads(scores) if scores else []
    game = Game(game_id, f"{home} vs {away}", venue, video)

    events = db.execute(
        "select event_type, video_timestamp, faceoff_x, faceoff_y, shot_outcome from events "
        "where game_id = ? and event_type in "
        "('faceoff', 'stoppage', 'period_start', 'period_end', 'shot_attempt') "
        "order by video_timestamp, case event_type when 'period_start' then 0 else 1 end",
        (game_id,),
    ).fetchall()

    period, start, pending = 0, None, None
    last_ms = 0
    for kind, ms, x, y, outcome in events:
        if kind == "period_start":
            if start is not None:  # previous period had no tagged end
                game.periods.append(_period(period, start, last_ms, False, scores))
            period, start, pending = period + 1, ms, None
        elif kind == "period_end":
            game.periods.append(_period(period, start, ms, True, scores))
            game.period_ends.append(ms)
            start, pending = None, None
        elif kind == "stoppage" or (kind == "shot_attempt" and outcome == "goal"):
            if pending is not None and abs(pending.ms - ms) < 1000:
                continue  # a goal tagged next to its stoppage is one whistle
            pending = Stop(ms, period, goal=kind == "shot_attempt")
            game.stops.append(pending)
        elif kind == "faceoff":
            game.faceoffs.append(ms)
            if pending is not None and x is not None and abs(x) < 5 and abs(y) < 5:
                pending.goal = True
            pending = None
        last_ms = max(last_ms, ms)
    if start is not None:
        game.periods.append(_period(period, start, last_ms, False, scores))
    return game


def _period(number, start, end, complete, scores):
    goals = scores[number - 1]["home"] + scores[number - 1]["away"] if number <= len(scores) else None
    return Period(number, start, end, complete, goals)


def decode(video: str, start_s: float, end_s: float) -> np.ndarray:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", video,
        "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def tonality(x: np.ndarray, offset_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The probe's score: band peak-to-median ratio times band energy share."""
    win = np.hanning(N).astype(np.float32)
    freqs = np.fft.rfftfreq(N, 1 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    above = freqs >= 200
    frames = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP]
    score = np.empty(len(frames), dtype=np.float32)
    for i in range(0, len(frames), 20000):
        S = np.abs(np.fft.rfft(frames[i : i + 20000] * win, axis=1)) ** 2
        Sb = S[:, band]
        peak_ratio = (Sb.max(axis=1) + 1e-12) / (np.median(Sb, axis=1) + 1e-12)  # 1 on silence
        share = Sb.sum(axis=1) / (S[:, above].sum(axis=1) + 1e-12)
        score[i : i + 20000] = np.log10(peak_ratio) * share
    t = offset_s + (np.arange(len(score)) * HOP + N / 2) / SR
    k = 9  # ~150 ms
    return t, np.convolve(score, np.ones(k) / k, mode="same")


def pick_peaks(t: np.ndarray, z: np.ndarray, thr: float) -> np.ndarray:
    chosen: list[float] = []
    for i in np.argsort(z)[::-1]:
        if z[i] < thr:
            break
        if all(abs(t[i] - c) >= MIN_GAP_S for c in chosen):
            chosen.append(t[i])
    return np.array(sorted(chosen))


def score_game(game: Game, thr: float) -> dict:
    t0 = time.perf_counter()
    ts, ss = [], []
    for p in game.periods:
        x = decode(game.video, p.start_ms / 1000 - 2, p.end_ms / 1000 + 8)
        t, s = tonality(x, p.start_ms / 1000 - 2)
        ts.append(t)
        ss.append(s)
    audio_s = sum((p.end_ms - p.start_ms) / 1000 + 10 for p in game.periods)
    t, s = np.concatenate(ts), np.concatenate(ss)
    z = (s - np.median(s)) / (np.percentile(s, 75) - np.percentile(s, 25) + 1e-12)
    dets = pick_peaks(t, z, thr)
    zi = np.interp(dets, t, z)
    elapsed = time.perf_counter() - t0

    # Match each stop to the nearest detection within the loose window.
    stops = []
    for st in game.stops:
        tag = st.ms / 1000
        near = dets[np.abs(dets - tag) <= LOOSE_S]
        off = (near - tag)[np.argmin(np.abs(near - tag))] if len(near) else None
        m = (t >= tag - LOOSE_S) & (t <= tag + LOOSE_S)
        stops.append((st, off, float(z[m].max()) if m.any() else float("nan")))

    # Classify detections that match no stop.
    stop_s = np.array([st.ms / 1000 for st in game.stops])
    face_s = np.array(sorted(game.faceoffs)) / 1000
    end_s = np.array(game.period_ends) / 1000
    kinds = []
    for d in dets:
        if len(stop_s) and np.min(np.abs(stop_s - d)) <= LOOSE_S:
            kinds.append("stop")
        elif len(end_s) and np.min(np.abs(end_s - d)) <= PERIOD_END_S:
            kinds.append("period_end")
        else:
            prev_stop = stop_s[stop_s < d].max(initial=-1)
            prev_face = face_s[face_s < d].max(initial=-1)
            # dead time runs from the stop to just after the next faceoff
            dead = prev_stop > prev_face or (prev_face >= 0 and d - prev_face <= 1.0)
            # two faceoffs with no stop tagged between them: the stop wasn't tagged
            next_face = face_s[face_s > d].min(initial=np.inf)
            gap = prev_face >= 0 and not ((stop_s > prev_face) & (stop_s < next_face)).any()
            kinds.append("line_change" if dead else "untagged" if gap else "live")

    return {
        "game": game, "dets": dets, "z": zi, "kinds": kinds, "stops": stops,
        "elapsed": elapsed, "audio_s": audio_s,
    }


def fmt(sec: float) -> str:
    return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{sec % 60:05.2f}"


def report(r: dict) -> None:
    g: Game = r["game"]
    print(f"\n## Game {g.id}: {g.name} ({g.venue})")
    for p in g.periods:
        inferred = sum(1 for st in g.stops if st.period == p.number and st.goal)
        n = sum(1 for st in g.stops if st.period == p.number)
        flag = "" if inferred == p.league_goals else "  <-- goal count mismatch"
        print(f"  P{p.number}: {n} stops, goals inferred {inferred} / league {p.league_goals}"
              f"{'' if p.complete else ' (partly tagged)'}{flag}")

    def rate(sel, window):
        hits = sum(1 for st, off, _ in sel if off is not None and abs(off) <= window)
        return hits, len(sel)

    non_goal = [x for x in r["stops"] if not x[0].goal]
    goals = [x for x in r["stops"] if x[0].goal]
    for label, sel in (("Stop whistles (bar, no goals)", non_goal), ("Goal whistles (info)", goals)):
        h1, n = rate(sel, BAR_S)
        h4, _ = rate(sel, LOOSE_S)
        pct = lambda h: f"{100 * h / n:.0f}%" if n else "-"
        print(f"  {label}: {h1}/{n} ({pct(h1)}) within 1 s, {h4}/{n} ({pct(h4)}) within 4 s")

    kinds = r["kinds"]
    complete = [p for p in g.periods if p.complete]
    print(f"  Detections: {len(kinds)} total, {kinds.count('live')} live-play false alarms, "
          f"{kinds.count('line_change')} dead time, {kinds.count('period_end')} period-end, "
          f"{kinds.count('untagged')} in an untagged stop")
    print(f"  Speed: {r['audio_s'] / 60:.0f} min of audio in {r['elapsed']:.1f} s "
          f"({r['audio_s'] / r['elapsed']:.0f}x real time)")

    offs = [off for st, off, _ in non_goal if off is not None]
    if offs:
        print(f"  Offset detection - tag (non-goal hits): median {np.median(offs):+.2f} s, "
              f"80% within ±{np.percentile(np.abs(offs), 80):.2f} s")
    misses = [(st, off, zmax) for st, off, zmax in non_goal if off is None or abs(off) > BAR_S]
    if misses:
        print("  Stop whistles missed (1 s):")
        for st, off, zmax in misses:
            why = "no detection" if off is None else f"nearest {off:+.1f} s"
            print(f"    P{st.period} {fmt(st.ms / 1000)}  {why}, best z {zmax:.1f}")
    live = [(d, z) for d, z, k in zip(r["dets"], r["z"], kinds) if k == "live"]
    if live:
        print("  Live-play false alarms:")
        for d, z in live:
            print(f"    {fmt(d)}  z {z:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", type=int)
    ap.add_argument("--threshold", type=float, default=8.0)
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    ids = args.games or [r[0] for r in db.execute(
        "select distinct game_id from events where event_type = 'stoppage' order by 1")]
    print(f"# Pitch-tonality detector, threshold {args.threshold}")
    for gid in ids:
        report(score_game(load_game(db, gid), args.threshold))


if __name__ == "__main__":
    main()
