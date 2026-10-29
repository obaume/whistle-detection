# Whistle Detection

Finds referee whistle blasts in ice-hockey game footage and hands them to [hockey-analyzer](https://github.com/obaume/hockey-analyzer) as candidates for the tagger to review, so stoppages don't have to be found by scrubbing.

## Language

**Whistle blast**:
The actual sound of an official blowing a whistle, as it occurs in the footage's audio. What the tool is trying to find; it may or may not be found.
_Avoid_: whistle (ambiguous with the app's **Whistle**)

**Stop whistle**:
A whistle blast that stops live play. It is what the tagger needs to find, and the only kind the accuracy bar counts.
_Avoid_: stoppage (that's hockey-analyzer's logged event, not the sound)

**Line-change whistle**:
A whistle blast during dead time, after a stop whistle and before the next faceoff, from the referee signalling that the line change is over and the faceoff can take place. Normally one per faceoff, except the faceoff that opens a period.
_Avoid_: dead-time false alarm (it's a real whistle blast, not a mistake)

**Detection**:
One output of this tool: a video timestamp where it believes a whistle blast occurs, with a confidence and the name and version of the detector that produced it. Raw and unreviewed.
_Avoid_: Whistle, proposal, prediction

**Whistle** (hockey-analyzer's term):
The app's record of a Detection once imported, awaiting or carrying the tagger's resolution. Never used for anything inside this repo.
_Avoid_: using it for a Detection or a whistle blast

## Relationships

- A **Detection** of a **whistle blast** becomes a **Whistle** when hockey-analyzer imports it.
- A **whistle blast** is either a **stop whistle** or a **line-change whistle**. Between two faceoffs there is normally one of each, in that order.
