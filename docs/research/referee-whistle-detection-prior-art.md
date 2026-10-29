# Referee whistle detection: prior art

Researched 2026-10-15. Sources are primary wherever I could reach them: repos and source code, papers, dataset pages, official product pages, rulebooks and patents. Where a claim could not be checked against a primary source, it is marked **UNVERIFIED**.

---

## TL;DR

**Yes, similar things exist, but nothing is a drop-in, maintained, general-purpose "referee whistle detector" for ordinary audio (phone, camera, broadcast).** What exists falls into five groups:

1. **Commercial, closed, solved for one narrow setup.** Basketball "whistle-controlled timing systems" stop the game clock when a referee blows. Examples are [Precision Time Systems](https://precisiontime.net/) (NBA/WNBA/NCAA/EuroLeague) and the [Bodet Sport electronic whistle](https://www.bodet-sport.com/products/sports-display-control/electronic-whistle.html). They use a **body-worn mic next to the referee's mouth** and **calibrate to each referee's whistle before every game**. FIBA has a formal spec for them: [≤ 0.1 s response, and they must ignore external whistles](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-equipment-v1-0.pdf).
2. **Open-source and mature, but tied to robots.** In RoboCup SPL robot soccer, [whistle detection is a "game critical module"](https://spl.robocup.org/wp-content/uploads/SPL-Rules-2024.pdf), so several teams have published real-time detectors in C++ or Rust plus labelled datasets: [B-Human](https://github.com/bhuman/BHumanCodeRelease), [Nao Devils](https://github.com/NaoDevils/AudioProcessing), [HULKs](https://github.com/HULKs/hulk), [HTWK](https://github.com/NaoHTWK/WhistleDetection) and [Bembelbots](https://github.com/Bembelbots/NaoWhistleDetection). This is the best code to learn from, but it targets NAO robot microphones in quiet-ish halls.
3. **Academic work for sports broadcast highlights.** A small body of mostly DSP papers (2011 to 2025) bandpass the 3 to 4.5 kHz region, then use energy, entropy or tonal-peak rules or simple classifiers. Reported results are around [92.5% sensitivity / 80.5% reliability](https://www.ijcaonline.org/archives/volume12/number11/1729-2340/) and [93% recall / 88% precision](https://arxiv.org/abs/1411.6496). None released code that I found.
4. **General audio taggers with a "Whistle" class, usable zero-shot.** AudioSet has `Whistle` (`/m/0l156k`), and [YAMNet](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv), [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) and [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) all output it. The class is generic, not referee-specific, and its labels are only [~60% accurate per Google's own estimate](https://research.google.com/audioset/dataset/whistle.html).
5. **Hobby repos and small datasets.** For example, a [volleyball whistle dataset on Hugging Face](https://huggingface.co/datasets/GYdevy/volleyball-whistles) (MIT, 3,492 annotations) and its [DSP+CNN pipeline](https://github.com/GYdevy/Volleylitics).

**What this means for building one.** The signal is well understood. A referee whistle is a strong, near-tonal sound with a fundamental of roughly **2.2 to 4.2 kHz**, two or three close tones and harmonics (see the [DSP section](#6-classical-dsp-approaches-and-whistle-frequency-characteristics)). A DSP baseline is therefore easy. The hard parts are:

- robustness to crowd noise, fan whistles, vuvuzelas, squeaking shoes and commentary;
- telling *the referee's* whistle apart from other whistles;
- the lack of a public, broadcast-scale, labelled benchmark.

The proven recipe in both robotics and commercial systems is a **cheap spectral prior combined with a small learned classifier, plus temporal smoothing**. Commercial systems add **per-whistle calibration**. The obvious gaps an open project could fill are a maintained, hardware-agnostic library (Python, or WASM/mobile) and an open benchmark across sports.

---

## 1. Open-source referee or sports whistle detectors

Repo metadata (language, license, last push, stars) comes from the GitHub API on 2026-10-15. "Last push" is `pushed_at`.

### 1a. RoboCup (robot soccer): the most mature open code

**Context.** In the RoboCup Standard Platform League, the head referee starts play, signals goals and ends halves with a real whistle. Teams must not react to whistles from other fields, and the 2024 rules list "Whistle detection" among the game-critical modules ([SPL Rules 2024, §3, §4, §6](https://spl.robocup.org/wp-content/uploads/SPL-Rules-2024.pdf)). The rules also say the referee "may choose to use any normal sports whistle" ([same](https://spl.robocup.org/wp-content/uploads/SPL-Rules-2024.pdf)), which pushes teams toward whistle-agnostic detectors.

| Project | What it does / approach | Stack | License | Last push | Reported accuracy |
|---|---|---|---|---|---|
| [bhuman/BHumanCodeRelease](https://github.com/bhuman/BHumanCodeRelease), module [`Src/Modules/Modeling/WhistleDetector`](https://github.com/bhuman/BHumanCodeRelease/tree/master/Src/Modules/Modeling/WhistleDetector) | Hybrid: a "physical model" (FFT peak in a band) and a neural net (`WhistleNetMk16.onnx`), with weighted confidences, an adaptive threshold, and a minimum-detections / max-pause temporal logic. The header says it is "based on the 2023 code release of the Nao Devils" by Kleingarn, Brämer and Röfer. The default config searches **2500–3700 Hz** ([`whistleDetector.cfg`](https://github.com/bhuman/BHumanCodeRelease/blob/master/Config/Scenarios/Default/whistleDetector.cfg)). | C++, FFTW, ONNX | Custom BSD-style B-Human license with RoboCup-usage conditions ([License.md](https://github.com/bhuman/BHumanCodeRelease/blob/master/License.md)) | 2025-11-06 | None in repo. See paper below. |
| [NaoDevils/AudioProcessing](https://github.com/NaoDevils/AudioProcessing) | Python scripts and notebooks to train the whistle-detection and whistle direction/distance networks, export to TFLite, plus annotation tools. Authors: Kleingarn & Brämer. | Python 3.9, TF/TFLite | MIT | 2025-11-14 | None in README |
| [HULKs/hulk](https://github.com/HULKs/hulk), crate [`crates/nodes/whistle_detection`](https://github.com/HULKs/hulk/tree/main/crates/nodes/whistle_detection) | Pure DSP: per-channel FFT, band energy compared with background-noise statistics. Default band **2700–4000 Hz**, 16 kHz sample rate, 1024-sample buffers ([`whistle_detection.json5`](https://github.com/HULKs/hulk/blob/main/etc/parameters/base/whistle_detection.json5)). | Rust | GPL-3.0 | 2026-10-15 (active) | None in repo |
| [NaoHTWK/WhistleDetection](https://github.com/NaoHTWK/WhistleDetection) | FFT, then a TFLite classifier (`data/whistle-classifier.tflite`) with a probability threshold. ALSA capture. | C++, FFTW, TFLite | README says "License AGPL" (no LICENSE file detected by GitHub) | 2023-02-14 | None |
| [Bembelbots/NaoWhistleDetection](https://github.com/Bembelbots/NaoWhistleDetection) | FFT-based classifier library plus a WAV debugging tool. | C++, FFTW, ALSA | MIT | 2017-11-16 (stale) | None |
| [bit-bots/whistle_detection](https://github.com/bit-bots/whistle_detection) | ResNet classifier trained on the B-Human 2017 SPL whistle dataset. README: "still under heavy development". | Python | MIT | 2023-01-31 (stale) | None |
| [Lvn-7/WhistleDetection-Pytorch](https://github.com/Lvn-7/WhistleDetection-Pytorch) | PyTorch re-implementation of the Nao Devils approach on the TU Dortmund dataset. | Python, PyTorch | None declared | 2026-04-21 | None found |
| [SentaFito53/Real-Time-Referee-Whistle-Recognition-on-Radxa-Rock-5B-…](https://github.com/SentaFito53/Real-Time-Referee-Whistle-Recognition-on-Radxa-Rock-5B-Using-ONNX-Optimized-CNN) | Collect, annotate, train a CNN, export ONNX, run real time on a Radxa Rock 5B (RoboCup Humanoid League). | Python, PyTorch, ONNX Runtime | None declared | 2026-05-11 | None found |
| Austrian Kangaroos (2014) | Whistle detector from the 2014 SPL technical challenge, listed on the [SPL open-source page](https://spl.robocup.org/open-source/) ([tgz](http://www.austrian-kangaroos.com/public/code/WhistleDetector.tgz)). | **UNVERIFIED** (not downloaded) | **UNVERIFIED** | 2014 | **UNVERIFIED** |

**Associated papers (RoboCup):**

- Kleingarn & Brämer, "Neural Network and Prior Knowledge Ensemble for Whistle Recognition", RoboCup 2023, LNCS ([DOI 10.1007/978-3-031-55015-7_2](https://doi.org/10.1007/978-3-031-55015-7_2)). This is the method behind the Nao Devils / B-Human detector. **UNVERIFIED:** the paper is closed access. I could not read its accuracy numbers; search-engine snippets say the ensemble beats either component alone and runs on the NAO.
- Kleingarn & Brämer, "Direction and Distance Estimation of Whistle Events on a NAO Robot", RoboCup 2024 (Springer 2025). The [SPL publications page](https://spl.robocup.org/publications/) says it won the best paper award ([Springer](https://link.springer.com/chapter/10.1007/978-3-031-85859-8_1)). Details are **UNVERIFIED** (paywalled).
- Ansalone et al., "Real-Time Multimodal Signal Processing for HRI in RoboCup: Understanding a Human Referee", AIRO 2024. Uses continuous-kernel CNNs for whistle detection on a NAO ([arXiv:2411.17347](https://arxiv.org/abs/2411.17347)). The abstract gives no accuracy figures.
- Backer & Visser, "Learning to recognize horn and whistle sounds for humanoid robots", BNAIC 2014 ([PDF](http://staff.fnwi.uva.nl/a.visser/publications/BackerBNAIC2014.pdf)). It compares FFT/DWT features with LogReg, SGD, SVM and AdaBoost on NAO recordings. All combinations scored so high that the authors say "a more challenging dataset is needed". The dataset is small and indoor.
- Bonarini, Lavatelli & Matteucci, "A Composite System for Real-Time Robust Whistle Recognition", RoboCup 2005 (LNCS 4020, cited as ref. [48] in [arXiv:1411.6496](https://arxiv.org/abs/1411.6496)). The [Politecnico di Milano project page](https://robocup.deib.polimi.it/MRT/WhistleRecognition.html) describes preprocessing plus a recurrent perceptron and an event counter. The full text is **UNVERIFIED**.
- Lopes, Ribeiro & Carvalho (Univ. Minho), "Whistle Sound Recognition in a Noisy Environment" ([PDF](https://repositorium.uminho.pt/server/api/core/bitstreams/ab9932ad-efff-416b-bc8b-6900299edc00/content)). It is a low-cost microcontroller with a 2–4 kHz analog bandpass, FFT and a configurable frequency/amplitude/repeat-count mask. It notes the method "cannot easily distinguish between slightly different referee whistles". Year **UNVERIFIED** (not printed in the PDF text).

### 1b. Sports-video and highlight repos (non-robot)

| Project | What it does / approach | Stack | License | Last push | Reported accuracy |
|---|---|---|---|---|---|
| [GYdevy/Volleylitics](https://github.com/GYdevy/Volleylitics) | Volleyball match analytics. Step 1 is whistle detection: DSP candidate detection with an estimated whistle band, then a ResNet CNN filter (`scripts/audio_pipeline.py`). Whistles are then paired into rallies. | Python, PyTorch | None declared | 2026-08-10 | The script computes P/R/F1, but no numbers are published in the README |
| [ZeyaRabani/Replay PR #23](https://github.com/ZeyaRabani/Replay/pull/23) | Soccer highlights audio track. It computes "2–4.5 kHz tonality" and frame-level `detect_whistles`, plus PANNs CNN14 tagging. Merged 2026-09-24. | Python | None declared | 2026-10-13 | None |
| [rachev3/basketball-highlights-extractor](https://github.com/rachev3/basketball-highlights-extractor) | Audio-peak highlight extractor that *filters out* referee whistles using frequency analysis. | Python | None declared | 2025-04-22 | None |

### 1c. Generic whistle detectors (not referee-specific)

These detect human whistling or kettle/pressure-cooker whistles and are mentioned only for completeness:

- [shubhamjain/whistlerr](https://github.com/shubhamjain/whistlerr): JS/WebAudio, MIT, last push 2022. It implements Nilsson et al.'s *Human Whistle Detection and Frequency Estimation* and flags "high energy spikes in the band of 500-5000Hz". The README says it "may fail against significant background noise".
- [RobQuistNL/node-whistle-detection](https://github.com/RobQuistNL/node-whistle-detection): Node.js, Apache-2.0, last push 2015.
- The [GitHub topic `whistle-detector`](https://github.com/topics/whistle-detector) and many "cooker whistle" repos are not relevant.

**Hugging Face:** a model search for "whistle" ([API](https://huggingface.co/api/models?search=whistle)) turned up nothing referee-related. Hits are mostly an ASR model family named "Whistle" (THU-SPMI) and a gated `Vinayak-0/whistle-classifier` whose card I could not read (**UNVERIFIED**). **Kaggle** was not searched systematically (**UNVERIFIED**).

---

## 2. Academic work on whistle detection in sports audio

| Paper | Sport / data | Method | Band | Results |
|---|---|---|---|---|
| Kathirvel, Manikandan & Soman, "Automated Referee Whistle Sound Detection for Extraction of Highlights from Sports Video", *IJCA* 12(11), 2011 ([page](https://www.ijcaonline.org/archives/volume12/number11/1729-2340/), [PDF](https://www.ijcaonline.org/volume12/number11/pxc3872340.pdf)) | ~12 h 11 min of American football, soccer and basketball; 16 kHz mono | Amplitude normalisation, least-squares linear-phase FIR bandpass, short-time energy, amplitude- and time-dependent thresholds | "approximately is 3750 Hz–4100 Hz", "determined empirically" | 324 detected, 42 FP, 26 FN; sensitivity 92.5%, reliability 80.5% |
| Raventós, Quijada, Torres & Tarrés, "Automatic Summarization of Soccer Highlights Using Audio-visual Descriptors", 2014 ([arXiv:1411.6496](https://arxiv.org/abs/1411.6496)) | Spanish league broadcasts (Catalan TV), 48 kHz | Goertzel DFT bins in band, then energy threshold, then **spectral-entropy** test (tonal vs spread), then keep frames with **2–3 peaks**. They state the whistle "spectrum is made of 3 tones at very close frequencies" that shift with blowing and whistle model. | 3.5–4.5 kHz | Recall 93%, precision 88% on the annotated database |
| Sicília Sansi (tutor F. Tarrés), *Detecció del xiulet per la generació de resums en vídeos esportius*, UPC bachelor thesis, 2013 ([UPCommons](https://upcommons.upc.edu/entities/publication/7b24742c-6ffc-4bc7-afa1-43f1e0eabe4e)) | European football | PCA-based detection | **UNVERIFIED** | "Satisfactory", no numbers on the abstract page |
| Trinh et al., "Detecting Referee Whistle Sounds in Soccer Videos Using Machine Learning Models", *J. Data Science & AI* 4(1), 2025 ([page](https://publications.isods.org/index.php/jdsai/article/view/33)) | Soccer footage | 1 s window, 50% overlap, STFT features; Naive Bayes vs SVM vs NN; PSD-threshold post-processing | 3300–4200 Hz | Abstract only says "highly effective". **UNVERIFIED:** the PDF requires login, so dataset and metrics were not checked. |
| Zhang & Ellis, "Detecting sound events in basketball video archive", Columbia E6820 project report, 2001 ([PDF](https://www.ee.columbia.edu/~dpwe/classes/e6820-2001-01/projects/dqzhang.pdf)) | Basketball | Focus is cheering (MFCC + NN) and dribbling; cited as whistle-related prior art by [1411.6496](https://arxiv.org/abs/1411.6496) | n/a | n/a |
| Baijal et al., "Sports highlights generation based on acoustic events detection: A rugby case study", IEEE ICCE 2015 ([arXiv:1509.06279](https://arxiv.org/abs/1509.06279)) | Rugby broadcasts | Multi-stage acoustic event classification | **UNVERIFIED** | The abstract does not say whether whistle is one of the event classes (**UNVERIFIED**) |
| Xu et al., "Audio for Sports Highlight Detection: A Comparative Empirical Study", 2026 ([arXiv:2609.17923](https://arxiv.org/abs/2609.17923)) | SV-Highlights benchmark | Audio-only GRU on pretrained audio embeddings vs AV fusion | n/a | Shows that audio alone is a strong highlight signal. Whistle is not a target class (per abstract). |

**Other observations:**

- **DCASE.** I found no DCASE challenge task or dataset centred on referee whistles. **UNVERIFIED:** I did not search every DCASE year's task list exhaustively.
- **SoccerNet.** The main public soccer-broadcast benchmark has 17 action-spotting classes and **no whistle class** ([task page](https://www.soccer-net.org/tasks/action-spotting)). Several labels (Kick-off, Foul, Offside, Penalty, Yellow card) normally coincide with a whistle, so they could serve as weak labels. Video access is NDA/password-gated per the [SoccerNet README](https://github.com/SoccerNet/SoccerNet).

---

## 3. Commercial products and sports-tech systems

### Precision Time Systems (PTS): basketball whistle-activated clock stop

- **Official site.** [precisiontime.net](https://precisiontime.net/) (precisiontime.com redirects there). Products are "Generation Six" ("our most precise whistle detection system to date"), Generation Five ("used exclusively in the NBA, WNBA, and G League", retired 2026-09-27) and Generation Four (college conferences and EuroLeague). Hardware is a dual collar mic and belt pack per referee, plus a base station on the game clock. The workflow is calibrate, detect, transmit, stop, and the timekeeper keeps a manual override. League deployment claims (WNBA, Big Ten, SEC, NBA preseason rollout) come from the vendor's own page.
- **How detection works, per the PTS patent** [US 10,504,300 B2](https://patents.google.com/patent/US10504300B2/en) (Costabile & Whitley; filed 2015, granted 2019; reexamination requested 2022):
  - It says the reference whistles "emit an audible sonic signal around 3150 hertz" and that the system "may be made particularly responsive to that whistle by being tuned to 3150 hertz". The context is the Fox 40 Classic.
  - The front-end filter has −3 dB cutoffs of about 1.25 kHz and 7.5 kHz.
  - An FFT finds the strongest peak plus two peaks above and two below, forming a **five-frequency "fingerprint"** per official. It is captured by having each official blow several times before the game.
  - Match threshold is "preferably approximately 90%".
- **Other PTS patents** are listed on the vendor's [patents page](https://precisiontime.net/elementor-page-602/): US 7,920,052 B2, US 12,046,081 B2, US 12,318,676 B2 and US 2024/0408474 A1.
- **History and context (secondary, for colour only).** [Sports Video Group, 2020](https://www.sportsvideo.org/2020/07/16/officials-whistles-go-digital/) says the Remote Whistle Timing System was introduced in 1997 and that calibration has the official blow twice before play.

### Bodet Sport: electronic whistle (basketball)

- [Official product page](https://www.bodet-sport.com/products/sports-display-control/electronic-whistle.html): a 62 g clip-on box per referee with a microphone cable. It talks to the scoring-table keyboard by radio and stops the clock "in a single whistle blow".
- It lists compatible whistles (Fox Mini, Fox 40 Classic, Fox 40 Sonik, Molten Dolfin Pro/B, Blazza Titanium) and claims a "patented concept" reliable "whatever the level of ambient noise". There is no algorithm detail and no latency figure.

### FIBA specification for whistle-controlled timing

The [FIBA Official Basketball Rules 2026: Basketball Equipment](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-equipment-v1-0.pdf), §13, says the system:

- "shall stop the game clock when the referee's whistle is blown, with a response time of 0.1 seconds or less";
- must work "at all locations on the basketball court";
- "will not stop the game clock with any external whistle noise".

§14 / Table 7 sets referee-whistle loudness at **105 dB (Level 1) / 95 dB (Level 2) at 3 m and 8.3 kPa**. Its "Frequency range" column shows **170 Hz (Level 1) / 150 Hz (Level 2)**. **UNVERIFIED interpretation:** the extracted table is ambiguous about whether this is a bandwidth or a minimum spread. It is clearly not a centre frequency.

### Broadcast and analytics vendors

I found **no official product page from a broadcast or analytics vendor** (e.g., Hudl, Pixellot, Sportradar, WSC Sports) that advertises referee-whistle detection as a feature. **UNVERIFIED:** such features may exist internally without being marketed. Searches surfaced only academic papers and patents.

Two adjacent patents:

- [US 11,501,746 B2 "Toot suite whistle pack"](https://patents.google.com/patent/US11501746): sets of distinguishable whistles, optionally emitting an RF "chirp" to timestamp AV feeds.
- [US 10,129,608](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10129608): highlight detection from audio cues. Not read in full (**UNVERIFIED**).

---

## 4. General-purpose audio classifiers with a whistle class

Class names and IDs below are from the files themselves, downloaded 2026-10-15.

### AudioSet ontology

From [`ontology.json`](https://github.com/audioset/ontology/blob/master/ontology.json) and [`class_labels_indices.csv`](http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv):

| Class | MID | Index in 527-class label set | Parent | Definition (ontology) |
|---|---|---|---|---|
| **Whistle** | `/m/0l156k` | 402 | Alarm | "The sound of an instrument that produces sound from a stream of forced air." Children: Kettle whistle, Steam whistle. |
| Steam whistle | `/m/06hck5` | 403 | Whistle | Steam-driven device |
| Kettle whistle | `/g/11b630rrvh` | **not in the 527-class set** | Whistle; Domestic sounds | |
| Whistling | `/m/01w250` | 40 | Human sounds | Lips-produced whistling. Child: Wolf-whistling (`/m/079vc8`). |
| Train whistle | `/m/04zmvq` | 330 | Train | |

There is **no "Referee whistle" or "Sports whistle" class.** A referee pea or pealess whistle falls under the generic `Whistle`. Google's [class page for Whistle](https://research.google.com/audioset/dataset/whistle.html) reports 663 videos (1.8 h): 60 eval, 61 balanced-train and 542 unbalanced-train. Its quality estimate is "Medium": 6 of 10 sampled labels were correct, and the unbalanced set "may be significantly lower".

### Models that output these classes

- **YAMNet** (Google): 521 AudioSet classes. Its [class map](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv) has `396,/m/0l156k,Whistle` (also `35 Whistling`, `324 Train whistle`, `397 Steam whistle`).
  - Input is resampled to 16 kHz, with a log-mel front end over **125–7500 Hz** and 0.96 s patches at a 0.48 s hop. Balanced mAP on the AudioSet eval set is 0.306 ([README](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md), [`params.py`](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/params.py)). The repo is Apache-2.0 ([LICENSE](https://github.com/tensorflow/models/blob/master/LICENSE)).
  - **MediaPipe Audio Classifier** ships YAMNet as its only and recommended model for Android, iOS, Python and Web ([docs](https://developers.google.com/edge/mediapipe/solutions/audio/audio_classifier)). That makes YAMNet the easiest on-device zero-shot baseline.
- **PANNs** ([qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn)): 527 classes, so it includes Whistle at index 402. Best reported mAP is 0.439 (Wavegram-Logmel-CNN), and CNN14 checkpoints are on Zenodo. MIT license; last push 2024-07-25.
- **AST** ([MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)): `config.json` maps `"402": "Whistle"` and `"40": "Whistling"`. BSD-3-Clause; source in [YuanGongND/ast](https://github.com/YuanGongND/ast).
- **Edge Impulse.** There is no built-in whistle model. A public sample project ([#233106](https://studio.edgeimpulse.com/public/233106/latest), Apache-2.0) classifies "clap" vs "whistle" from about 1.5 min of 16 kHz audio. It is a toy, and it does not say what kind of whistle.

**Caveat (my inference, not measured).** These models were trained on a generic, noisily labelled `Whistle` class. How well they score referee whistles in crowd noise, versus fan whistles, kettle whistles or squeaking sneakers, is **not documented anywhere I found**. Treat them as a baseline to measure, not a solution.

---

## 5. Datasets containing whistle sounds

| Dataset | Whistle content | Size | License | Notes |
|---|---|---|---|---|
| **TU Dortmund "Whistle Detection Dataset" v1.0** (Kleingarn, Brämer; 2024) ([DOI](https://doi.org/10.17877/tudodata-2024-m0f6bmi1), [landing](https://data.tu-dortmund.de/citation?persistentId=doi:10.17877/tudodata-2024-m0f6bmi1)) | Referee whistles recorded on a NAO V6 robot (RoboCup) | ~2.8 GB: NumPy spectrogram train/test splits plus 2 TestBench WAVs | **CC BY-SA 4.0** | Not all data could be published "for legal reasons" (landing page). Features are 1024-sample Hamming-windowed spectra in dB. |
| **TU Dortmund "Whistle Localisation Dataset" v1.0** (2024) ([DOI](https://doi.org/10.17877/tudodata-2024-m0fdqqyq)) | Whistles at various distances and angles, NAO V6 | 149 files (147 audio) | **CC BY-SA 4.0** | Labels via the [AudioProcessing](https://github.com/NaoDevils/AudioProcessing) tooling |
| **B-Human SPL whistle dataset 2017** ([index](https://b-human.informatik.uni-bremen.de/public/datasets/whistle-2017/), listed on [SPL datasets](https://spl.robocup.org/datasets/)) | 10 multichannel recordings from RoboCup matches with per-channel start/end labels (format described in [bit-bots README](https://github.com/bit-bots/whistle_detection)) | `whistle.zip`, 290 MiB | **Not stated (UNVERIFIED)** | |
| **Volleylitics whistle dataset** ([HF: GYdevy/volleyball-whistles](https://huggingface.co/datasets/GYdevy/volleyball-whistles)) | 14 full indoor volleyball match WAVs (22.05 kHz); `whistles_all.json` has **3,492** annotations (serve 1,522; rally_end 1,520; other 450) across 10 matches (my count from the file) | ~14 matches | **MIT** (card metadata) | Closest public "real referee in a real gym" data. Single uploader, small, not peer reviewed. |
| **AudioSet** ([site](https://research.google.com/audioset/)) | `Whistle` /m/0l156k: 663 YouTube clips, ~60% label quality ([class page](https://research.google.com/audioset/dataset/whistle.html)) | 10 s clips | Labels CC BY 4.0, audio is YouTube (**license of the underlying audio UNVERIFIED** here) | Generic whistles, not referee-specific |
| **FSD50K** ([Zenodo](https://zenodo.org/records/4060432)) | **No whistle class.** I checked `vocabulary.csv` (200 classes): no Whistle, Whistling or Steam/Train/Kettle whistle. Nearest is `Alarm` /m/07pp_mv. | 51,197 clips, 108 h | CC BY (dataset); per-clip CC0/BY/BY-NC/Sampling+ | Useful as negatives |
| **ESC-50** ([meta CSV](https://github.com/karolpiczak/ESC-50/blob/master/meta/esc50.csv)) | **No whistle class** (50 classes checked) | 2,000 clips | CC BY-NC (**UNVERIFIED** here) | Useful as negatives |
| **SoccerNet** ([task](https://www.soccer-net.org/tasks/action-spotting), [repo](https://github.com/SoccerNet/SoccerNet)) | No whistle labels. Broadcast audio with 17 event classes that often co-occur with whistles. | 500 games | Code MIT; videos NDA-gated | Candidate for weak supervision |
| **Freesound** individual clips, e.g. [referee-whistle.wav](https://freesound.org/people/Pablo-F/sounds/90743/) | Isolated referee whistles | Varies | Per-clip CC | Good for augmentation and synthetic mixing |

---

## 6. Classical DSP approaches and whistle frequency characteristics

### What the primary sources say about the frequencies

Manufacturer spec sheets do **not** publish a frequency. I could not find a Hz figure on Fox 40's or Acme's own pages. A "3600 Hz" figure for the Acme Thunderer 60.5 appears only on retailer listings (**UNVERIFIED**). The best primary sources are therefore patents (with measured spectra) and papers:

| Source | Whistle | Stated frequency |
|---|---|---|
| PTS patent [US 10,504,300 B2](https://patents.google.com/patent/US10504300B2/en) | Fox 40 Classic (pealess, 3-chamber) | "around 3150 hertz" |
| Fox 40 patent [US 8,028,642 B2](https://patents.google.com/patent/US8028642B2/en) (Foxcroft; assigned to Fox 40 International) | A Fox 40 finger-grip design | "peak frequency of approximately 2216 hertz". Claims recite a principal frequency of 2.0–2.4 kHz and a pulse (beat) frequency below 100 Hz. Contrasts with a traditional pea whistle's ~50 Hz pulse. |
| Molten patent [US 6,837,177 B2](https://patents.google.com/patent/US6837177B2/en) | Molten two-chamber designs | Chambers at 3.4 kHz and 3.7 kHz, giving a 0.3 kHz beat. Measured fundamentals 3.06–3.23 kHz with harmonics at ~6.2, 9.6 … 15.5 kHz. |
| Shishido patent [US 8,006,634 B2](https://patents.google.com/patent/US8006634B2/en) | Prior-art statement | "from 3.3 to 3.5 kHz for pea-type whistles, and from 3.5 to 3.7 kHz for pealess-type whistles". Its embodiments measure peaks at 2750–3688 Hz plus a secondary peak near 7 kHz or 10 kHz. |
| [Raventós et al. 2014](https://arxiv.org/abs/1411.6496) | Professional soccer whistles in broadcast | Band 3.5–4.5 kHz; "3 tones at very close frequencies" that drift with excitation, blowing direction and model |
| [Kathirvel et al. 2011](https://www.ijcaonline.org/volume12/number11/pxc3872340.pdf) | Broadcast (3 sports) | 3750–4100 Hz, chosen empirically |
| [Trinh et al. 2025](https://publications.isods.org/index.php/jdsai/article/view/33) | Soccer | 3300–4200 Hz |
| [Lopes et al. (U. Minho)](https://repositorium.uminho.pt/server/api/core/bitstreams/ab9932ad-efff-416b-bc8b-6900299edc00/content) | Referee whistles, close mic | Peaks "around 2.3 and 2.9 kHz"; filter 2–4 kHz |
| HULKs [params](https://github.com/HULKs/hulk/blob/main/etc/parameters/base/whistle_detection.json5) / B-Human [config](https://github.com/bhuman/BHumanCodeRelease/blob/master/Config/Scenarios/Default/whistleDetector.cfg) | Any sports whistle (SPL) | 2700–4000 Hz / 2500–3700 Hz search bands |
| [Backer & Visser 2014](http://staff.fnwi.uva.nl/a.visser/publications/BackerBNAIC2014.pdf) | "a complex whistle sound, as used by a referee" | "broad peak between 6 and 7kHz". This is an **outlier**, possibly a harmonic or a specific whistle. Not reconciled. |

**Synthesis (my inference from the table above).**

- The fundamentals of common referee whistles cluster around **~2.2–4.2 kHz**.
- Pealess multi-chamber whistles produce **two or three close tones**, whose beating is audible as a ~50–400 Hz modulation.
- Strong **harmonics** run up to ~15 kHz.
- Models that use a 16 kHz sample rate (Nyquist 8 kHz) or YAMNet's 125–7500 Hz mel range see the fundamental and the first harmonic, but not the higher ones.
- The PTS patent's 1.25–7.5 kHz front end and its five-peak fingerprint are consistent with this.

### DSP recipes documented in the sources

1. **Bandpass and short-time energy with thresholds.** Kathirvel 2011 and the HULKs implementation: band energy compared with running background statistics. Cheap, but prone to false alarms from cheering ([Raventós et al.](https://arxiv.org/abs/1411.6496) call this out explicitly).
2. **Tonality and peak-structure checks.** Raventós 2014: in-band spectral entropy (low means tonal), plus requiring 2–3 peaks. The PTS patent: centre peak plus four neighbouring peaks matched to a calibrated fingerprint.
3. **Temporal logic.** Minimum consecutive detections, maximum gap and refractory period (B-Human's `minDetections`, `maxDetectionPause`, `minTimeBetweenWhistles`). Repeat counter R (Lopes et al.).
4. **Hybrid DSP prior and small NN.** The Nao Devils / B-Human ensemble, and Volleylitics' DSP candidates followed by a CNN filter.
5. **Calibration to the specific whistle or referee.** PTS and Bodet; B-Human has a `freqCalibration` option.

---

## 7. Gaps and opportunities

- **No maintained, hardware-agnostic open library.** The good open code (B-Human, Nao Devils, HULKs, HTWK) is embedded in NAO robot frameworks: ALSA, NAO mic geometry, and C++/Rust module systems with RoboCup-specific licenses (B-Human) or GPL/AGPL. A permissively licensed Python package with optional WASM/mobile/ONNX export, taking any mono stream, does not appear to exist.
- **No public benchmark for broadcast or field-side referee whistles.** Academic papers used private broadcast data, and SoccerNet has no whistle labels. The only open real-match sets are robot-soccer recordings (TU Dortmund, CC BY-SA) and one small volleyball set (MIT). A multi-sport, labelled benchmark with crowd, commentary and fan-whistle negatives would be novel and useful.
- **"Referee whistle vs other whistles" is unsolved in the open.** Commercial systems sidestep it with a body-worn mic and per-referee calibration. FIBA requires rejecting "any external whistle noise". There is no open method for doing this from a distant microphone.
- **The zero-shot baseline is unmeasured.** Nobody appears to have published how YAMNet, PANNs or AST's `Whistle` class performs on referee whistles. A quick evaluation would be informative and cheap.
- **Weak supervision from existing labels.** SoccerNet events (kick-off, foul, offside, cards) and volleyball serves or rally ends co-occur with whistles. They could bootstrap labels at scale. This is my suggestion, not something I found done.
- **Latency.** FIBA's 0.1 s budget is far below YAMNet's ~0.975 s minimum input window ([YAMNet README](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md)). Real-time clock control needs a short-window DSP or a tiny model, not a generic tagger.

## 8. Open questions

1. **Use case.** Live real-time trigger (clock stop, robot start, app event) or offline indexing of recorded matches? This sets the latency budget: 0.1 s per FIBA vs seconds.
2. **Microphone placement.** Body-worn near the referee (the commercial approach), phone or camera at the sideline, or broadcast mix? This dominates SNR and the "which whistle" problem.
3. **Sport(s) and venue.** Indoor gym reverberation and shoe squeaks (basketball, volleyball) vs outdoor stadium crowd, vuvuzelas and fan whistles (soccer).
4. **Must it identify *the* referee's whistle,** or is "a sports whistle was blown" enough?
5. **Target platform:** browser, mobile, embedded MCU or server. This decides between DSP-first and NN-first, and whether YAMNet/MediaPipe is viable.
6. **License constraints.** The B-Human code has RoboCup-specific conditions, HULKs is GPL-3.0, HTWK states AGPL, and the TU Dortmund data is CC BY-SA (share-alike).
7. **Still unverified:**
   - accuracy numbers in the Kleingarn & Brämer papers (paywalled);
   - the dataset and metrics of Trinh et al. 2025 (login-walled);
   - the meaning of FIBA Table 7's "Frequency range 170/150 Hz";
   - official manufacturer frequency specs for Fox 40 and Acme (not published on their sites, as far as I found);
   - the license of the B-Human 2017 dataset;
   - whether any broadcast or analytics vendor ships whistle detection.
