# Match momentum, critical moments and tactic notes

This layer sits on top of the offline analysis in [MATCH_ANALYSIS.md](MATCH_ANALYSIS.md). It reads the saved shot files, derives a momentum timeline, a ranked list of critical moments and live tactic notes, and exports them with the tracked pitch positions as a static web viewer (`web/`). The viewer replays precomputed output; **detection, tracking and pitch mapping never run in the browser**.

Everything is an estimate from broadcast video. It is not official match data, and the release gates in [MATCH_ANALYSIS.md](MATCH_ANALYSIS.md) still apply.

## Run it

```bash
python -m pip install -r requirements-analysis.txt
python tools/setup_pitch.py
python -m match_analysis analyze \
  --video input_videos/08fd33_4.mp4 \
  --checkpoint runs/performance-study/improved/weights/best.pt \
  --config configs/match.json --device 0 --output artifacts/match

# CPU only, repeatable on a finished run. Needs ffmpeg for the browser-friendly copies.
python -m match_analysis export-viewer --run artifacts/match --output web/data
python -m http.server -d web 8000      # then open http://localhost:8000
```

`export-viewer` writes `analysis.json` plus `clip.mp4` (H.264) and `clip.webm` (VP9, for browsers without H.264), both 720p without audio. Use `--no-transcode` to copy an already-compatible MP4. Serve `web/` over HTTP with byte-range support (GitHub Pages, `http.server` for playback, or any static host) because browsers need ranges to seek video. Other runs can be opened with `index.html?data=path/to/folder/`.

Pass `--metadata` to `analyze` with attacking-direction intervals to enable the territory component and direction-aware wording (see `configs/match_metadata.example.json`). Without it the viewer states that direction is unknown and never guesses.

## What each panel means

### Pitch

Top-down view in metres on a 105 × 68 m pitch. Each dot is a tracked person mapped through the PnLCalib homography: solid for observed, hollow dashed for a short velocity prediction (at most one second). Team colours are the accumulated jersey clusters, goalkeepers are yellow-green and referees dark with a gold ring. The dashed outline is the **camera footprint**: the ground area the broadcast camera currently covers, so it is obvious which part of the pitch is observed and which is not. Replays, close-ups and failed calibrations are greyed out rather than guessed.

### Momentum index

Per second, up to four components are computed from **observed** evidence only, each from Team A's perspective in [-1, 1]:

| Component | Weight | Definition |
| --- | ---: | --- |
| `control` | 0.30 | Share of covered observed team-control samples won by A versus B (the pipeline's strict control: ball within 2.5 m, 0.5 s persistence) |
| `proximity` | 0.20 | Which team the nearest assigned player to the ball belongs to, when that player is within 6 m |
| `superiority` | 0.30 | (A − B) / (A + B) among the six observed assigned players nearest the ball (or camera focus), within 25 m, needing at least 4 |
| `territory` | 0.20 | Ball or focus position along the pitch signed by A's attacking direction; only when direction metadata is supplied |

The raw index is the weighted mean of the components that exist and needs at least 0.30 of the total possible weight. It is then smoothed with a support-weighted Gaussian (σ = 2.5 s) so single noisy seconds do not drive it, and **gaps stay gaps**: with less than 15% smoothed support the timeline shows "no reliable signal" instead of interpolating. Support is the fraction of the possible weight actually observed; low-support stretches are drawn dashed. A value near zero means balanced evidence, not missing evidence.

The **camera focus** is the pitch point under the image centre, from the calibrated homography. Broadcast cameras follow play, so it stands in for the ball's location when the ball is missing, but it is a proxy and every value that uses it says so. A detected ball only counts if it passes a **sanity gate** (an observed player within 15 m); isolated detections such as adverts or spots fall back to the camera focus. The coverage panel reports both the raw detection rate and the share that passes the gate.

The **tempo** lane is the mean speed of observed players, measured from one-second displacements and ignoring jumps above 12 m/s.

### Critical moments

Ranked by a score in [0, 1], with the evidence behind each one. Near-duplicates (same type within 4 s) are merged and the strongest 14 are kept. Tactic notes that overlap a moment are attached to it as "Shape" lines.

| Type | Trigger |
| --- | --- |
| Momentum swing | The smoothed index holds ±0.20 or beyond for one team, then flips to the other within 12 s, with combined peaks of at least 0.6 |
| Sustained pressure | The index stays at ±0.35 or beyond for at least 5 s |
| Control change | The pipeline's own possession-change event: 0.5 s of continuous observed control by the other team |
| Penalty-area play | The ball (or camera focus when the ball is missing) stays inside a penalty area for at least 1.5 s; the favoured team comes from observed control, else more visible players near the ball |
| Fast ball | Observed ball speed between 16 and 30 m/s across consecutive samples, a pass or shot candidate (faster is treated as a mapping glitch) |
| Crowding | Eight or more observed players within 8 m of the ball for at least 1.5 s |
| Tempo surge | Mean observed player speed at least 20% (and 0.5 m/s) above the clip median for 3 s or more |

Confidence is `medium` only when most of the supporting samples used the observed ball; otherwise `low`. Nothing is labelled "high": no event type here has been validated against independent labels.

### Tactic notes

A live **shape readout** for each team is always shown (depth × width of the visible group against the team's own clip median, tagged compact, typical or stretched). Event notes appear when something notable holds. All of them use the **visible group** of each team, meaning six or more observed, assigned outfield players; with fewer, geometry is withheld.

| Note | Rule |
| --- | --- |
| Compact / stretched | Visible-group depth at least 3 m below (compact, and width in the lowest 30%) or above (stretched) the team's own median and beyond its 30th/70th percentile, held for 3 s |
| Closing in / dropping away | The group's centre moves 12 m or more towards or away from the camera focus within 5 s (medians over 3 s). Measured relative to the play, because the visible group follows the camera and raw movement would mostly be panning |
| Big gap | Centres of the two visible groups 20 m or more apart for 3 s. A *small* gap is not reported: intermingled groups are normal |
| Overload | Two or more observed players more among the six nearest the ball than the opposition for 2 s |

Notes describe what is visible, never a full formation, defensive line, running load or intent.

## Data contract

`analysis.json` (`schema_version` 1) holds: `source` (video files, size, duration, model/video hashes, hardware and throughput), `pitch`, `teams`, `summary`, `coverage`, `momentum` (per-bin value, support, components, anchor), `moments`, `tactics`, `limitations` and `frames`. Each frame is `{t, s, e, p, b, c, h}`: time, shot index, eligible flag, people rows `[id, kind, x, y, observed, x1, y1, x2, y2]` (kind 0/1 team A/B, 2 unassigned player, 3 goalkeeper, 4 referee), ball `[x, y, observed, px, py, airborne]`, observed controlling team, and the image-to-pitch homography. `tests/test_viewer_data.py` checks the committed payload on every deploy.

## Limits worth knowing

- Ball recall is the weak link: the detector's held-out ball AP50 is 0.58 and a broadcast ball is a few pixels wide, so ball-based components are often missing and the index leans on the camera-focus proxy.
- Team assignment uses jersey colour only; goalkeepers and ambiguous kits stay unassigned.
- Wide/replay segmentation is a heuristic. A missed replay can inject non-live positions; excluded intervals are shown as such.
- One clip is a demonstration. Nothing here establishes accuracy on other matches, leagues or camera styles. Use the evaluation commands in [MATCH_ANALYSIS.md](MATCH_ANALYSIS.md) with independently reviewed labels before relying on any number.
