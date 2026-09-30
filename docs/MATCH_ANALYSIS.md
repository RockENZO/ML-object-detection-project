# Automatic football broadcast analysis (experimental)

## Scope and commands

The analysis is offline and automatic. The local dashboard requires no detector, calibration model or GPU. IDs belong to a camera shot; these are estimated spatial/control statistics, not official passes, shots, tackles, fouls, xG or named-player statistics.

Use Python 3.12. Install the platform-appropriate PyTorch/torchvision wheels first (the recorded development runtime used torch 2.14.0/torchvision 0.29.0). Colab/Kaggle can retain their provided compatible CUDA runtime; its actual versions are recorded in the run manifest and checked on resume. Then:

```bash
python -m pip install -r requirements-analysis.txt
python tools/setup_pitch.py
python -m match_analysis analyze \
  --video input_videos/08fd33_4.mp4 \
  --checkpoint runs/performance-study/improved/weights/best.pt \
  --config configs/match.json --device 0 --output artifacts/match
# Repeat with --resume after interruption, using identical arguments/code.
python -m pip install -r requirements-dashboard.txt
python -m streamlit run dashboard/app.py -- --run artifacts/match
```

Supply the previously trained four-class checkpoint; generic COCO weights are rejected. The improved checkpoint is a local training artifact, not downloaded automatically. CPU and Apple `--device mps` are supported; use `--device 0` on cloud CUDA. The notebook at `notebooks/football_analysis_colab.ipynb` contains the same commands, durable-output setup and an actual-GPU benchmark step. Kaggle uses identical commands and outputs.

The dashboard is designed for a **local browser and local Streamlit server**. It streams the registered MP4 through a loopback HTTP server with byte-range support. Select the local MP4 if cloud paths differ. Its video element is the sole playback clock: video seeking drives pitch positions, trajectories, coverage and the trend cursor. Event buttons seek to the supporting uncertainty interval. Animation payloads cover a 60-second window plus 10 seconds of trail history. Source MP4 must use a browser-supported codec; H.264 is recommended.

## Inputs, coordinates and evidence

Original decoded presentation timestamps are retained, sampled at 10 Hz (a 25 Hz source alternates 80/120 ms sample gaps). Raw video is never loaded in full. Hashes cover video, checkpoint, configuration, metadata, analysis source files and the pitch-asset manifest. Camera-shot files are atomically committed before checkpoints. An interruption reprocesses the active shot. Completed shots and match-level colour centres survive recovery. Resume rejects changed inputs/settings/code; keep the recorded runtime environment for repeatability.

Default pitch dimensions are 105 x 68 metres. Coordinates start at one pitch corner, with x along the length. Real distances remain approximate unless actual dimensions are supplied. `configs/match_metadata.example.json` shows attacking-direction intervals in **video seconds**, not scoreboard time; replace its example boundaries with real halftime/direction boundaries. A/B are colour clusters, not automatically recognized team names. Only supply names after verifying the cluster mapping. Missing direction disables territorial pressure.

PnLCalib source commit and SHA-256 of both single-view weights are pinned in `match_analysis/pnl_assets.json`. The subprocess isolates its `utils` and `model` modules from this repository. It evaluates landmarks at 2 Hz, propagates calibration with forward/backward optical flow, rejects poor/incomplete fits and produces missing pitch positions on failure. People use BoT-SORT; ball observations use an independent velocity filter. Goalkeepers and ambiguous jerseys remain unassigned. Central torso colour crops preserve green kits.

Scene cuts use HSV-histogram change; grass fraction/person counts nominate wide shots. Repeated image hashes flag replay candidates, with uncertainty excluded immediately. **These are unvalidated segmentation heuristics**, not a trained broadcast/replay model. Same-colour cuts and never-before-seen replay views can be missed; static sequences can be excluded incorrectly. Airborne-ball uncertainty is likewise a conservative image heuristic. These limitations remain release risks requiring independently annotated broadcasts.

## Continuity, events and statistical denominators

- Hollow/dashed marks are velocity predictions, capped at one second in the same eligible calibrated shot. Cuts, excluded views and calibration failure clear motion history.
- Predictions never supply possession or heatmap/zone observations. Missing ball, airborne uncertainty, unassigned nearest player or ambiguous proximity produces unknown control.
- Observed control needs 0.5 seconds of persistence. A change requires continuous observed evidence from Team A to B or B to A in an uninterrupted eligible shot. Missing segments reset the chain. The event timestamp is the first supported competing-team observation, with the interval since the last supported prior-team observation and a later confirmation time.
- Possession percentage denominator is **covered eligible control time**. Eligible unknown duration and excluded duration are reported separately. A high conditional possession percentage with little coverage cannot represent the match.
- Ball-zone occupancy uses observed possession in left/middle/right thirds. Player-zone observation seconds and heatmaps count observed, assigned, eligible positions only; offscreen players are absent.
- Territorial pressure is each team's share of rolling 30-second covered possession in its opponent's half. It requires known direction throughout the covered window. Unknown/excluded time remains visible.

## Artifact contract

| File | Contents |
| --- | --- |
| `manifest.json` | Input/model/config/code/asset hashes, runtime, original video path/end, actual throughput, experimental status |
| `checkpoint.json` | Last durable shot boundary, colour centres, replay history, processing counters |
| `shots/000000.jsonl` | Original timestamp/frame, view/cut eligibility, calibration H/status, shot-local boxes/IDs/team, observed/predicted positions, ball, possession, event |
| `tracks.csv` | Timestamped roles, IDs, team, coordinates and observation flags |
| `events.json`, `events.csv` | Change timestamps, team direction, supporting uncertainty and confirmation time |
| `summary.json` | Covered/unknown/excluded denominators, conditional possession, counts, zones, limitations |
| `trends.json` | 30-second covered possession and territorial-pressure series |
| `heatmaps.npz` | Assigned observed player sample counts in a 21 x 14 grid |

## Public benchmark preparation

SoccerNet [GSR](https://github.com/SoccerNet/sn-gamestate) provides spatial/team/tracking labels. The [ball-action task](https://www.soccer-net.org/tasks/ball-action-spotting) provides temporal action labels; those **do not label possession changes**. Obtain permitted footage and independently review/annotate eligible/replay intervals and control-change events before final evaluation. Never convert a shot/pass/foul action label into a possession-change label by assumption.

A pinned public GSR sample can be downloaded without its entire 11 GB validation archive:

```bash
python tools/download_gsr_sample.py --clip SNGS-033 --split valid --output artifacts/gsr
ffmpeg -framerate 25 -i artifacts/gsr/SNGS-033/img1/%06d.jpg \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart artifacts/gsr/SNGS-033.mp4
python tools/convert_gsr.py --labels artifacts/gsr/SNGS-033/Labels-GameState.json \
  --video artifacts/gsr/SNGS-033.mp4 --output artifacts/gsr/annotations.json
```

The converter preserves source match `info.game_id`, requires label version >=1.3 and shifts centred GSR coordinates by +52.5,+34. Outside-pitch coordinates are retained for image matching but omitted from spatial error. The re-encoded MP4 hash and source annotation hash must accompany results. Public sample data are not committed here.

Create `sources.json` containing records with verified `match_id`, `clip_id`, `video` and `annotations` paths. Then freeze **before evaluation or tuning**:

```bash
python -m match_analysis freeze --sources sources.json --output partitions.json
python -m match_analysis analyze --video artifacts/gsr/SNGS-033.mp4 \
  --checkpoint /path/to/best.pt --device 0 --output artifacts/gsr-run
python -m match_analysis evaluate --run artifacts/gsr-run \
  --annotations artifacts/gsr/annotations.json --partition partitions.json --output evaluation.json
```

Deterministic source-match hashes allocate development (60%), validation (20%) and evaluation (20%) groups. All clips from one match stay together. Existing partition/evaluation files cannot be overwritten. Hashes catch mutated video/labels. Expand the source manifest to enough independent matches in all partitions before certifying a release. A single public development clip is a diagnostic, never a final benchmark. Freeze event annotations after human review; keep final evaluation matches untouched by tuning.

### Annotation schema and metrics

Annotation JSON contains `clip_id`, `match_id`, `frames` and optionally `transitions`:

```json
{
  "clip_id": "source-clip", "match_id": "verified-source-match",
  "team_mapping": {"A": "home", "B": "away"},
  "frames": [
    {"frame_index": 0, "timestamp": 0.0, "eligible": true, "replay": false,
     "people": [{"id": "gt-1", "role": "player", "team": "home", "bbox": [100,100,130,170], "xy": [20,34]}],
     "ball": {"bbox": [110,165,116,171]}}
  ],
  "transitions": [{"timestamp": 12.4, "from_team": "A", "to_team": "B"}]
}
```

This is schema illustration, not match evidence. Frame indices must match the sampled source grid. Omit unlabeled fields instead of filling assumed ground truth. `ball:null` means genuinely absent, rather than unannotated. Include **all** reviewed transitions over the analyzed interval to measure precision/recall. Anonymous A/B team evaluation may explicitly set `team_label_permutation_invariant:true`; this evaluates clustering up to a global label swap, not correct named-team identification. Keep names/directions mapped from independently verified metadata.

Tracking IDF1 uses global identity assignment from compatible same-role image boxes at IoU >=0.5. Team accuracy uses assigned matched players and reports assignment coverage. Median player pitch error reports mapped-match coverage. Raw detector and filtered observed-ball image precision/recall use IoU >=0.5 and are reported separately. Segmentation confusion counts and possession coverage require independent eligible/replay labels. Possession changes use one-to-one team-direction matching within +/-1 second, reporting precision, recall and timestamp MAE. This is the documented repository protocol, **not official SoccerNet GS-HOTA**. Aggregate independent source matches, report abstention/missing coverage and inspect failures.

Release targets: IDF1 >=0.75; assigned team accuracy >=90%; median player pitch error <=2 m; change precision >=0.85 and recall >=0.70 at +/-1 second. Missing or unmet metrics keep the release experimental. Do not update resume/profile accuracy or full-match runtime claims until independent results support them.

## Verification and outstanding gates

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Tests exercise cut/replay exclusion, ambiguous/unassigned teams, missing/airborne balls, direction changes, one-second expiry, unknown calibration, unique temporal matching, predictions excluded from summaries, interrupted-shot recovery, mutated input rejection and MP4 byte-range seeking. Browser verification checks real playback seeking and clock synchronization. Synthetic fixtures are only behavioral tests and never benchmark results.

Actual cloud throughput is a separate gate: allocate a Colab/Kaggle GPU, run the notebook's 30-second benchmark and publish its manifest/GPU type before projecting full-match time. Current local GPU measurements cannot substitute for this. Human-verified temporal/segmentation labels and held-out evaluation results are also still required.

## Measured development result (2026-09-30)

On public GSR SNGS-033 (one 30-second development clip, not final evaluation): IDF1 **0.539**; assigned team accuracy **96.9% at 40.5% assignment coverage**; median player pitch error **0.62 m at 83.9% projection coverage on matched players**. Raw ball precision/recall were **18.4%/12.9%**; filtered observed-ball precision/recall were **45.2%/6.6%**. Observed possession coverage was **0%** on that clip, so no transition accuracy claim is supported.

The existing 30-second smoke video processed 300 samples in **43.85 seconds** on an Apple M4 Pro using MPS (startup included, 6.84 sampled frames/s). This is a local measurement; cloud throughput and full-match time remain unmeasured. The complete hashes, actual durations, failure examples and experimental gates are in [match_analysis_results.json](match_analysis_results.json). The pipeline can be demonstrated as an engineering implementation; accuracy claims must await the missing independent evaluation gates.
