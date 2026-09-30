"""Local dashboard: streamlit run dashboard/app.py -- --run artifacts/match."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from match_analysis.hashing import digest
from match_analysis.media import VideoServer
from match_analysis.summary import load_frames

st.set_page_config(page_title="Football broadcast analysis", layout="wide")
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path)
args, _ = parser.parse_known_args()
st.title("Football match reconstruction")
run = Path(
    st.sidebar.text_input("Analysis directory", str(args.run or "artifacts/match"))
).resolve()
if not (run / "summary.json").is_file():
    st.info("Run the analyze command first, then select its output directory.")
    st.stop()
manifest = json.loads((run / "manifest.json").read_text())
summary = json.loads((run / "summary.json").read_text())
st.warning(
    "Experimental estimates. Predictions are visual aids. Missing/replay/uncertain observations do not contribute to statistics."
)
video = st.sidebar.text_input("Local MP4", manifest["video"], key="video_" + str(run))


@st.cache_data
def video_hash(path, size, modified):
    return digest(path)


source = Path(video)
if not source.is_file():
    st.error("Select the original local MP4 for this analysis.")
    st.stop()
stat = source.stat()
if (
    video_hash(str(source.resolve()), stat.st_size, stat.st_mtime_ns)
    != manifest["signature"]["video_sha256"]
):
    st.error("Video hash differs from this analysis. Select the original analyzed MP4.")
    st.stop()


@st.cache_resource
def transport():
    return VideoServer()


@st.cache_data
def window_frames(path, modified, window):
    # Read a bounded animation window. Raw video is streamed, never base64 encoded.
    lo = max(0, window * 60 - 10)
    hi = (window + 1) * 60
    return [f for f in load_frames(path) if lo <= f["timestamp"] < hi]


player = components.declare_component(
    "football_clock", path=str(Path(__file__).parent / "player")
)
config = manifest["signature"]["config"]
window = st.session_state.get("analysis_window", 0)
seek = st.session_state.get("analysis_seek", 0)
try:
    url = transport().register(video)
except ValueError as error:
    st.error(str(error))
    st.stop()
trends = json.loads((run / "trends.json").read_text())
events = json.loads((run / "events.json").read_text())
coaching = (
    json.loads((run / "coaching.json").read_text())
    if (run / "coaching.json").exists()
    else None
)
result = player(
    video_url=url,
    frames=window_frames(str(run), (run / "summary.json").stat().st_mtime, window),
    window_start=window * 60,
    window_end=(window + 1) * 60,
    duration=manifest["video_end"],
    length=config["pitch_length"],
    width=config["pitch_width"],
    events=events,
    coaching=coaching["records"] if coaching else [],
    trends=trends,
    seek=seek,
    key=str(run),
)
if result and result["window"] != window:
    st.session_state["analysis_window"] = result["window"]
    st.session_state["analysis_seek"] = result["time"]
    st.rerun()
cols = st.columns(4)
cols[0].metric("Eligible video", f"{summary['eligible_fraction']:.1%}")
cols[1].metric(
    "Possession coverage in eligible time", f"{summary['possession_coverage']:.1%}"
)
cols[2].metric("Unknown eligible time", f"{summary['unknown_eligible_seconds']:.1f}s")
cols[3].metric("Observed possession changes", summary["possession_changes"])
st.write("Covered possession only:", summary["observed_possession_percent"])
st.caption(
    "Team A/B are stable colour clusters; supplied team names require a verified cluster-to-name mapping. No player identity across cuts. Distances use supplied or assumed pitch dimensions."
)
if not any(row["A_territorial_pressure"] is not None for row in trends):
    st.info(
        "Attacking direction is unknown: territorial pressure is unavailable. Left/right occupancy remains available."
    )
else:
    st.line_chart(
        pd.DataFrame(trends).set_index("timestamp")[
            ["A_territorial_pressure", "B_territorial_pressure"]
        ]
    )
st.subheader("Observed player heatmaps")
maps = (
    dict(np.load(run / "heatmaps.npz"))
    if (run / "heatmaps.npz").exists()
    else {"A": np.zeros((14, 21)), "B": np.zeros((14, 21))}
)
cols = st.columns(2)
for column, team in zip(cols, ("A", "B")):
    fig, ax = plt.subplots(figsize=(7, 4))
    plot = ax.imshow(
        maps[team],
        origin="lower",
        extent=(0, config["pitch_length"], 0, config["pitch_width"]),
        cmap="magma",
    )
    ax.set(
        title=f"Team {team} — observed sample counts",
        xlabel="Pitch x (m)",
        ylabel="Pitch y (m)",
    )
    fig.colorbar(plot, ax=ax)
    column.pyplot(fig)
    plt.close(fig)
st.write(
    "Possession ball-zone seconds (left/middle/right thirds):",
    summary["zone_occupancy_seconds"],
)
st.subheader("Artifacts")
for name in (
    "possession.csv",
    "possession_diagnostics.json",
    "coaching.json",
    "coaching.csv",
    "tracks.csv",
    "events.csv",
    "events.json",
    "summary.json",
    "trends.json",
    "manifest.json",
):
    if not (run / name).exists():
        continue
    st.download_button(
        name,
        (run / name).read_bytes(),
        file_name=name,
        mime="text/csv" if name.endswith(".csv") else "application/json",
    )
with st.expander("Coverage and limitations"):
    st.json(summary)
