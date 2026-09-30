"""Human-only evaluation annotation; never imports or displays model predictions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import streamlit.components.v1 as components

from match_analysis.annotations import reviewed
from match_analysis.hashing import digest
from match_analysis.media import VideoServer

st.set_page_config(page_title="Independent match annotation", layout="wide")
p = argparse.ArgumentParser()
for name in ("video", "annotations"):
    p.add_argument("--" + name, required=True, type=Path)
a, _ = p.parse_known_args()
truth = json.loads(a.annotations.read_text())
st.title("Independent ground-truth review")
st.caption(
    "Watch the source video without model output. A/B must be mapped to actual jersey teams independently. Record the first supported observed-control change; leave uncertain intervals unlabelled."
)
reviewer = st.text_input("Human reviewer identifier")
team_a = st.text_input("Ground-truth team label for A", "left")
team_b = st.text_input("Ground-truth team label for B", "right")


@st.cache_resource
def transport():
    return VideoServer()


player = components.declare_component(
    "ground_truth_clock", path=str(Path(__file__).parent / "annotation_player")
)
state = st.session_state.setdefault(
    "review", {"intervals": [], "transitions": [], "start": None, "last_revision": None}
)
click = player(video_url=transport().register(a.video), key="annotation")
if click and click["revision"] != state["last_revision"]:
    state["last_revision"] = click["revision"]
    action = click["action"]
    t = click["time"]
    if action == "start":
        state["start"] = {"start": t, "view": click["view"]}
    elif action == "end" and state["start"]:
        state["intervals"].append({**state["start"], "end": t})
        state["start"] = None
    elif action in ("ab", "ba"):
        state["transitions"].append(
            {
                "timestamp": t,
                "from_team": team_a if action == "ab" else team_b,
                "to_team": team_b if action == "ab" else team_a,
            }
        )
st.write("Intervals")
st.dataframe(state["intervals"])
st.write("Observed changes")
st.dataframe(state["transitions"])
if st.button("Undo last interval") and state["intervals"]:
    state["intervals"].pop()
    st.rerun()
if st.button("Undo last change") and state["transitions"]:
    state["transitions"].pop()
    st.rerun()
complete = st.checkbox(
    "I reviewed the entire analyzed video and included every supported control change"
)
if reviewer and complete:
    try:
        result = reviewed(
            truth,
            state["intervals"],
            state["transitions"],
            reviewer,
            digest(a.annotations),
        )
        st.download_button(
            "Download reviewed ground truth",
            json.dumps(result, indent=2, allow_nan=False),
            file_name=truth["clip_id"] + ".reviewed.json",
            mime="application/json",
        )
    except ValueError as error:
        st.error(str(error))
else:
    st.info(
        "Ground truth is not certified until a human reviewer confirms complete review. Freeze the reviewed file into a new partition before temporal evaluation."
    )
