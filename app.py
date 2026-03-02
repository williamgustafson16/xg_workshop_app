#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:19:41 2026

@author: williamgustafson
"""

# app.py
# pip install streamlit pandas numpy matplotlib
# streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

st.set_page_config(page_title="Interactive xG Workshop (Fixed 2–1 Match)", layout="wide")

# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #D0E7F6;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


BASE_RATE = 0.0927
LOGIT_BASE = np.log(BASE_RATE / (1 - BASE_RATE))

INDUSTRY_PRE = [
    "Distance to goal",
    "Angle",
    "Height of ball",
    "Pressure from opponent",
    "Opponents in front of shot",
    "GK positioning",
    "Header",
    "Open-play",
    "First touch",
    "Previous pass location",
]
INDUSTRY_POST = ["Speed of shot", "Placement of shot"]

OTHER = [
    "Player quality",
    "GK quality",
    "Wind",
    "Rain",
    "Ball spin",
    # "Strong foot",
    "Foot placement on ball",
    "GK vision of ball",
    "GK reaction time",
    "GK Jump Ability",
    "GK Size",
    "Player emotional state",
    "GK emotional state",
    "Defenders movement post-shot",
]

FEATURE_MAP = {
    "Distance to goal": "dist",
    "Angle": "angle",
    "Height of ball": "height_of_ball",
    "Pressure from opponent": "pressure",
    "Opponents in front of shot": "opponents_in_front",
    # "Strong foot": "strong_foot",
    "GK positioning": "gk_positioning",
    "Speed of shot": "speed_of_shot",
    "Placement of shot": "placement_of_shot",
}
FEATURE_MAP["Header"] = "is_header"
FEATURE_MAP |= {
    "Open-play": "open_play",
    "First touch": "first_touch",
    "Previous pass location": "prev_pass_loc",
}


# made-up coefficients on standardized features
COEFS = {
    "dist": -0.95,
    "angle": 0.90,
    "height_of_ball": -0.35,
    "pressure": -0.45,
    "opponents_in_front": -0.55,
    "gk_positioning": -0.30,
    "speed_of_shot": 0.35,
    "placement_of_shot": 0.95,
}
COEFS |= {
    "open_play": 0.18,
    "first_touch": 0.22,
    "prev_pass_loc": 0.35,
}

# fixed standardization for workshop
SCALE = {
    "dist": (18.0, 8.0),
    "angle": (0.55, 0.25),
    "height_of_ball": (0.45, 0.35),
    "pressure": (0.30, 0.46),
    "opponents_in_front": (2.0, 1.4),
    # "strong_foot": (0.60, 0.49),
    "gk_positioning": (0.0, 1.0),
    "speed_of_shot": (24.0, 6.0),
    "placement_of_shot": (0.55, 0.25),
}
SCALE |= {
    "open_play": (0.78, 0.41),       # binary-ish
    "first_touch": (0.28, 0.45),     # binary-ish
    "prev_pass_loc": (0.55, 0.22),   # 0..1 quality
}


COEFS["is_header"] = -1.10   # strong penalty when chosen
SCALE["is_header"] = (0.20, 0.40)
# calibration targets (when ALL pre-shot selected, no post-shot):
# Home xG ≈ 1.1, Away xG ≈ 1.7 (14 shots each)
K_PRE = 0.50
K_POST = 0.45  # just a reasonable spread when post-shot is added
TEAM_OFFSETS = {"Home": -1.2148816665078983, "Away": -0.42867115195744954}





OTHER_FEATURE_MAP = {
    "Player quality": "player_quality",
    "GK quality": "gk_quality",
    "Wind": "wind",
    "Rain": "rain",
    "Ball spin": "ball_spin",
    # "Strong foot": "strong_foot",
    "Foot placement on ball": "foot_placement",
    "GK vision of ball": "gk_vision",
    "GK reaction time": "gk_reaction_time",
    "GK Jump Ability": "gk_jump",
    "GK Size": "gk_size",
    "Player emotional state": "player_emotion",
    "GK emotional state": "gk_emotion",
    "Defenders movement post-shot": "def_move_post",
}


# map labels -> cols
FEATURE_MAP |= {k: v for k, v in OTHER_FEATURE_MAP.items()}

# coefficients (standardized). signs chosen to make sense.
COEFS |= {
    "player_quality":  0.35,
    "gk_quality":     -0.35,
    "wind":           -0.12,
    "rain":           -0.10,
    "ball_spin":       0.10,
    # "strong_foot": 0.25,
    "foot_placement":  0.55,
    "gk_vision":      -0.45,
    "gk_reaction_time": 0.35,   # higher = slower => higher xG
    "gk_jump":        -0.12,
    "gk_size":        -0.10,
    "player_emotion":  0.10,
    "gk_emotion":     -0.08,
    "def_move_post":  -0.18,
}

# scaling for 0..1 features
SCALE |= {col: (0.5, 0.25) for col in OTHER_FEATURE_MAP.values()}






def add_other_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # deterministic pseudo-random in [0,1] from shot_id (no RNG)
    u = ((df["shot_id"] * 1103515245 + 12345) % 2**31) / 2**31
    u2 = ((df["shot_id"] * 1664525 + 1013904223) % 2**32) / 2**32

    # some sensible, deterministic signals
    # (note: these are "potential" — not real physics — just workshop knobs)
    team_boost = (df["team"].eq("Home").astype(float) * 0.05)  # tiny home edge
    goal = df["goal"].astype(float)

    df["player_quality"] = (0.55 + 0.25*u + team_boost).clip(0, 1)
    df["gk_quality"] = (0.55 + 0.25*u2 - team_boost).clip(0, 1)

    df["wind"] = (0.2 + 0.6*((df["minute"] % 30) / 30)).clip(0, 1)
    df["rain"] = (0.1 + 0.8*(df["minute"] >= 60).astype(float)).clip(0, 1)

    df["ball_spin"] = (0.25 + 0.6*u2).clip(0, 1)
    df["foot_placement"] = (0.30 + 0.65*u).clip(0, 1)

    df["gk_vision"] = (0.70 - 0.10*df["opponents_in_front"]/5 - 0.10*df["rain"]).clip(0, 1)
    df["gk_reaction_time"] = (0.55 + 0.35*df["rain"] + 0.10*u2).clip(0, 1)   # higher = slower
    df["gk_jump"] = (0.55 + 0.35*u).clip(0, 1)
    df["gk_size"] = (0.55 + 0.35*u2).clip(0, 1)

    df["player_emotion"] = (0.50 + 0.35*(df["team"].eq("Away").astype(float)) + 0.15*u).clip(0, 1)
    df["gk_emotion"] = (0.50 + 0.15*u2).clip(0, 1)
    # df["strong_foot"] = (0.30 + 0.65*u).clip(0, 1)

    # "Defenders movement post-shot" (bigger => more disruption)
    df["def_move_post"] = (0.25 + 0.10*df["pressure"] + 0.10*df["opponents_in_front"]/5 + 0.55*u).clip(0, 1)

    # crucial: make "other" features *informative* about outcome deterministically,
    # but not a direct copy; we bias a few that plausibly relate to conversion:
    # better foot placement + worse GK vision => more likely goal (softly)
    df["foot_placement"] = (df["foot_placement"] + 0.25*goal).clip(0, 1)
    df["gk_vision"] = (df["gk_vision"] - 0.25*goal).clip(0, 1)

    return df

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@st.cache_data
def fixed_match():
    # 28 shots total (14 each), score 2–1.
    # One Home goal (shot 21) is from outside the box (dist=22).
    rows = [
    # id, min, team, player, x, y, dist, angle, h, press, opp, strong, gk, header, speed, place, goal
    ( 1,  3, "Home", "P9",  93, 36, 14, 0.85, 0.20, 0, 1, -0.4, 0, 26, 0.70, 0),
    ( 2,  5, "Away", "P11", 88, 30, 20, 0.55, 0.40, 1, 3,  0.2, 0, 22, 0.45, 0),
    ( 3, 10, "Home", "P7",  96, 33, 10, 1.05, 0.15, 0, 0, -0.8, 1, 28, 0.82, 1),  # header goal
    ( 4, 12, "Away", "P8",  92, 44, 16, 0.70, 0.25, 0, 1, -0.1, 0, 24, 0.62, 0),
    ( 5, 16, "Home", "P10", 86, 26, 23, 0.40, 0.35, 1, 4,  0.4, 0, 20, 0.40, 0),
    ( 6, 21, "Away", "P9",  97, 34,  9, 1.10, 0.10, 0, 1, -0.6, 0, 27, 0.76, 1),
    ( 7, 25, "Home", "P6",  90, 20, 20, 0.52, 0.55, 1, 2,  0.3, 0, 23, 0.55, 0),
    ( 8, 28, "Away", "P7",  84, 40, 25, 0.32, 0.70, 1, 5,  0.6, 0, 18, 0.35, 0),
    ( 9, 32, "Home", "P11", 94, 28, 13, 0.90, 0.30, 0, 1, -0.5, 0, 25, 0.60, 0),
    (10, 35, "Away", "P10", 89, 36, 18, 0.60, 0.35, 0, 2,  0.1, 0, 21, 0.50, 0),
    (11, 38, "Home", "P8",  80, 34, 28, 0.25, 0.20, 1, 4,  0.5, 0, 19, 0.30, 0),
    (12, 41, "Away", "P6",  95, 50, 14, 0.65, 0.25, 0, 1, -0.3, 0, 26, 0.55, 0),
    (13, 44, "Home", "P9",  99, 34,  7, 1.20, 0.10, 0, 0, -1.0, 1, 30, 0.88, 0),  # header
    (14, 48, "Away", "P8",  87, 18, 23, 0.38, 0.45, 1, 3,   0.4, 0, 20, 0.40, 0),
    (15, 52, "Home", "P7",  92, 40, 16, 0.70, 0.40, 1, 2,   0.2, 0, 24, 0.62, 0),
    (16, 55, "Away", "P11", 91, 32, 17, 0.62, 0.30, 0, 2,   0.3, 0, 23, 0.50, 0),
    (17, 59, "Home", "P10", 96, 38, 11, 0.95, 0.25, 0, 1,  -0.7, 0, 27, 0.75, 0),
    (18, 62, "Away", "P9",  83, 34, 27, 0.28, 0.60, 1, 4,   0.7, 0, 18, 0.30, 0),
    (19, 67, "Home", "P6",  88, 46, 20, 0.50, 0.45, 1, 3,   0.4, 0, 22, 0.52, 0),
    (20, 71, "Away", "P7",  94, 34, 13, 0.88, 0.15, 0, 1,  -0.4, 0, 25, 0.65, 0),
    (21, 76, "Home", "P11", 83, 33, 22, 0.35, 0.20, 0, 2,   0.2, 0, 30, 0.92, 1),  # outside box goal
    (22, 81, "Away", "P10", 90, 42, 18, 0.58, 0.35, 0, 2,   0.2, 0, 23, 0.48, 0),
    (23, 84, "Home", "P8",  85, 30, 24, 0.35, 0.40, 1, 4,   0.6, 0, 19, 0.35, 0),
    (24, 90, "Away", "P6",  98, 34,  8, 1.15, 0.10, 0, 1,  -0.8, 1, 28, 0.70, 0),  # header
    (25,  8, "Away", "P5",  82, 22, 28, 0.22, 0.55, 1, 4,   0.9, 0, 19, 0.28, 0),
    (26, 30, "Away", "P3",  93, 40, 15, 0.75, 0.25, 0, 1,  -0.2, 0, 25, 0.60, 0),
    (27, 19, "Home", "P4",  91, 36, 17, 0.62, 0.30, 0, 2,   0.1, 0, 24, 0.50, 0),
    (28, 60, "Home", "P2",  79, 40, 29, 0.20, 0.50, 1, 5,  0.8, 0, 18, 0.25, 0),
    (29, 22, "Home", "P5",  94, 30, 15, 0.80, 0.25, 0, 1,  -0.3, 0, 26, 0.65, 0),
    (30, 74, "Home", "P3",  88, 42, 19, 0.55, 0.35, 1, 3,  0.4, 0, 22, 0.48, 0),
    
    (31, 15, "Away", "P4",  91, 28, 17, 0.62, 0.30, 0, 2,  0.2, 0, 24, 0.55, 0),
    (32, 83, "Away", "P2",  85, 36, 23, 0.40, 0.45, 1, 4,  0.6, 0, 19, 0.35, 0),
    ]
    
    
    df = pd.DataFrame(
        rows,
        columns=[
          "shot_id","minute","team","player","x","y","dist","angle","height_of_ball",
          "pressure","opponents_in_front","gk_positioning",
          "is_header",
          "speed_of_shot","placement_of_shot","goal",
        ],
    )
    
    # --- derived / fixed "event-like" features (deterministic) ---
    # open play: most shots open play, a few not
    df["open_play"] = (~df["shot_id"].isin([14, 22, 25])).astype(int)
    
    # first touch: some shots are first-time finishes (incl. a header)
    df["first_touch"] = (df["shot_id"].isin([3, 6, 13, 17, 24, 26])).astype(int)
    
    # previous pass location quality proxy: 0..1 (higher = better preceding pass)
    # (uses shot x/y so it feels coherent; deterministic)
    px = (df["x"] / 105.0)
    py = 1.0 - (np.abs(df["y"] - 34.0) / 34.0)
    df["prev_pass_loc"] = (0.55 * px + 0.45 * py).clip(0, 1)
    
    
    
    away = df["team"] == "Away"
    home = ~away
    
    # Away creates better chances (pre-shot)
    df.loc[away, "dist"] *= 0.92                      # closer
    df.loc[away, "angle"] *= 1.10                     # better angles
    df.loc[away, "pressure"] = np.minimum(df.loc[away, "pressure"] + 0, 1)
    df.loc[away, "opponents_in_front"] = np.clip(df.loc[away, "opponents_in_front"] - 1, 0, 5)
    
    # Home creates slightly worse chances (pre-shot)
    df.loc[home, "dist"] *= 1.04
    df.loc[home, "angle"] *= 0.95
    df.loc[home, "opponents_in_front"] = np.clip(df.loc[home, "opponents_in_front"] + 1, 0, 5)
    
    # Home finishes better (post-shot), Away finishes worse (post-shot)
    df.loc[home, "speed_of_shot"] *= 1.08
    df.loc[home, "placement_of_shot"] = np.clip(df.loc[home, "placement_of_shot"] * 1.15, 0, 1)
    
    df.loc[away, "speed_of_shot"] *= 0.92
    df.loc[away, "placement_of_shot"] = np.clip(df.loc[away, "placement_of_shot"] * 0.85, 0, 1)
    
    # keep angle in bounds
    df["angle"] = df["angle"].clip(0, 1.6)
    
    df = add_other_features(df)


    return df

def explained_variance_proxy(pre_selected, post_selected, other_selected):
    base = 0.19 * (len(pre_selected) / len(INDUSTRY_PRE)) + 0.09 * (len(post_selected) / len(INDUSTRY_POST))
    other_frac = len(other_selected) / len(OTHER)
    return base + (0.98 - base) * other_frac

def _zscore(series, mu, sd):
    return (series.to_numpy(dtype=float) - mu) / sd



TARGET_TOTAL_XG = 3  # 2.5956
K_OTHER_MAX = 2.2                # controls how strongly "Other" separates shots

def _solve_global_shift(z, target_sum, iters=35):
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = sigmoid(z + mid).sum()
        if s < target_sum:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def compute_xg(df, pre_selected, post_selected, other_selected):
    if (len(pre_selected) + len(post_selected) + len(other_selected)) == 0:
        return np.full(len(df), BASE_RATE, dtype=float)

    z = np.full(len(df), LOGIT_BASE, dtype=float)
    pre_term = np.zeros(len(df))
    for lab in pre_selected:
        col = FEATURE_MAP[lab]
        mu, sd = SCALE[col]
        pre_term += COEFS[col] * _zscore(df[col], mu, sd)

    post_term = np.zeros(len(df))
    for lab in post_selected:
        col = FEATURE_MAP[lab]
        mu, sd = SCALE[col]
        post_term += COEFS[col] * _zscore(df[col], mu, sd)

    other_term = np.zeros(len(df))
    for lab in other_selected:
        col = FEATURE_MAP[lab]
        mu, sd = SCALE[col]
        other_term += COEFS[col] * _zscore(df[col], mu, sd)

    # base model
    z_model = z + K_PRE * pre_term + K_POST * post_term + 0.8 * other_term

    other_frac = len(other_selected) / len(OTHER) if len(OTHER) else 0.0

    if other_frac > 0:
        # deterministic "truth pull": as more OTHER selected -> probs -> {0.0024, 0.98}
        p_target = 0.0024 + (0.98 - 0.0024) * df["goal"].to_numpy(dtype=float)
        logit_target = np.log(p_target / (1 - p_target))

        w = other_frac ** 1.2  # ramp-up
        z_model = (1 - w) * z_model + w * logit_target

    # keep total ~ 3.0 for any selection combo (except the "no params" special-case above)
    shift = _solve_global_shift(z_model, TARGET_TOTAL_XG)
    return sigmoid(z_model + shift)


def plot_pitch(df, xg_col="xg", title="Shots sized by xG"):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 105); ax.set_ylim(0, 68)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)

    line = dict(color="black", lw=2)

    ax.plot([0,105,105,0,0],[0,0,68,68,0], **line)
    ax.plot([52.5,52.5],[0,68], **line)
    ax.add_patch(plt.Circle((52.5,34), 9.15, fill=False, **line))
    ax.add_patch(plt.Circle((52.5,34), 0.8, color="black"))

    ax.plot([16.5,16.5],[34-20.16,34+20.16], **line)
    ax.plot([0,16.5],[34-20.16,34-20.16], **line)
    ax.plot([0,16.5],[34+20.16,34+20.16], **line)

    ax.plot([105-16.5,105-16.5],[34-20.16,34+20.16], **line)
    ax.plot([105,105-16.5],[34-20.16,34-20.16], **line)
    ax.plot([105,105-16.5],[34+20.16,34+20.16], **line)

    ax.plot([5.5,5.5],[34-9.16,34+9.16], **line)
    ax.plot([0,5.5],[34-9.16,34-9.16], **line)
    ax.plot([0,5.5],[34+9.16,34+9.16], **line)

    ax.plot([105-5.5,105-5.5],[34-9.16,34+9.16], **line)
    ax.plot([105,105-5.5],[34-9.16,34-9.16], **line)
    ax.plot([105,105-5.5],[34+9.16,34+9.16], **line)

    ax.add_patch(plt.Circle((11,34), 0.8, color="black"))
    ax.add_patch(plt.Circle((105-11,34), 0.8, color="black"))

    dfp = df.copy()
    away = dfp["team"] == "Away"
    dfp.loc[away, "x"] = 105 - dfp.loc[away, "x"]


    home = dfp[dfp["team"] == "Home"]
    away = dfp[dfp["team"] == "Away"]
    
    ax.scatter(home["x"], home["y"],
               s=60 + 1200*home[xg_col],
               alpha=0.78,
               marker="o",
               color="#1f77b4",
               edgecolors="black")
    
    ax.scatter(away["x"], away["y"],
               s=60 + 1200*away[xg_col],
               alpha=0.78,
               marker="o",
               color="#d62728",
               edgecolors="black")
    
    # goal markers
    hg = home[home["goal"] == 1]
    ag = away[away["goal"] == 1]
    ax.scatter(hg["x"], hg["y"], s=260, marker="*", color="#1f77b4", edgecolors="black")
    ax.scatter(ag["x"], ag["y"], s=260, marker="*", color="#d62728", edgecolors="black")
    
    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Home',
               markerfacecolor='#1f77b4', markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Away',
               markerfacecolor='#d62728', markeredgecolor='black', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Goal',
               markerfacecolor='black', markeredgecolor='black', markersize=10),
    ]
    
    ax.legend(handles=legend_elements, loc="lower left", frameon=False)
    return fig


# ---------------- UI ----------------
df = fixed_match().copy()
home_goals = int(df.loc[(df.team == "Home") & (df.goal == 1)].shape[0])
away_goals = int(df.loc[(df.team == "Away") & (df.goal == 1)].shape[0])


st.markdown(
    "<h1 style='text-align: center;'>Interactive xG Workshop</h1>",
    unsafe_allow_html=True
)
import base64

def add_logo(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="position:absolute; top:-80px; right:5px;">
            <img src="data:image/png;base64,{data}" width="220">
        </div>
        """,
        unsafe_allow_html=True
    )

add_logo("logo.png")

with st.sidebar:
    st.header(f"Score: Home {home_goals}–{away_goals} Away")
    st.caption("No parameters selected ⇒ all shots xG = 0.0927.")
    st.divider()

    st.markdown("### Industry Standard Parameters")

    st.subheader("Pre-shot parameters")
    pre_selected = [lab for lab in INDUSTRY_PRE if st.checkbox(lab, value=False, key=f"pre_{lab}")]

    st.subheader("Post-shot parameters")
    post_selected = [lab for lab in INDUSTRY_POST if st.checkbox(lab, value=False, key=f"post_{lab}")]

    st.divider()
    st.markdown("### Other Potential Parameters")
    other_selected = [lab for lab in OTHER if st.checkbox(lab, value=False, key=f"other_{lab}")]

print(df.columns)
df["xg"] = compute_xg(df, pre_selected, post_selected, other_selected)
ev = explained_variance_proxy(pre_selected, post_selected, other_selected)

home_xg = df.loc[df.team == "Home", "xg"].sum()
away_xg = df.loc[df.team == "Away", "xg"].sum()

# c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("Home xG", f"{home_xg:.2f}")
c2.metric("Away xG", f"{away_xg:.2f}")
c3.metric("Shots (H/A)", f"{(df.team=='Home').sum()} / {(df.team=='Away').sum()}")
# c4.metric("Explained variance", f"{100*ev:.1f}%")

left, right = st.columns([1.1, 0.9])
with left:
    st.pyplot(plot_pitch(df, "xg"))
with right:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.hist(df["xg"], bins=18, alpha=0.8)
    ax.set_title("xG distribution")
    ax.set_xlabel("xG"); ax.set_ylabel("count")
    st.pyplot(fig)

st.subheader("Shots table")
cols = [
    "shot_id","minute","team","player","goal","xg",
    "dist","angle","height_of_ball","pressure","opponents_in_front","gk_positioning",
    "speed_of_shot","placement_of_shot",
]
st.dataframe(df[cols].sort_values(["minute","shot_id"]), use_container_width=True)






