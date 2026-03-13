import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Scout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    .tag-pl  { background:#1e3a5f; color:#7db8f7; padding:3px 10px;
               border-radius:99px; font-size:12px; font-weight:500; }
    .tag-l1  { background:#0f3d22; color:#6ee7a0; padding:3px 10px;
               border-radius:99px; font-size:12px; font-weight:500; }
    .tag-pos { background:#2a2a2a; color:#aaaaaa; padding:3px 10px;
               border-radius:99px; font-size:12px; }

    .kpi-card {
        background: #1e1e2e;
        border: 1px solid #333355;
        border-radius: 12px;
        padding: 16px 14px 12px;
        text-align: center;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-label {
        font-size: 11px;
        color: #888;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-orig {
        font-size: 12px;
        color: #666;
        margin-bottom: 2px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #f0f0f0;
        line-height: 1.1;
    }
    .kpi-delta-pos {
        font-size: 12px;
        color: #6ee7a0;
        margin-top: 4px;
    }
    .kpi-delta-neg {
        font-size: 12px;
        color: #f87171;
        margin-top: 4px;
    }
    .kpi-delta-neu {
        font-size: 12px;
        color: #888;
        margin-top: 4px;
    }
    .stat-section-title {
        font-size: 13px;
        font-weight: 600;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 16px 0 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
FEATURES = [
    "Goals p 90", "Assists p 90", "Shots p 90", "% Shots on target",
    "Pass completion %", "Progressive Passes", "Progressive Carries",
    "Interceptions", "Tackles Won", "Key passes",
    "% Successful take-ons", "Shot creating actions p 90",
    "Goal creating actions p 90",
]

FEATURE_LABELS = {
    "Goals p 90"                 : "Goals / 90",
    "Assists p 90"               : "Assists / 90",
    "Shots p 90"                 : "Shots / 90",
    "% Shots on target"          : "Shot accuracy %",
    "Pass completion %"          : "Pass completion %",
    "Progressive Passes"         : "Progressive passes",
    "Progressive Carries"        : "Progressive carries",
    "Interceptions"              : "Interceptions",
    "Tackles Won"                : "Tackles won",
    "Key passes"                 : "Key passes",
    "% Successful take-ons"      : "Dribble success %",
    "Shot creating actions p 90" : "Shot-creating / 90",
    "Goal creating actions p 90" : "Goal-creating / 90",
}

ATTACKING  = ["Goals p 90", "Assists p 90", "Shots p 90", "% Shots on target",
               "Shot creating actions p 90", "Goal creating actions p 90"]
POSSESSION = ["Pass completion %", "Progressive Passes",
               "Progressive Carries", "Key passes", "% Successful take-ons"]
DEFENSIVE  = ["Interceptions", "Tackles Won"]


# ── Generator architecture ────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, input_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Load dataset ──────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    base = Path(__file__).parent
    data_path = base / "data"
    dfs = []
    for f in sorted(data_path.glob("cleaned_*.csv")):
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["Matches Played"] >= 5].copy()

    combined["player"] = combined["player"].astype(str).str.strip()
    combined["comp"]   = combined["comp"].astype(str).str.strip()
    combined["squad"]  = combined["squad"].astype(str).str.strip()

    combined["player_lower"] = (
        combined["player"]
        .str.lower()
        .str.strip()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
    )
    return combined


# ── Load models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = Path(__file__).parent
    scaler = joblib.load(base / "models" / "scaler.pkl")
    G_AB   = Generator(len(FEATURES))
    G_BA   = Generator(len(FEATURES))
    G_AB.load_state_dict(torch.load(base / "models" / "G_AB.pt", map_location="cpu"))
    G_BA.load_state_dict(torch.load(base / "models" / "G_BA.pt", map_location="cpu"))
    G_AB.eval()
    G_BA.eval()
    return G_AB, G_BA, scaler


# ── Search players ────────────────────────────────────────────────
def search_players(df, query, league_filter=None):
    q = query.lower().strip()
    if not q:
        return pd.DataFrame()

    # Also normalise query for accent-insensitive matching
    q_norm = (
        pd.Series([q])
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .iloc[0]
    )

    mask = (
        df["player_lower"].str.contains(q, na=False, regex=False) |
        df["player_lower"].str.contains(q_norm, na=False, regex=False)
    )
    results = df[mask].copy()

    if league_filter:
        league_mask = results["comp"].str.contains(
            league_filter, case=False, na=False, regex=False
        )
        results = results[league_mask]

    results = (
        results
        .dropna(subset=FEATURES)
        .sort_values("season", ascending=False)
        .drop_duplicates(subset=["player", "squad"])
        .head(10)
    )
    return results


# ── Translate stats ───────────────────────────────────────────────
def translate(stats_dict, direction, G_AB, G_BA, scaler):
    vec    = np.array([[stats_dict[f] for f in FEATURES]])
    scaled = scaler.transform(vec)
    x      = torch.FloatTensor(scaled)
    model  = G_AB if direction == "L1_to_PL" else G_BA
    with torch.no_grad():
        out = model(x).numpy()
    return dict(zip(FEATURES, scaler.inverse_transform(out)[0]))


# ── KPI card HTML ─────────────────────────────────────────────────
def kpi_card(label, orig, proj, src_label):
    delta = proj - orig
    if delta > 0.005:
        delta_html = f'<div class="kpi-delta-pos">▲ +{delta:.2f} vs {src_label}</div>'
    elif delta < -0.005:
        delta_html = f'<div class="kpi-delta-neg">▼ {delta:.2f} vs {src_label}</div>'
    else:
        delta_html = f'<div class="kpi-delta-neu">— {delta:.2f} vs {src_label}</div>'

    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-orig">{src_label}: {orig:.2f}</div>
        <div class="kpi-value">{proj:.2f}</div>
        {delta_html}
    </div>
    """


# ── Radar chart ───────────────────────────────────────────────────
def radar_chart(orig_dict, proj_dict):
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    ranges = {
        "Goals p 90": (0, 1.5), "Assists p 90": (0, 1.0), "Shots p 90": (0, 8),
        "% Shots on target": (0, 100), "Pass completion %": (0, 100),
        "Progressive Passes": (0, 150), "Progressive Carries": (0, 150),
        "Interceptions": (0, 80), "Tackles Won": (0, 80), "Key passes": (0, 100),
        "% Successful take-ons": (0, 100),
        "Shot creating actions p 90": (0, 10),
        "Goal creating actions p 90": (0, 5),
    }

    def norm(v, feat):
        lo, hi = ranges[feat]
        return max(0, min(1, (v - lo) / (hi - lo)))

    orig_n = [norm(orig_dict[f], f) for f in FEATURES]
    proj_n = [norm(proj_dict[f], f) for f in FEATURES]
    cats   = labels + [labels[0]]
    orig_n = orig_n + [orig_n[0]]
    proj_n = proj_n + [proj_n[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=orig_n, theta=cats, fill="toself", name="Original league",
        line=dict(color="#1D9E75", width=2), fillcolor="rgba(29,158,117,0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=proj_n, theta=cats, fill="toself", name="Projected league",
        line=dict(color="#185FA5", width=2), fillcolor="rgba(24,95,165,0.15)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[], gridcolor="#e5e7eb"),
            angularaxis=dict(gridcolor="#e5e7eb")
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        height=400,
        margin=dict(t=30, b=60, l=50, r=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Delta bar chart ───────────────────────────────────────────────
def delta_chart(orig_dict, proj_dict):
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    deltas = [proj_dict[f] - orig_dict[f] for f in FEATURES]
    colors = ["#1D9E75" if d >= 0 else "#E24B4A" for d in deltas]

    fig = go.Figure(go.Bar(
        x=deltas, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{'+'if d>=0 else ''}{d:.2f}" for d in deltas],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(
            zeroline=True, zerolinewidth=1.5,
            zerolinecolor="#9ca3af", title="Stat change"
        ),
        height=400,
        margin=dict(t=10, b=20, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Comparison radar ──────────────────────────────────────────────
def comparison_radar(p1_orig, p1_proj, p2_orig, p2_proj, name1, name2):
    ranges = {
        "Goals p 90": (0, 1.5), "Assists p 90": (0, 1.0), "Shots p 90": (0, 8),
        "% Shots on target": (0, 100), "Pass completion %": (0, 100),
        "Progressive Passes": (0, 150), "Progressive Carries": (0, 150),
        "Interceptions": (0, 80), "Tackles Won": (0, 80), "Key passes": (0, 100),
        "% Successful take-ons": (0, 100),
        "Shot creating actions p 90": (0, 10),
        "Goal creating actions p 90": (0, 5),
    }

    def norm(v, feat):
        lo, hi = ranges[feat]
        return max(0, min(1, (v - lo) / (hi - lo)))

    labels = [FEATURE_LABELS[f] for f in FEATURES]
    cats   = labels + [labels[0]]
    p1p    = [norm(p1_proj[f], f) for f in FEATURES] + [norm(p1_proj[FEATURES[0]], FEATURES[0])]
    p2p    = [norm(p2_proj[f], f) for f in FEATURES] + [norm(p2_proj[FEATURES[0]], FEATURES[0])]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=p1p, theta=cats, fill="toself", name=f"{name1} (proj)",
        line=dict(color="#185FA5", width=2), fillcolor="rgba(24,95,165,0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=p2p, theta=cats, fill="toself", name=f"{name2} (proj)",
        line=dict(color="#D85A30", width=2), fillcolor="rgba(216,90,48,0.15)"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   tickvals=[], gridcolor="#e5e7eb")),
        showlegend=True,
        legend=dict(orientation="h", y=-0.12),
        height=420,
        margin=dict(t=30, b=70, l=50, r=50),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ══════════════════════════════════════════════════════════════════
df_all          = load_dataset()
G_AB, G_BA, scaler = load_models()

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("## ⚽ Football Scout — League Stat Translator")
st.caption(
    "Search any player from the dataset · Auto-fills stats · "
    "CycleGAN projects how their numbers would look in another league"
)
st.divider()

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_single, tab_compare, tab_shortlist, tab_info = st.tabs([
    "🔍 Player lookup",
    "⚖️ Compare two players",
    "📋 Shortlist ranker",
    "ℹ️ How it works",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Single player lookup
# ══════════════════════════════════════════════════════════════════
with tab_single:

    col_search, col_settings = st.columns([3, 1])

    with col_settings:
        st.markdown("**Translation**")
        direction = st.radio(
            "Direction", ["Ligue 1 → PL", "PL → Ligue 1"],
            label_visibility="collapsed"
        )
        dir_key    = "L1_to_PL" if "Ligue 1" in direction else "PL_to_L1"
        src_league = "Ligue 1" if dir_key == "L1_to_PL" else "Premier League"
        tgt_league = "Premier League" if dir_key == "L1_to_PL" else "Ligue 1"

    with col_search:
        st.markdown("**Search player**")
        query = st.text_input(
            "Player name",
            placeholder="Type a name, e.g. Mbappe, Salah, Benzema...",
            label_visibility="collapsed",
            key="player_search"
        )

    # ── Debug toggle ──────────────────────────────────────────────
    if st.checkbox("Show available leagues in dataset", value=False):
        st.write(df_all["comp"].value_counts())

    # ── Search results ────────────────────────────────────────────
    if query and len(query) >= 2:

        all_results    = search_players(df_all, query, league_filter=None)
        league_results = search_players(df_all, query, league_filter=src_league)

        if len(league_results) == 0:
            if len(all_results) > 0:
                leagues_found = all_results["comp"].unique().tolist()
                st.warning(
                    f"**'{query}'** not found in **{src_league}** data. "
                    f"Found in: {', '.join(leagues_found)}. "
                    f"Switch direction or check the league filter."
                )
                with st.expander("Show matches from other leagues"):
                    for _, row in all_results.iterrows():
                        st.markdown(
                            f"**{row['player']}** · {row.get('comp','?')} · "
                            f"{row.get('squad','?')} · {row.get('season','?')}"
                        )
            else:
                q_ascii = (
                    query.lower()
                    .encode("ascii", errors="ignore")
                    .decode("ascii")
                    .strip()
                )
                fallback = search_players(df_all, q_ascii, league_filter=None)
                if len(fallback) > 0:
                    st.warning(f"No exact match for **'{query}'**. Did you mean one of these?")
                    for _, row in fallback.iterrows():
                        st.markdown(
                            f"**{row['player']}** · {row.get('comp','?')} · "
                            f"{row.get('squad','?')} · {row.get('season','?')}"
                        )
                else:
                    st.error(
                        f"No players found for **'{query}'**. "
                        f"Check spelling or try fewer characters."
                    )

        else:
            st.markdown(f"**{len(league_results)} result(s) in {src_league} — select:**")

            options = {}
            for _, row in league_results.iterrows():
                label = (
                    f"{row['player']}  ·  "
                    f"{row.get('squad','?')}  ·  "
                    f"{row.get('season','?')}  ·  "
                    f"{row.get('pos','?')}"
                )
                options[label] = row

            chosen_label = st.radio(
                "Select", list(options.keys()),
                label_visibility="collapsed"
            )
            chosen_row   = options[chosen_label]
            player_stats = {f: chosen_row[f] for f in FEATURES}

            st.divider()

            # ── Player header ─────────────────────────────────────
            h1, h2, h3, h4 = st.columns([3, 1, 1, 1])

            with h1:
                st.markdown(f"### {chosen_row['player']}")
                league_tag = "tag-l1" if src_league == "Ligue 1" else "tag-pl"
                tgt_tag    = "tag-pl" if tgt_league == "Premier League" else "tag-l1"
                st.markdown(
                    f'<span class="{league_tag}">{src_league}</span> &nbsp;→&nbsp; '
                    f'<span class="{tgt_tag}">{tgt_league}</span> &nbsp;&nbsp;'
                    f'<span class="tag-pos">{chosen_row.get("pos","?")}</span> &nbsp;'
                    f'<span class="tag-pos">{chosen_row.get("squad","?")}</span> &nbsp;'
                    f'<span class="tag-pos">{chosen_row.get("season","?")}</span>',
                    unsafe_allow_html=True
                )

            with h2:
                age = chosen_row.get("age", "—")
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">Age</div>'
                    f'<div class="kpi-value">{age}</div></div>',
                    unsafe_allow_html=True
                )

            with h3:
                matches = int(chosen_row.get("Matches Played", 0))
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">Matches</div>'
                    f'<div class="kpi-value">{matches}</div></div>',
                    unsafe_allow_html=True
                )

            with h4:
                nation = chosen_row.get("nation", "—")
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">Nation</div>'
                    f'<div class="kpi-value" style="font-size:16px">{nation}</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown(" ")
            projected = translate(player_stats, dir_key, G_AB, G_BA, scaler)

            # ── Key metric cards ──────────────────────────────────
            key_metrics = [
                "Goals p 90", "Assists p 90", "Shots p 90",
                "Pass completion %", "Shot creating actions p 90",
                "Goal creating actions p 90",
            ]
            card_html = ""
            for feat in key_metrics:
                orig = player_stats[feat]
                proj = projected[feat]
                card_html += (
                    f'<div style="flex:1;min-width:140px;">'
                    f'{kpi_card(FEATURE_LABELS[feat], orig, proj, src_league)}'
                    f'</div>'
                )
            st.markdown(
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 20px;">'
                f'{card_html}</div>',
                unsafe_allow_html=True
            )

            # ── Charts ────────────────────────────────────────────
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"**Radar — {src_league} vs {tgt_league} projection**")
                st.plotly_chart(
                    radar_chart(player_stats, projected),
                    use_container_width=True
                )

            with c2:
                st.markdown("**Stat changes after translation**")
                st.plotly_chart(
                    delta_chart(player_stats, projected),
                    use_container_width=True
                )

            # ── Full stat breakdown ───────────────────────────────
            st.markdown("**Full stat breakdown**")
            categories = {
                "Attacking"  : ATTACKING,
                "Possession" : POSSESSION,
                "Defensive"  : DEFENSIVE,
            }

            for cat_name, cat_feats in categories.items():
                st.markdown(
                    f'<div class="stat-section-title">{cat_name}</div>',
                    unsafe_allow_html=True
                )
                rows = []
                for f in cat_feats:
                    orig  = player_stats[f]
                    proj  = projected[f]
                    delta = proj - orig
                    arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "—")
                    rows.append({
                        "Stat"               : FEATURE_LABELS[f],
                        src_league           : round(orig, 3),
                        f"{tgt_league} proj" : round(proj, 3),
                        "Δ Change"           : f"{arrow} {abs(delta):.3f}",
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Stat"               : st.column_config.TextColumn(width="medium"),
                        src_league           : st.column_config.NumberColumn(format="%.3f"),
                        f"{tgt_league} proj" : st.column_config.NumberColumn(format="%.3f"),
                        "Δ Change"           : st.column_config.TextColumn(width="small"),
                    }
                )

            # ── Download ──────────────────────────────────────────
            all_rows = []
            for f in FEATURES:
                orig  = player_stats[f]
                proj  = projected[f]
                all_rows.append({
                    "stat"                    : FEATURE_LABELS[f],
                    src_league                : round(orig, 3),
                    f"{tgt_league}_projected" : round(proj, 3),
                    "delta"                   : round(proj - orig, 3),
                })
            csv_out = pd.DataFrame(all_rows).to_csv(index=False)
            st.download_button(
                f"Download {chosen_row['player']} projection",
                data=csv_out,
                file_name=f"{chosen_row['player'].replace(' ','_')}_projection.csv",
                mime="text/csv",
            )

    elif query and len(query) < 2:
        st.info("Type at least 2 characters to search.")

    else:
        st.info(
            f"Search for a player above. Showing **{src_league}** players only "
            f"for the selected direction."
        )

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Compare two players
# ══════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("Search two players and compare their projected stats side by side.")

    src_cmp = "Ligue 1"
    tgt_cmp = "Premier League"
    dir_key_cmp = "L1_to_PL"

    def player_search_widget(col, key_prefix, src_league):
        with col:
            st.markdown(f"**Player {key_prefix[-1]}** — {src_league}")
            q = st.text_input(
                "Name", key=f"{key_prefix}_query",
                placeholder="Search player...",
                label_visibility="collapsed"
            )
            selected = None
            if q and len(q) >= 2:
                res = search_players(df_all, q, league_filter=src_league)
                if len(res) == 0:
                    st.warning("No results.")
                else:
                    opts = {}
                    for _, row in res.iterrows():
                        lbl = (f"{row['player']} · "
                               f"{row.get('squad','?')} · "
                               f"{row.get('season','?')}")
                        opts[lbl] = row
                    chosen_lbl = st.radio(
                        "Select", list(opts.keys()),
                        label_visibility="collapsed",
                        key=f"{key_prefix}_radio"
                    )
                    selected = opts[chosen_lbl]
                    st.caption(
                        f"{selected.get('pos','?')} · "
                        f"Age {selected.get('age','?')} · "
                        f"{int(selected.get('Matches Played',0))} matches"
                    )
            return selected

    p1_col, p2_col = st.columns(2)
    p1_row = player_search_widget(p1_col, "p1", src_cmp)
    p2_row = player_search_widget(p2_col, "p2", src_cmp)

    if p1_row is not None and p2_row is not None:
        st.divider()

        p1_orig = {f: p1_row[f] for f in FEATURES}
        p2_orig = {f: p2_row[f] for f in FEATURES}
        p1_proj = translate(p1_orig, dir_key_cmp, G_AB, G_BA, scaler)
        p2_proj = translate(p2_orig, dir_key_cmp, G_AB, G_BA, scaler)

        n1 = p1_row["player"]
        n2 = p2_row["player"]

        st.markdown(f"### {n1}  vs  {n2}")
        st.caption(f"Both projected to {tgt_cmp}")

        st.plotly_chart(
            comparison_radar(p1_orig, p1_proj, p2_orig, p2_proj, n1, n2),
            use_container_width=True
        )

        st.markdown("**Head-to-head projected stats**")
        cmp_rows = []
        for f in FEATURES:
            v1     = p1_proj[f]
            v2     = p2_proj[f]
            winner = n1 if v1 > v2 else (n2 if v2 > v1 else "Draw")
            cmp_rows.append({
                "Stat"           : FEATURE_LABELS[f],
                f"{n1} (proj)"   : round(v1, 3),
                f"{n2} (proj)"   : round(v2, 3),
                "Better"         : winner,
            })
        cmp_df = pd.DataFrame(cmp_rows)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        csv_cmp = cmp_df.to_csv(index=False)
        st.download_button(
            f"Download comparison: {n1} vs {n2}",
            data=csv_cmp,
            file_name=f"comparison_{n1.replace(' ','_')}_vs_{n2.replace(' ','_')}.csv",
            mime="text/csv",
        )
    else:
        st.info("Search and select both players above to see the comparison.")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Shortlist ranker
# ══════════════════════════════════════════════════════════════════
with tab_shortlist:
    st.markdown(
        "Search multiple players, add them to a shortlist, "
        "then rank by any projected stat."
    )

    if "shortlist" not in st.session_state:
        st.session_state.shortlist = []

    src_sl = "Ligue 1"
    tgt_sl = "Premier League"
    dir_sl = "L1_to_PL"

    add_col, list_col = st.columns([1, 2])

    with add_col:
        st.markdown("**Add player to shortlist**")
        sl_query = st.text_input(
            "Search", placeholder="Type player name...",
            key="sl_query", label_visibility="collapsed"
        )
        if sl_query and len(sl_query) >= 2:
            sl_results = search_players(df_all, sl_query, league_filter=src_sl)
            if len(sl_results) == 0:
                st.warning("No results.")
            else:
                sl_opts = {}
                for _, row in sl_results.iterrows():
                    lbl = (
                        f"{row['player']} · "
                        f"{row.get('squad','?')} · "
                        f"{row.get('season','?')}"
                    )
                    sl_opts[lbl] = row

                sl_chosen_lbl = st.radio(
                    "Select player", list(sl_opts.keys()),
                    label_visibility="collapsed", key="sl_radio"
                )
                sl_chosen = sl_opts[sl_chosen_lbl]

                if st.button("Add to shortlist", type="primary"):
                    existing = [
                        p["player"] + str(p.get("season", ""))
                        for p in st.session_state.shortlist
                    ]
                    uid = sl_chosen["player"] + str(sl_chosen.get("season", ""))
                    if uid in existing:
                        st.warning("Already in shortlist.")
                    else:
                        entry = {f: sl_chosen[f] for f in FEATURES}
                        entry["player"] = sl_chosen["player"]
                        entry["squad"]  = sl_chosen.get("squad", "?")
                        entry["season"] = sl_chosen.get("season", "?")
                        entry["pos"]    = sl_chosen.get("pos", "?")
                        st.session_state.shortlist.append(entry)
                        st.success(f"Added {sl_chosen['player']}")

    with list_col:
        st.markdown(f"**Shortlist ({len(st.session_state.shortlist)} players)**")

        if len(st.session_state.shortlist) == 0:
            st.info("No players added yet. Search and add players on the left.")
        else:
            to_remove = None
            for i, p in enumerate(st.session_state.shortlist):
                rc1, rc2 = st.columns([5, 1])
                rc1.markdown(
                    f"**{p['player']}** &nbsp; "
                    f'<span class="tag-pos">{p["squad"]}</span> &nbsp;'
                    f'<span class="tag-pos">{p["season"]}</span> &nbsp;'
                    f'<span class="tag-pos">{p["pos"]}</span>',
                    unsafe_allow_html=True
                )
                if rc2.button("✕", key=f"rm_{i}"):
                    to_remove = i

            if to_remove is not None:
                st.session_state.shortlist.pop(to_remove)
                st.rerun()

            if st.button("Clear all"):
                st.session_state.shortlist = []
                st.rerun()

    # ── Rank shortlist ────────────────────────────────────────────
    if len(st.session_state.shortlist) >= 2:
        st.divider()
        st.markdown(f"**Projected {tgt_sl} rankings**")

        rank_by = st.selectbox(
            "Rank by projected stat",
            options=FEATURES,
            format_func=lambda f: FEATURE_LABELS[f]
        )

        rank_rows = []
        for p in st.session_state.shortlist:
            orig = {f: p[f] for f in FEATURES}
            proj = translate(orig, dir_sl, G_AB, G_BA, scaler)
            rank_rows.append({
                "Player"               : p["player"],
                "Club"                 : p["squad"],
                "Season"               : p["season"],
                "Position"             : p["pos"],
                f"Original ({src_sl})" : round(orig[rank_by], 3),
                f"Projected ({tgt_sl})": round(proj[rank_by], 3),
                "Delta"                : round(proj[rank_by] - orig[rank_by], 3),
            })

        rank_df = (
            pd.DataFrame(rank_rows)
            .sort_values(f"Projected ({tgt_sl})", ascending=False)
            .reset_index(drop=True)
        )
        rank_df.index += 1

        st.dataframe(rank_df, use_container_width=True)

        fig_rank = go.Figure()
        fig_rank.add_trace(go.Bar(
            name=f"Original ({src_sl})",
            x=rank_df["Player"],
            y=rank_df[f"Original ({src_sl})"],
            marker_color="#1D9E75", opacity=0.75
        ))
        fig_rank.add_trace(go.Bar(
            name=f"Projected ({tgt_sl})",
            x=rank_df["Player"],
            y=rank_df[f"Projected ({tgt_sl})"],
            marker_color="#185FA5", opacity=0.9
        ))
        fig_rank.update_layout(
            barmode="group",
            yaxis_title=FEATURE_LABELS[rank_by],
            xaxis_tickangle=-25,
            height=380,
            margin=dict(t=10, b=80),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        csv_rank = rank_df.to_csv(index=False)
        st.download_button(
            "Download shortlist rankings",
            data=csv_rank,
            file_name="shortlist_rankings.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════════════════════════
# TAB 4 — How it works
# ══════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("How this tool works")
    st.markdown("""
    #### The cross-league translation problem
    A player scoring 0.5 goals/90 in Ligue 1 is not the same as a player
    scoring 0.5 goals/90 in the Premier League. League difficulty, pressing
    intensity, tactical systems and pace all compress or inflate stats
    differently. This tool uses a **CycleGAN** to learn those distribution
    differences and translate stats between leagues.

    #### Three ways to use it

    | Tab | Use case |
    |---|---|
    | Player lookup | Search any player, see instant projections with radar + delta chart |
    | Compare two | Head-to-head projected comparison of two targets |
    | Shortlist ranker | Build a list of targets, rank by any projected stat |

    #### What the model learned
    Trained on **7 seasons (2017–18 to 2023–24)** with ~3,200 player-seasons
    per league. It learned that moving from Ligue 1 to the Premier League
    typically involves higher pressing (interceptions and tackles increase),
    slightly lower goal and shot rates, and tighter pass completion figures.

    #### Limitations
    - Goalkeepers have a different stat profile — results unreliable for GKs
    - Players with fewer than 5 matches have been excluded from the dataset
    - The model projects statistical output, not ability — context (system fit,
      injuries, motivation) is not captured
    - Best used for shortlisting and comparison, not final decisions
    """)

    st.markdown("#### Features used")
    st.dataframe(
        pd.DataFrame([
            {"Feature": FEATURE_LABELS[f], "Column in dataset": f}
            for f in FEATURES
        ]),
        use_container_width=True,
        hide_index=True
    )