"""
Delhivery Network Intelligence Dashboard
==========================================
Live delay risk scoring, bottleneck detection, and corridor audit
for Delhivery's logistics network.

Run with:
    pip install streamlit plotly pandas numpy networkx
    streamlit run delhivery_dashboard.py
    
Place the following CSVs in the same directory:
    - delivery_data.csv
    - Top_Bottleneck_Hubs.csv
    - SLA_Breach_Corridors.csv
"""

import pathlib
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────
RED    = "#E8002D"   # Delhivery red
ORANGE = "#FF6B35"
AMBER  = "#FFC947"
GREEN  = "#2ECC71"
DARK   = "#0D1117"
CARD   = "#161B22"
BORDER = "#30363D"
TEXT   = "#E6EDF3"
MUTED  = "#8B949E"

CUSTOM_CSS = f"""
<style>
    /* ── global ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {DARK};
        color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background-color: {CARD};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}

    /* ── metric cards ── */
    .kpi-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }}
    .kpi-value {{
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .kpi-label {{
        font-size: 0.78rem;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }}
    .kpi-delta {{
        font-size: 0.82rem;
        margin-top: 6px;
    }}
    .red   {{ color: {RED}; }}
    .amber {{ color: {AMBER}; }}
    .green {{ color: {GREEN}; }}

    /* ── section headers ── */
    .section-header {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {TEXT};
        border-left: 3px solid {RED};
        padding-left: 10px;
        margin: 8px 0 16px 0;
    }}

    /* ── risk badge ── */
    .badge-critical {{ background:{RED};    color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; }}
    .badge-high     {{ background:{ORANGE}; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; }}
    .badge-medium   {{ background:{AMBER};  color:#000; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; }}
    .badge-low      {{ background:{GREEN};  color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; }}

    /* ── plotly charts transparent ── */
    .js-plotly-plot .plotly {{ background: transparent !important; }}
    div[data-testid="stHorizontalBlock"] > div {{ gap: 12px; }}

    /* ── tab bar ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {CARD};
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {MUTED};
        background: transparent;
        border-radius: 6px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {RED} !important;
        color: #fff !important;
    }}
    /* ── scrollable table ── */
    .scroll-table {{ max-height: 400px; overflow-y: auto; }}
    hr {{ border-color: {BORDER}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    BASE_DIR: Path = Path(__file__).parent
    df   = pd.read_csv(BASE_DIR / "delivery_data.csv")
    hubs = pd.read_csv(BASE_DIR / "Top_Bottleneck_Hubs.csv")
    sla  = pd.read_csv(BASE_DIR / "SLA_Breach_Corridors.csv")

    # ── delivery_data enrichment ──
    df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
    df["hour"]      = df["trip_creation_time"].dt.hour
    df["dow"]       = df["trip_creation_time"].dt.day_name()
    df["state"]     = df["source_name"].str.extract(r"\(([^)]+)\)")
    df["dst_state"] = df["destination_name"].str.extract(r"\(([^)]+)\)")
    df["is_breach"] = df["factor"] > 1.2

    df["dist_band"] = pd.cut(
        df["actual_distance_to_destination"],
        bins=[0, 50, 100, 200, 500, 1e6],
        labels=["<50 km", "50–100 km", "100–200 km", "200–500 km", ">500 km"],
    )

    # Composite risk score for hubs (0-100)
    hubs["risk_score"] = (
        0.40 * (hubs["Betweenness_Centrality"] / hubs["Betweenness_Centrality"].max()) +
        0.25 * (hubs["Total_Degree"]           / hubs["Total_Degree"].max()) +
        0.20 * (hubs["PageRank"]               / hubs["PageRank"].max()) +
        0.15 * (hubs["In_Degree"]              / hubs["In_Degree"].max())
    ) * 100

    hubs["risk_tier"] = pd.cut(
        hubs["risk_score"],
        bins=[0, 20, 40, 65, 100],
        labels=["Low", "Medium", "High", "Critical"],
    )

    # SLA corridor risk
    sla["corridor_risk"] = (
        0.5 * (sla["total_breach_score"] / sla["total_breach_score"].max()) +
        0.3 * (sla["max_delay_ratio"]    / sla["max_delay_ratio"].max()) +
        0.2 * (sla["total_trips"]        / sla["total_trips"].max())
    ) * 100

    return df, hubs, sla


@st.cache_data(show_spinner=False)
def build_graph(hubs, sla, top_n_hubs=50, top_n_corridors=80):
    G = nx.DiGraph()

    for _, r in hubs.head(top_n_hubs).iterrows():
        G.add_node(r["Hub"],
                   betweenness=r["Betweenness_Centrality"],
                   pagerank=r["PageRank"],
                   total_degree=r["Total_Degree"],
                   risk_score=r["risk_score"],
                   risk_tier=str(r["risk_tier"]),
                   in_degree=r["In_Degree"],
                   out_degree=r["Out_Degree"])

    for _, r in sla.head(top_n_corridors).iterrows():
        for n in [r["source_name"], r["destination_name"]]:
            if n not in G.nodes:
                G.add_node(n, betweenness=0, pagerank=0, total_degree=0,
                           risk_score=0, risk_tier="Low", in_degree=0, out_degree=0)
        G.add_edge(r["source_name"], r["destination_name"],
                   breach_score=r["total_breach_score"],
                   max_delay=r["max_delay_ratio"],
                   trips=r["total_trips"],
                   corridor_risk=r["corridor_risk"])

    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=60)
    return G, pos


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def risk_color(score):
    if score >= 65: return RED
    if score >= 40: return ORANGE
    if score >= 20: return AMBER
    return GREEN

def plotly_defaults(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    )
    return fig

def kpi_html(value, label, color=TEXT, delta=None):
    delta_html = f'<div class="kpi-delta" style="color:{MUTED}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def sidebar(hubs, sla, df):
    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:12px 0 20px">'
            f'<span style="font-size:2rem">🚚</span><br>'
            f'<span style="font-size:1.1rem;font-weight:700;color:{RED}">Delhivery</span><br>'
            f'<span style="font-size:0.75rem;color:{MUTED}">Network Intelligence</span>'
            f'</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="section-header">Filters</div>', unsafe_allow_html=True)

        states = sorted(df["state"].dropna().unique())
        sel_states = st.multiselect("Source State", states, default=[], placeholder="All states")

        route_types = df["route_type"].unique().tolist()
        sel_rt = st.multiselect("Route Type", route_types, default=route_types)

        dist_bands = ["<50 km", "50–100 km", "100–200 km", "200–500 km", ">500 km"]
        sel_dist = st.multiselect("Distance Band", dist_bands, default=dist_bands)

        delay_thresh = st.slider("Delay Ratio Threshold (breach)", 1.0, 3.0, 1.2, 0.05)

        st.markdown("---")
        st.markdown(f'<div class="section-header">Graph Controls</div>', unsafe_allow_html=True)
        top_n_hubs = st.slider("Hubs shown in graph", 20, 80, 50, 5)
        top_n_corr = st.slider("Corridors shown in graph", 30, 120, 80, 10)

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.72rem;color:{MUTED};text-align:center">'
            f'Data: Delhivery Trip Segments<br>'
            f'Graph: NetworkX Spring Layout<br>'
            f'Risk score: Betweenness × PageRank<br>'
            f'SLA breach: Actual/OSRM &gt; {delay_thresh:.2f}'
            f'</div>', unsafe_allow_html=True)

    return sel_states, sel_rt, sel_dist, delay_thresh, top_n_hubs, top_n_corr


# ─────────────────────────────────────────────
# TAB 1 ─ EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────
def tab_overview(df, hubs, sla, delay_thresh):
    # ── apply threshold to this view ──
    df = df.copy()
    df["is_breach"] = df["factor"] > delay_thresh

    total_trips  = df["trip_uuid"].nunique()
    breach_rate  = df["is_breach"].mean() * 100
    median_delay = df["factor"].median()
    critical_hubs = int((hubs["risk_tier"] == "Critical").sum())
    top_corridor_risk = sla.iloc[0]["total_breach_score"]

    # ── KPI row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(kpi_html(f"{total_trips:,}", "Unique Trips", TEXT), unsafe_allow_html=True)
    with c2: st.markdown(kpi_html(f"{breach_rate:.1f}%", "SLA Breach Rate", RED, f"Threshold: ×{delay_thresh:.2f}"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_html(f"×{median_delay:.2f}", "Median Delay Ratio", AMBER, "Actual / OSRM"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_html(f"{critical_hubs}", "Critical Hubs", RED, "Betweenness top tier"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_html(f"{len(sla[sla['max_delay_ratio']>2]):,}", "Chronic Corridors", ORANGE, ">2× OSRM delay"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Delay Ratio Distribution by Route Type</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for rt, clr in [("FTL", RED), ("Carting", AMBER)]:
            vals = df[df["route_type"] == rt]["factor"].clip(0, 8)
            fig.add_trace(go.Histogram(
                x=vals, name=rt, nbinsx=60,
                marker_color=clr, opacity=0.75,
                histnorm="probability density"
            ))
        fig.add_vline(x=delay_thresh, line_dash="dash", line_color=GREEN,
                      annotation_text=f"Breach >{delay_thresh:.2f}", annotation_font_color=GREEN)
        fig.update_layout(barmode="overlay", xaxis_title="Delay Ratio (Actual/OSRM)",
                          yaxis_title="Density")
        plotly_defaults(fig, 340)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Route Type SLA Scorecard</div>', unsafe_allow_html=True)
        rt_stats = df.groupby("route_type").agg(
            Trips=("factor", "count"),
            Median_Delay=("factor", "median"),
            Mean_Delay=("factor", "mean"),
            Breach_Pct=("is_breach", lambda x: x.mean() * 100),
        ).reset_index().round(2)

        fig2 = go.Figure(data=[go.Table(
            header=dict(
                values=["Route Type", "Trips", "Median ×", "Mean ×", "Breach %"],
                fill_color=BORDER, font=dict(color=TEXT, size=12),
                align="center", height=32
            ),
            cells=dict(
                values=[rt_stats[c] for c in ["route_type", "Trips", "Median_Delay", "Mean_Delay", "Breach_Pct"]],
                fill_color=[[CARD, CARD] * 10],
                font=dict(color=[TEXT, TEXT, AMBER, ORANGE, RED], size=12),
                align="center", height=30
            )
        )])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=150)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header" style="margin-top:12px">Breach Rate by Distance Band</div>', unsafe_allow_html=True)
        band_stats = df.groupby(["dist_band", "route_type"])["is_breach"].mean().mul(100).reset_index()
        band_stats.columns = ["Band", "Route", "Breach%"]
        fig3 = px.bar(band_stats, x="Band", y="Breach%", color="Route",
                      barmode="group",
                      color_discrete_map={"FTL": RED, "Carting": AMBER})
        fig3.update_layout(xaxis_title="", yaxis_title="Breach %", showlegend=True)
        plotly_defaults(fig3, 220)
        st.plotly_chart(fig3, use_container_width=True)

    # ── hourly heatmap ──
    st.markdown('<div class="section-header">Hourly Delay Pattern — Median Factor by Hour & Route Type</div>', unsafe_allow_html=True)
    hourly = df.groupby(["hour", "route_type"])["factor"].median().reset_index()
    fig_h = px.line(hourly, x="hour", y="factor", color="route_type",
                    color_discrete_map={"FTL": RED, "Carting": AMBER},
                    markers=True)
    fig_h.add_hline(y=delay_thresh, line_dash="dot", line_color=GREEN,
                    annotation_text=f"Breach threshold ×{delay_thresh:.2f}")
    fig_h.update_layout(xaxis_title="Hour of Day (0–23)", yaxis_title="Median Delay Ratio",
                         xaxis=dict(tickmode="linear", dtick=2))
    plotly_defaults(fig_h, 300)
    st.plotly_chart(fig_h, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2 ─ NETWORK GRAPH
# ─────────────────────────────────────────────
def tab_network(hubs, sla, top_n_hubs, top_n_corr):
    st.markdown('<div class="section-header">Interactive Network Graph — Hub Bottlenecks & Delay Corridors</div>',
                unsafe_allow_html=True)
    st.caption("Node size = Betweenness Centrality | Node color = Risk Score (red = critical) | Edge width = Breach Score")

    G, pos = build_graph(hubs, sla, top_n_hubs, top_n_corr)

    # ── edges ──
    edge_traces = []
    max_breach = max((d.get("breach_score", 1) for _, _, d in G.edges(data=True)), default=1)

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        breach = data.get("breach_score", 0)
        ratio  = breach / max_breach
        width  = 1 + ratio * 6
        clr    = f"rgba({int(232*ratio + 48*(1-ratio))},{int(2*ratio + 211*(1-ratio))},{int(45*ratio + 113*(1-ratio))},0.6)"

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=clr),
            hoverinfo="skip", showlegend=False
        ))

    # ── nodes ──
    node_x, node_y, node_text, node_hover, node_size, node_color = [], [], [], [], [], []
    max_bc = max((d.get("betweenness", 0.001) for _, d in G.nodes(data=True)), default=0.001)

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        short = node.split(" (")[0][:20]
        node_text.append(short)
        rs = data.get("risk_score", 0)
        node_color.append(rs)
        bc = data.get("betweenness", 0)
        node_size.append(10 + (bc / max_bc) * 40)
        node_hover.append(
            f"<b>{node}</b><br>"
            f"Risk Score: {rs:.1f}<br>"
            f"Betweenness: {bc:.4f}<br>"
            f"In-degree: {data.get('in_degree',0)}<br>"
            f"Out-degree: {data.get('out_degree',0)}<br>"
            f"PageRank: {data.get('pagerank',0):.5f}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        textfont=dict(size=8, color=TEXT),
        hovertext=node_hover, hoverinfo="text",
        marker=dict(
            size=node_size, color=node_color,
            colorscale=[[0, GREEN], [0.4, AMBER], [0.7, ORANGE], [1.0, RED]],
            cmin=0, cmax=100,
            colorbar=dict(title="Risk Score", thickness=12, tickfont=dict(color=TEXT)),
            line=dict(width=1, color=BORDER)
        ),
        showlegend=False
    )

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        hovermode="closest"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── legend ──
    lc1, lc2, lc3, lc4 = st.columns(4)
    for col, tier, clr, desc in [
        (lc1, "Critical", RED,    "Betweenness >65th pctile"),
        (lc2, "High",     ORANGE, "Betweenness 40–65th"),
        (lc3, "Medium",   AMBER,  "Betweenness 20–40th"),
        (lc4, "Low",      GREEN,  "Betweenness <20th"),
    ]:
        col.markdown(
            f'<div style="background:{CARD};border:1px solid {clr};border-radius:8px;'
            f'padding:10px;text-align:center">'
            f'<span style="color:{clr};font-weight:700">{tier}</span><br>'
            f'<span style="font-size:0.75rem;color:{MUTED}">{desc}</span></div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 3 ─ BOTTLENECK HUBS
# ─────────────────────────────────────────────
def tab_hubs(hubs):
    st.markdown('<div class="section-header">Top Bottleneck Hubs — Risk Leaderboard</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns([3, 2])

    with c_left:
        top20 = hubs.head(20).copy()
        fig = go.Figure()
        colors = [risk_color(s) for s in top20["risk_score"]]
        short_names = top20["Hub"].str.split(r" \(").str[0]

        fig.add_trace(go.Bar(
            x=top20["risk_score"],
            y=short_names,
            orientation="h",
            marker_color=colors,
            text=top20["risk_score"].round(1).astype(str) + " pts",
            textposition="inside",
            textfont=dict(color="#fff", size=11),
            customdata=np.column_stack([
                top20["Betweenness_Centrality"].round(4),
                top20["Total_Degree"],
                top20["PageRank"].round(5),
                top20["Hub"]
            ]),
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "Risk Score: %{x:.1f}<br>"
                "Betweenness: %{customdata[0]}<br>"
                "Total Degree: %{customdata[1]}<br>"
                "PageRank: %{customdata[2]}<extra></extra>"
            )
        ))
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Composite Risk Score (0–100)",
        )
        plotly_defaults(fig, 520)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown('<div class="section-header">Graph Metric Deep Dive</div>', unsafe_allow_html=True)

        # Scatter: betweenness vs pagerank, sized by degree
        top40 = hubs.head(40).copy()
        top40["short"] = top40["Hub"].str.split(" (", regex = False).str[0]
        fig2 = px.scatter(
            top40, x="Betweenness_Centrality", y="PageRank",
            size="Total_Degree", color="risk_score",
            hover_name="short",
            color_continuous_scale=[[0, GREEN], [0.5, AMBER], [1.0, RED]],
            size_max=35, labels={"risk_score": "Risk Score"}
        )
        fig2.update_coloraxes(colorbar_tickfont_color=TEXT)
        plotly_defaults(fig2, 280)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">In vs Out Degree (Top 15)</div>', unsafe_allow_html=True)
        top15 = hubs.head(15).copy()
        top15["short"] = top15["Hub"].str.split(" (", regex = False).str[0].str[:18]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="In-degree",  x=top15["short"], y=top15["In_Degree"],  marker_color=RED))
        fig3.add_trace(go.Bar(name="Out-degree", x=top15["short"], y=top15["Out_Degree"], marker_color=AMBER))
        fig3.update_layout(barmode="group", xaxis_tickangle=-40, xaxis_title="", yaxis_title="Degree")
        plotly_defaults(fig3, 240)
        st.plotly_chart(fig3, use_container_width=True)

    # ── full ranked table ──
    st.markdown('<div class="section-header">Full Hub Rankings</div>', unsafe_allow_html=True)

    display_hubs = hubs[["Hub", "risk_score", "risk_tier", "Betweenness_Centrality",
                          "In_Degree", "Out_Degree", "PageRank",
                          "Clustering_Coefficient", "Total_Degree"]].copy()
    display_hubs.columns = ["Hub", "Risk Score", "Tier", "Betweenness",
                             "In-Deg", "Out-Deg", "PageRank", "Clustering", "Total Deg"]
    display_hubs["Risk Score"] = display_hubs["Risk Score"].round(1)
    display_hubs["Betweenness"] = display_hubs["Betweenness"].round(5)
    display_hubs["PageRank"] = display_hubs["PageRank"].round(6)
    display_hubs["Clustering"] = display_hubs["Clustering"].round(4)

    n_show = st.slider("Show top N hubs", 10, 100, 30, 5, key="hub_table_n")

    def color_tier(val):
        colors = {"Critical": f"color:{RED};font-weight:700",
                  "High": f"color:{ORANGE};font-weight:700",
                  "Medium": f"color:{AMBER}",
                  "Low": f"color:{GREEN}"}
        return colors.get(str(val), "")

    styled = display_hubs.head(n_show).style \
        .applymap(color_tier, subset=["Tier"]) \
        .background_gradient(subset=["Risk Score"], cmap="Reds") \
        .format({"Risk Score": "{:.1f}", "Betweenness": "{:.5f}", "PageRank": "{:.6f}"})
    st.dataframe(styled, use_container_width=True, height=420)


# ─────────────────────────────────────────────
# TAB 4 ─ CORRIDOR AUDIT
# ─────────────────────────────────────────────
def tab_corridors(sla, df):
    st.markdown('<div class="section-header">Corridor SLA Breach Audit</div>', unsafe_allow_html=True)

    # top corridor chart
    top_n = st.slider("Show top N corridors", 10, 50, 20, 5, key="corr_n")
    top_c = sla.head(top_n).copy()
    top_c["label"] = (
        top_c["source_name"].str.split(" (", regex=False).str[0].str[:15]
        + " → "
        + top_c["destination_name"].str.split(" (", regex=False).str[0].str[:15]
    )

    top_c["state"] = top_c["source_name"].str.extract(r"\(([^)]+)\)")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-header">Total Breach Score by Corridor</div>', unsafe_allow_html=True)
        fig = px.bar(top_c, x="total_breach_score", y="label",
                     orientation="h", color="max_delay_ratio",
                     color_continuous_scale=[[0, AMBER], [0.5, ORANGE], [1, RED]],
                     hover_data=["total_trips", "max_delay_ratio"],
                     labels={"total_breach_score": "Breach Score",
                             "max_delay_ratio": "Max Delay ×"})
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Total Breach Score",
                          coloraxis_colorbar_tickfont_color=TEXT)
        plotly_defaults(fig, 560)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Max Delay Ratio — Worst Offenders</div>', unsafe_allow_html=True)
        worst = sla.nlargest(15, "max_delay_ratio").copy()
        worst["label"] = (worst["source_name"].str.split(" (", regex = False).str[0].str[:12] + " →\n" +
                          worst["destination_name"].str.split(" (", regex = False).str[0].str[:12])
        fig2 = go.Figure(go.Bar(
            x=worst["label"], y=worst["max_delay_ratio"],
            marker_color=[risk_color(v / worst["max_delay_ratio"].max() * 100)
                          for v in worst["max_delay_ratio"]],
            text=worst["max_delay_ratio"].round(1).astype(str) + "×",
            textposition="outside", textfont=dict(color=TEXT)
        ))
        fig2.add_hline(y=1.2, line_dash="dash", line_color=GREEN,
                       annotation_text="×1.2 SLA threshold")
        fig2.update_layout(xaxis_tickangle=-35, yaxis_title="Max Delay Ratio",
                           xaxis_title="")
        plotly_defaults(fig2, 320)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Breach Score by State</div>', unsafe_allow_html=True)
        state_breach = sla.copy()
        state_breach["state"] = state_breach["source_name"].str.extract(r"\(([^)]+)\)")
        state_agg = state_breach.groupby("state")["total_breach_score"].sum().nlargest(12).reset_index()
        fig3 = px.bar(state_agg, x="total_breach_score", y="state",
                      orientation="h",
                      color="total_breach_score",
                      color_continuous_scale=[[0, AMBER], [1, RED]])
        fig3.update_layout(yaxis=dict(autorange="reversed"),
                           xaxis_title="Total Breach Score", yaxis_title="",
                           showlegend=False, coloraxis_showscale=False)
        plotly_defaults(fig3, 300)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Delay over time ──
    st.markdown('<div class="section-header">Delay Ratio by Time of Day — All Corridors</div>',
                unsafe_allow_html=True)
    hourly_rt = df.groupby(["hour", "route_type"])["factor"].agg(
        median="median", p90=lambda x: x.quantile(0.9)
    ).reset_index()

    fig_t = make_subplots(rows=1, cols=2, subplot_titles=["FTL", "Carting"])
    for col_idx, rt in enumerate(["FTL", "Carting"], 1):
        sub = hourly_rt[hourly_rt["route_type"] == rt]
        clr = RED if rt == "FTL" else AMBER
        fig_t.add_trace(go.Scatter(x=sub["hour"], y=sub["median"], name=f"{rt} Median",
                                    line=dict(color=clr, width=2.5), mode="lines+markers"), row=1, col=col_idx)
        fig_t.add_trace(go.Scatter(x=sub["hour"], y=sub["p90"], name=f"{rt} P90",
                                    line=dict(color=clr, width=1, dash="dot"),
                                    fill="tonexty", fillcolor=f"rgba(232,0,45,0.08)"
                                    if rt == "FTL" else f"rgba(255,201,71,0.08)"), row=1, col=col_idx)
        fig_t.add_hline(y=1.2, line_dash="dash", line_color=GREEN, row=1, col=col_idx)

    fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(color=TEXT), height=300,
                         xaxis=dict(gridcolor=BORDER, title="Hour"),
                         xaxis2=dict(gridcolor=BORDER, title="Hour"),
                         yaxis=dict(gridcolor=BORDER, title="Delay Ratio"),
                         yaxis2=dict(gridcolor=BORDER),
                         margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_t, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 5 ─ FTL VS CARTING
# ─────────────────────────────────────────────
def tab_ftl_carting(df):
    st.markdown('<div class="section-header">FTL vs Carting Decision Intelligence</div>', unsafe_allow_html=True)

    df = df.copy()
    df["is_breach"] = df["factor"] > 1.2

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Delay Distribution — FTL vs Carting</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for rt, clr in [("FTL", RED), ("Carting", AMBER)]:
            vals = df[df["route_type"] == rt]["factor"].clip(0, 6)
            fig.add_trace(go.Violin(x=vals, name=rt, fillcolor=clr,
                                    line_color=clr, opacity=0.7, side="positive",
                                    meanline_visible=True, box_visible=True))
        fig.update_layout(violinmode="overlay", xaxis_title="Delay Ratio")
        plotly_defaults(fig, 320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Median Delay by Distance Band</div>', unsafe_allow_html=True)
        band_data = df.groupby(["dist_band", "route_type"])["factor"].median().reset_index()
        band_data.columns = ["Band", "Route", "Median Factor"]
        fig2 = px.bar(band_data, x="Band", y="Median Factor", color="Route",
                      barmode="group",
                      color_discrete_map={"FTL": RED, "Carting": AMBER},
                      text=band_data["Median Factor"].round(2))
        fig2.update_traces(textposition="outside")
        fig2.add_hline(y=1.2, line_dash="dot", line_color=GREEN)
        fig2.update_layout(yaxis_title="Median Delay Ratio", xaxis_title="")
        plotly_defaults(fig2, 320)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Decision matrix ──
    st.markdown('<div class="section-header">FTL vs Carting — Recommendation Matrix by Distance × Time Band</div>',
                unsafe_allow_html=True)

    df["time_band"] = pd.cut(df["hour"],
                              bins=[-1, 6, 12, 18, 23],
                              labels=["Night (0–6)", "Morning (6–12)", "Afternoon (12–18)", "Evening (18–24)"])

    matrix = df.groupby(["dist_band", "time_band", "route_type"])["factor"].median().unstack("route_type").reset_index()
    if "FTL" in matrix.columns and "Carting" in matrix.columns:
        matrix["Winner"] = np.where(matrix["FTL"] < matrix["Carting"], "FTL", "Carting")
        matrix["Delta"]  = (matrix["Carting"] - matrix["FTL"]).round(3)
        matrix["Rec"]    = matrix.apply(
            lambda r: f"✅ FTL (saves ×{abs(r['Delta']):.2f})" if r["Winner"] == "FTL"
                      else f"🛺 Carting (saves ×{abs(r['Delta']):.2f})", axis=1)
        st.dataframe(
            matrix[["dist_band", "time_band", "FTL", "Carting", "Delta", "Rec"]].rename(columns={
                "dist_band": "Distance", "time_band": "Time Band",
                "FTL": "FTL Delay ×", "Carting": "Carting Delay ×",
                "Delta": "Δ (Carting−FTL)", "Rec": "Recommendation"
            }).style.applymap(
                lambda v: f"color:{GREEN}" if "FTL" in str(v) else f"color:{AMBER}",
                subset=["Recommendation"]
            ).format({"FTL Delay ×": "{:.3f}", "Carting Delay ×": "{:.3f}", "Δ (Carting−FTL)": "{:+.3f}"}),
            use_container_width=True, height=300
        )

    # ── State-level FTL advantage ──
    st.markdown('<div class="section-header">FTL Advantage by State (Median Delay Gap)</div>',
                unsafe_allow_html=True)
    state_rt = df.groupby(["state", "route_type"])["factor"].median().unstack("route_type").dropna()
    if "FTL" in state_rt.columns and "Carting" in state_rt.columns:
        state_rt["FTL_Advantage"] = state_rt["Carting"] - state_rt["FTL"]
        state_rt = state_rt.sort_values("FTL_Advantage", ascending=False).reset_index()
        fig3 = go.Figure(go.Bar(
            x=state_rt["state"],
            y=state_rt["FTL_Advantage"],
            marker_color=[GREEN if v > 0 else RED for v in state_rt["FTL_Advantage"]],
            text=state_rt["FTL_Advantage"].round(2),
            textposition="outside"
        ))
        fig3.add_hline(y=0, line_color=MUTED)
        fig3.update_layout(xaxis_title="", yaxis_title="Carting − FTL Median Delay",
                           xaxis_tickangle=-45)
        fig3.add_annotation(text="← FTL Worse | FTL Better →",
                            xref="paper", yref="paper", x=0.5, y=1.05,
                            showarrow=False, font=dict(color=MUTED, size=10))
        plotly_defaults(fig3, 320)
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 6 ─ DELAY RISK SCORE
# ─────────────────────────────────────────────
def tab_risk_scoring(df, hubs, sla):
    st.markdown('<div class="section-header">Real-Time Delay Risk Intelligence</div>', unsafe_allow_html=True)
    st.caption("Composite risk scores combining graph centrality, corridor breach history, and live delay patterns.")

    # ── Hub risk tier donut ──
    c1, c2, c3 = st.columns(3)

    with c1:
        tier_counts = hubs["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig = go.Figure(go.Pie(
            labels=tier_counts["Tier"], values=tier_counts["Count"],
            hole=0.55,
            marker_colors=[GREEN, AMBER, ORANGE, RED],
            textfont=dict(color=TEXT)
        ))
        fig.update_layout(title=dict(text="Hub Risk Distribution", font=dict(color=TEXT)),
                          paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=40,b=0), height=280,
                          legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        breach_by_state = df.groupby("state").agg(
            breach_pct=("is_breach", lambda x: x.mean()*100),
            trips=("factor","count")
        ).reset_index().sort_values("breach_pct", ascending=False).head(12)

        fig2 = px.bar(breach_by_state, x="breach_pct", y="state",
                      orientation="h", color="breach_pct",
                      color_continuous_scale=[[0, AMBER], [0.7, ORANGE], [1, RED]],
                      text=breach_by_state["breach_pct"].round(1).astype(str)+"%")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(title=dict(text="SLA Breach % by State", font=dict(color=TEXT)),
                           yaxis=dict(autorange="reversed"), xaxis_title="Breach %", yaxis_title="",
                           coloraxis_showscale=False)
        plotly_defaults(fig2, 320)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        # Scatter: corridor risk vs trips
        fig3 = px.scatter(sla.head(50), x="total_trips", y="max_delay_ratio",
                          size="total_breach_score", color="corridor_risk",
                          color_continuous_scale=[[0, GREEN], [0.5, AMBER], [1.0, RED]],
                          hover_data=["source_name", "destination_name"],
                          size_max=30,
                          labels={"total_trips": "Trip Volume",
                                  "max_delay_ratio": "Max Delay ×",
                                  "corridor_risk": "Risk"})
        fig3.update_layout(title=dict(text="Corridor Risk Bubble Chart", font=dict(color=TEXT)),
                           coloraxis_colorbar_tickfont_color=TEXT)
        plotly_defaults(fig3, 320)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Risk calculator ──
    st.markdown("---")
    st.markdown('<div class="section-header">🔢 Live Risk Score Calculator</div>', unsafe_allow_html=True)
    st.caption("Estimate delay risk for a new shipment based on corridor and hub properties.")

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        inp_bc   = st.slider("Source Hub Betweenness", 0.0, 0.15, 0.05, 0.005)
    with rc2:
        inp_dist = st.number_input("Distance (km)", min_value=1, max_value=5000, value=150)
    with rc3:
        inp_hour = st.slider("Departure Hour", 0, 23, 10)
    with rc4:
        inp_rt   = st.selectbox("Route Type", ["FTL", "Carting"])

    # Simple rule-based risk model
    base = 1.85  # median factor
    bc_adj  = inp_bc * 15
    dist_adj = 0.1 if inp_dist > 200 else (-0.05 if inp_dist < 50 else 0)
    hr_adj  = 0.15 if inp_hour in [2, 3, 4] else (0.08 if inp_hour in [19, 20, 21] else 0)
    rt_adj  = 0.15 if inp_rt == "Carting" else 0
    pred_factor = base + bc_adj + dist_adj + hr_adj + rt_adj

    risk_pct = min(100, max(0, (pred_factor - 1) / 4 * 100))

    sr1, sr2, sr3 = st.columns(3)
    with sr1:
        st.markdown(kpi_html(f"×{pred_factor:.2f}", "Predicted Delay Factor",
                             risk_color(risk_pct)),
                    unsafe_allow_html=True)
    with sr2:
        sla_breach = "YES ⚠️" if pred_factor > 1.2 else "NO ✅"
        st.markdown(kpi_html(sla_breach, "SLA Breach Risk",
                             RED if pred_factor > 1.2 else GREEN),
                    unsafe_allow_html=True)
    with sr3:
        tier = "Critical" if risk_pct > 65 else ("High" if risk_pct > 40 else ("Medium" if risk_pct > 20 else "Low"))
        st.markdown(kpi_html(tier, "Risk Tier", risk_color(risk_pct)), unsafe_allow_html=True)

    # ── Gauge ──
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_factor,
        title=dict(text="Predicted Delay Factor", font=dict(color=TEXT, size=14)),
        delta=dict(reference=1.2, valueformat=".2f"),
        gauge=dict(
            axis=dict(range=[1, 5], tickfont=dict(color=TEXT)),
            bar=dict(color=risk_color(risk_pct)),
            steps=[
                dict(range=[1, 1.2], color=GREEN),
                dict(range=[1.2, 2.0], color=AMBER),
                dict(range=[2.0, 3.5], color=ORANGE),
                dict(range=[3.5, 5], color=RED),
            ],
            threshold=dict(line=dict(color=RED, width=3), thickness=0.75, value=1.2)
        )
    ))
    fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
                         height=250, margin=dict(l=20,r=20,t=40,b=10))
    st.plotly_chart(fig_g, use_container_width=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── load ──
    with st.spinner("Loading Delhivery network data…"):
        df, hubs, sla = load_data()

    # ── sidebar filters ──
    sel_states, sel_rt, sel_dist, delay_thresh, top_n_hubs, top_n_corr = sidebar(hubs, sla, df)

    # ── apply filters ──
    df_f = df.copy()
    if sel_states:
        df_f = df_f[df_f["state"].isin(sel_states)]
    if sel_rt:
        df_f = df_f[df_f["route_type"].isin(sel_rt)]
    if sel_dist:
        df_f = df_f[df_f["dist_band"].astype(str).isin(sel_dist)]
    df_f["is_breach"] = df_f["factor"] > delay_thresh

    # ── title bar ──
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:16px;padding:8px 0 20px">'
        f'<span style="font-size:2.2rem">🚚</span>'
        f'<div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{TEXT}">Delhivery Network Intelligence</div>'
        f'<div style="font-size:0.82rem;color:{MUTED}">Real-time delay risk scoring · Bottleneck detection · Corridor audit · Route decision engine</div>'
        f'</div></div>',
        unsafe_allow_html=True)

    # ── tabs ──
    tabs = st.tabs([
        "📊 Overview",
        "🌐 Network Graph",
        "🔴 Bottleneck Hubs",
        "🛣️ Corridor Audit",
        "🚛 FTL vs Carting",
        "⚡ Risk Scoring",
    ])

    with tabs[0]: tab_overview(df_f, hubs, sla, delay_thresh)
    with tabs[1]: tab_network(hubs, sla, top_n_hubs, top_n_corr)
    with tabs[2]: tab_hubs(hubs)
    with tabs[3]: tab_corridors(sla, df_f)
    with tabs[4]: tab_ftl_carting(df_f)
    with tabs[5]: tab_risk_scoring(df_f, hubs, sla)


if __name__ == "__main__":
    main()