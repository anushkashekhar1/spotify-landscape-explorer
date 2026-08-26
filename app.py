import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import spotipy 
from spotipy.oauth2 import SpotifyOAuth
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Musical Landscape",
    layout="wide",
    page_icon="🎧",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES  (Spotify dark theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── base ── */
    .stApp                          { background-color: #121212; color: #FFFFFF; }
    section[data-testid="stSidebar"]{ background-color: #000000; }

    /* ── headings ── */
    h1, h2, h3, h4                  { color: #1DB954; font-family: 'Circular', 'Helvetica Neue', sans-serif; }
    p, li, label                    { color: #B3B3B3; }

    /* ── green button ── */
    .stButton > button {
        background-color: #1DB954;
        color: #000000;
        border: none;
        border-radius: 500px;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        transition: transform 80ms ease;
    }
    .stButton > button:hover        { transform: scale(1.04); background-color: #1ed760; }

    /* ── file uploader ── */
    [data-testid="stFileUploadDropzone"] {
        background-color: #282828;
        border: 2px dashed #1DB954;
        border-radius: 12px;
    }

    /* ── tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #282828;
        border-radius: 500px;
        color: #B3B3B3;
        padding: 6px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"]   { background: #1DB954 !important; color: #000 !important; }

    /* ── metric cards ── */
    [data-testid="metric-container"] {
        background: #282828;
        border-radius: 12px;
        padding: 12px 18px;
        border-left: 4px solid #1DB954;
    }
    [data-testid="metric-container"] label { color: #B3B3B3 !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1DB954 !important; }

    /* ── multiselect tags ── */
    span[data-baseweb="tag"]         { background-color: #1DB954 !important; color: #000 !important; }

    /* ── divider ── */
    hr { border-color: #282828; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms',
]

PLOT_BG   = '#121212'
PAPER_BG  = '#121212'
GRID_CLR  = '#282828'
GREEN     = '#1DB954'
TEXT_CLR  = '#FFFFFF'

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def dark_fig(w=10, h=6):
    """Return a matplotlib figure pre-styled for the dark theme."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors=TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(GREEN)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5)
    return fig, ax


def run_pca(df_clean, n=2):
    scaler     = StandardScaler()
    scaled     = scaler.fit_transform(df_clean[FEATURES])
    pca        = PCA(n_components=n)
    components = pca.fit_transform(scaled)
    return scaled, pca, components

def get_recommendations(track_idx, scaled, df_ref, n=10):
    nn = NearestNeighbors(n_neighbors=n + 1, metric='cosine')
    nn.fit(scaled)
    distances, indices = nn.kneighbors(scaled[track_idx].reshape(1, -1))
    # first result is the track itself, skip it
    rec_idx = indices[0][1:]
    rec_dist = distances[0][1:]
    out = df_ref.iloc[rec_idx].copy()
    out['similarity'] = 1 - rec_dist  # cosine distance -> similarity
    return out


def run_clustering(scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    return labels, km

SP_SCOPES = "user-top-read user-library-read user-read-recently-played"

def sp_oauth():
    return SpotifyOAuth(
        client_id=st.secrets["spotify"]["client_id"],
        client_secret=st.secrets["spotify"]["client_secret"],
        redirect_uri=st.secrets["spotify"]["redirect_uri"],
        scope=SP_SCOPES,
        cache_path=None,
        show_dialog=True,
    )

def sp_client():
    oauth = sp_oauth()
    qp = st.query_params
    if "code" in qp and "sp_token" not in st.session_state:
        token = oauth.get_access_token(qp["code"], as_dict=True, check_cache=False)
        st.session_state["sp_token"] = token["access_token"]
        st.query_params.clear()
    if "sp_token" not in st.session_state:
        st.link_button("🔗 Connect with Spotify", oauth.get_authorize_url())
        return None
    return spotipy.Spotify(auth=st.session_state["sp_token"])

def genre_counts(sp, artist_ids):
    ids = list(dict.fromkeys(artist_ids))
    counts = {}
    for i in range(0, len(ids), 50):
        for artist in sp.artists(ids[i:i+50])["artists"]:
            for g in artist["genres"]:
                counts[g] = counts.get(g, 0) + 1
    return counts


def render_my_spotify():
    st.markdown("## 🔐 My Spotify")
    sp = sp_client()
    if sp is None:
        st.info("Connect your account to see your top tracks, liked songs, and genre breakdown.")
        return

    me = sp.current_user()
    st.success(f"Connected as **{me['display_name']}**")

    time_range = st.radio("Time range", ["short_term", "medium_term", "long_term"],
                           format_func=lambda x: {"short_term":"Last 4 weeks","medium_term":"Last 6 months","long_term":"All time"}[x],
                           horizontal=True)

    top = sp.current_user_top_tracks(limit=20, time_range=time_range)["items"]
    saved = sp.current_user_saved_tracks(limit=20)["items"]

    tab_top, tab_saved, tab_genres = st.tabs(["🔥 Top tracks", "💚 Liked songs", "🎼 Genre breakdown"])

    with tab_top:
        for i, t in enumerate(top, 1):
            artists = ", ".join(a["name"] for a in t["artists"])
            st.write(f"{i}. **{t['name']}** — {artists}")

    with tab_saved:
        for i, item in enumerate(saved, 1):
            t = item["track"]
            artists = ", ".join(a["name"] for a in t["artists"])
            st.write(f"{i}. **{t['name']}** — {artists}")

    with tab_genres:
        artist_ids = [a["id"] for t in top for a in t["artists"]]
        gc = genre_counts(sp, artist_ids)
        if gc:
            gdf = pd.DataFrame(sorted(gc.items(), key=lambda x: -x[1])[:15], columns=["genre","count"])
            fig = px.bar(gdf, x="count", y="genre", orientation="h",
                         template="plotly_dark", color_discrete_sequence=[GREEN])
            fig.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=TEXT_CLR,
                              yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough top track data yet to build a genre breakdown.")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem;'>
    <span style='font-size:3rem;'>🎧</span>
    <h1 style='font-size:2.6rem; margin:0;'>Spotify Musical Landscape</h1>
    <p style='color:#B3B3B3; font-size:1.05rem; margin-top:0.4rem;'>
        Dimensionality reduction · PCA · Genre comparison
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

render_my_spotify()
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.markdown("""
    <small>
    Download the dataset from<br>
    <a href='https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset'
       style='color:#1DB954;' target='_blank'>Kaggle → Spotify Tracks Dataset</a>
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LANDING STATE (no file yet)
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style='text-align:center; padding:4rem 2rem;'>
        <p style='font-size:1.2rem; color:#B3B3B3;'>
            👈 Upload your Spotify dataset from the sidebar to get started
        </p>
        <p style='color:#535353;'>Supports the Kaggle Spotify Tracks Dataset (.csv)</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# LOAD & CLEAN DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df_clean = df.dropna(subset=FEATURES).copy()
    return df, df_clean

df_raw, df = load_data(uploaded_file)

# ── quick sanity check ──
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    st.error(f"❌ These expected columns are missing: {missing}\nCheck your dataset.")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR CONTINUED: genre picker
# ─────────────────────────────────────────────
has_genre = 'track_genre' in df.columns

with st.sidebar:
    st.markdown("---")
    sample_size = st.slider("Sample size (rows)", 500, min(10_000, len(df)), min(3_000, len(df)), step=500)

    genre_filter = []
    if has_genre:
        st.markdown("### 🎵 Genre filter")
        all_genres = sorted(df['track_genre'].dropna().unique().tolist())
        genre_filter = st.multiselect(
            "Select genres to show (leave empty = all)",
            all_genres,
            default=[],
        )

# ── apply filters ──
df_work = df.copy()
if has_genre and genre_filter:
    df_work = df_work[df_work['track_genre'].isin(genre_filter)]

df_work = df_work.sample(n=min(sample_size, len(df_work)), random_state=42).reset_index(drop=True)

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎵 Tracks loaded",   f"{len(df_work):,}")
col2.metric("📐 Features used",   len(FEATURES))
col3.metric("🎸 Genres in view",  df_work['track_genre'].nunique() if has_genre else "N/A")
col4.metric("🗜️ PCA output dims", "2")

st.markdown("---")

# ─────────────────────────────────────────────
# RUN PCA
# ─────────────────────────────────────────────
scaled, pca_model, components = run_pca(df_work)

pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
if has_genre:
    pca_df['genre'] = df_work['track_genre'].values
if 'track_name' in df_work.columns:
    pca_df['track'] = df_work['track_name'].values
if 'artists' in df_work.columns:
    pca_df['artist'] = df_work['artists'].values

var1 = round(pca_model.explained_variance_ratio_[0] * 100, 1)
var2 = round(pca_model.explained_variance_ratio_[1] * 100, 1)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_heatmap, tab_variance, tab_static, tab_interactive, tab_compare,  tab_recommend, tab_cluster = st.tabs([
    "🌡️ Correlation Heatmap",
    "📈 Explained Variance",
    "🗺️ Static Landscape",
    "✨ Interactive Landscape",
    "⚖️ Genre Comparison",
    "🎯 Find Similar Tracks",
    "🧩 Discover Clusters",
])

# ══════════════════════════════════════
# TAB 1 – CORRELATION HEATMAP
# ══════════════════════════════════════
with tab_heatmap:
    st.markdown("### Feature Correlation Heatmap")
    st.markdown("Shows how the 10 audio features relate to each other across all tracks.")

    corr_matrix = pd.DataFrame(scaled, columns=FEATURES).corr()
    fig, ax = dark_fig(11, 8)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=ax,
        annot_kws={"size": 9, "color": "white"},
        linewidths=0.5,
        linecolor=PLOT_BG,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Audio Feature Correlations", fontsize=14, pad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    fig.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════
# TAB 2 – EXPLAINED VARIANCE
# ══════════════════════════════════════
with tab_variance:
    st.markdown("### Explained Variance by Principal Component")
    st.markdown("How much of the data's total variation each component captures.")

    pca_full   = PCA().fit(scaled)
    var_ratios = pca_full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(var_ratios)

    fig, ax = dark_fig(10, 5)
    bars = ax.bar(range(1, len(FEATURES) + 1), var_ratios, color=GREEN, alpha=0.85, zorder=3)
    ax.plot(range(1, len(FEATURES) + 1), cumulative, color='#ffffff', marker='o',
            linewidth=2, markersize=6, label='Cumulative %', zorder=4)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree Plot with Cumulative Variance")
    ax.set_xticks(range(1, len(FEATURES) + 1))
    ax.legend(facecolor='#282828', edgecolor='none', labelcolor='white')
    # annotate bars
    for bar, v in zip(bars, var_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha='center', va='bottom', color=TEXT_CLR, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    c1.metric("PC1 variance", f"{var1}%")
    c2.metric("PC2 variance", f"{var2}%")

# ══════════════════════════════════════
# TAB 3 – STATIC LANDSCAPE
# ══════════════════════════════════════
with tab_static:
    st.markdown("### Musical Landscape — Static 2D View")
    st.markdown(f"PC1 explains **{var1}%** · PC2 explains **{var2}%** of total variance.")

    fig, ax = dark_fig(10, 7)
    scatter_kwargs = dict(alpha=0.35, s=6, linewidths=0)

    if has_genre and len(pca_df['genre'].unique()) <= 30:
        genres  = pca_df['genre'].unique()
        palette = plt.cm.get_cmap('tab20', len(genres))
        for i, g in enumerate(genres):
            mask = pca_df['genre'] == g
            ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                       color=[palette(i)], label=g, **scatter_kwargs)
        ax.legend(fontsize=7, facecolor='#282828', edgecolor='none',
                  labelcolor='white', ncol=2, loc='upper right')
    else:
        ax.scatter(pca_df["PC1"], pca_df["PC2"], color=GREEN, **scatter_kwargs)

    ax.set_xlabel(f"PC1  ({var1}% variance)")
    ax.set_ylabel(f"PC2  ({var2}% variance)")
    ax.set_title("Musical Landscape (PCA 2D Projection)")
    fig.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════
# TAB 4 – INTERACTIVE LANDSCAPE
# ══════════════════════════════════════
with tab_interactive:
    st.markdown("### Interactive Musical Landscape")
    st.markdown("Hover over points to see track info. Use the legend to toggle genres.")

    hover_data = {}
    if 'track' in pca_df.columns:  hover_data['track']  = True
    if 'artist' in pca_df.columns: hover_data['artist'] = True

    fig = px.scatter(
        pca_df, x="PC1", y="PC2",
        color="genre" if has_genre else None,
        hover_data=hover_data or None,
        labels={"PC1": f"PC1 ({var1}%)", "PC2": f"PC2 ({var2}%)"},
        title="Interactive Musical Landscape",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        opacity=0.6,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font_color=TEXT_CLR,
        legend=dict(bgcolor='#282828', bordercolor='#535353', borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════
# TAB 5 – GENRE COMPARISON  ← your feature
# ══════════════════════════════════════
with tab_compare:
    st.markdown("### ⚖️ Genre Comparison")
    st.markdown("Pick two or more genres and see exactly how they differ in audio feature profiles and PCA space.")

    if not has_genre:
        st.warning("This dataset doesn't have a `track_genre` column — genre comparison isn't available.")
    else:
        all_g = sorted(df['track_genre'].dropna().unique().tolist())

        gc1, gc2 = st.columns([2, 1])
        with gc1:
            chosen = st.multiselect(
                "Choose genres to compare (2–6 recommended)",
                all_g,
                default=all_g[:3] if len(all_g) >= 3 else all_g,
                key="compare_genres",
            )

        if len(chosen) < 2:
            st.info("Select at least 2 genres to compare.")
        else:
            # ── subset + PCA for just the chosen genres ──
            df_cmp   = df[df['track_genre'].isin(chosen)].copy()
            df_cmp   = df_cmp.dropna(subset=FEATURES)
            per_genre = min(500, df_cmp['track_genre'].value_counts().min())
            df_cmp   = (df_cmp.groupby('track_genre', group_keys=False)
                               .apply(lambda x: x.sample(per_genre, random_state=42)))

            sc2        = StandardScaler()
            sc_data    = sc2.fit_transform(df_cmp[FEATURES])
            pca2       = PCA(n_components=2)
            comp2      = pca2.fit_transform(sc_data)
            cmp_pca_df = pd.DataFrame(comp2, columns=["PC1","PC2"])
            cmp_pca_df['genre'] = df_cmp['track_genre'].values

            v1c = round(pca2.explained_variance_ratio_[0]*100,1)
            v2c = round(pca2.explained_variance_ratio_[1]*100,1)

            # ── row 1: PCA scatter ──
            st.markdown("#### PCA Space — genres side by side")
            fig_cmp = px.scatter(
                cmp_pca_df, x="PC1", y="PC2", color="genre",
                labels={"PC1": f"PC1 ({v1c}%)", "PC2": f"PC2 ({v2c}%)"},
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                opacity=0.65,
            )
            fig_cmp.update_traces(marker=dict(size=5))
            fig_cmp.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                font_color=TEXT_CLR,
                legend=dict(bgcolor='#282828'),
                margin=dict(l=40,r=40,t=40,b=40),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # ── row 2: radar chart of mean feature profiles ──
            st.markdown("#### Audio Feature Profiles (Radar Chart)")

            RADAR_FEATURES = ['danceability','energy','speechiness',
                              'acousticness','instrumentalness','liveness','valence']

            means = (df_cmp.groupby('track_genre')[RADAR_FEATURES].mean())

            radar_fig = go.Figure()
            colors_palette = px.colors.qualitative.Vivid
            for idx, genre_name in enumerate(chosen):
                if genre_name not in means.index:
                    continue
                vals = means.loc[genre_name, RADAR_FEATURES].tolist()
                vals += vals[:1]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=RADAR_FEATURES + [RADAR_FEATURES[0]],
                    fill='toself',
                    name=genre_name,
                    line_color=colors_palette[idx % len(colors_palette)],
                    opacity=0.6,
                ))
            radar_fig.update_layout(
                polar=dict(
                    bgcolor='#1a1a1a',
                    radialaxis=dict(visible=True, range=[0,1], color=TEXT_CLR, gridcolor=GRID_CLR),
                    angularaxis=dict(color=TEXT_CLR, gridcolor=GRID_CLR),
                ),
                paper_bgcolor=PAPER_BG,
                font_color=TEXT_CLR,
                legend=dict(bgcolor='#282828'),
                margin=dict(l=60,r=60,t=60,b=60),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            # ── row 3: bar chart for every feature ──
            st.markdown("#### Mean Feature Values — Bar Comparison")
            bar_data = means[RADAR_FEATURES].reset_index().melt(
                id_vars='track_genre', var_name='Feature', value_name='Mean Value'
            )
            bar_fig = px.bar(
                bar_data, x='Feature', y='Mean Value', color='track_genre',
                barmode='group',
                template='plotly_dark',
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            bar_fig.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                font_color=TEXT_CLR,
                legend=dict(bgcolor='#282828'),
                margin=dict(l=40,r=40,t=40,b=40),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # ── row 4: stats table ──
            st.markdown("#### Summary Statistics per Genre")
            st.dataframe(
                means[RADAR_FEATURES].style
                    .background_gradient(cmap='Greens', axis=None)
                    .format("{:.3f}"),
                use_container_width=True,
            )

with tab_recommend:
    st.markdown("### 🎯 Find Similar Tracks")
    st.markdown("Pick a track and find its nearest neighbors in audio-feature space.")

    if 'track_name' not in df_work.columns:
        st.warning("Dataset needs a `track_name` column for this feature.")
    else:
        track_options = df_work['track_name'] + " — " + df_work.get('artists', '')
        chosen_track = st.selectbox("Choose a track", track_options)
        track_idx = track_options[track_options == chosen_track].index[0]
        # position within df_work (0-based), not the pandas index label
        pos = df_work.index.get_loc(track_idx)

        n_recs = st.slider("Number of recommendations", 5, 20, 10)
        recs = get_recommendations(pos, scaled, df_work, n=n_recs)

        cols_to_show = [c for c in ['track_name', 'artists', 'track_genre', 'similarity'] if c in recs.columns]
        st.dataframe(
            recs[cols_to_show].style.format({'similarity': '{:.3f}'}).background_gradient(
                subset=['similarity'], cmap='Greens'
            ),
            use_container_width=True,
        )

with tab_cluster:
    st.markdown("### 🧩 Discover Clusters")
    st.markdown("Unsupervised clusters found directly in the audio features — no genre labels used.")

    k = st.slider("Number of clusters", 2, 15, 8)
    labels, km_model = run_clustering(scaled, k)

    cluster_df = pca_df.copy()
    cluster_df['cluster'] = labels.astype(str)

    fig_cluster = px.scatter(
        cluster_df, x="PC1", y="PC2", color="cluster",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        opacity=0.6,
        title="Clusters in PCA Space",
    )
    fig_cluster.update_traces(marker=dict(size=4))
    fig_cluster.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=TEXT_CLR)
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("#### What defines each cluster")
    df_feat = pd.DataFrame(scaled, columns=FEATURES)
    df_feat['cluster'] = labels
    cluster_means = df_feat.groupby('cluster')[FEATURES].mean()
    overall_mean = df_feat[FEATURES].mean()
    deviation = (cluster_means - overall_mean).style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}")
    st.dataframe(deviation, use_container_width=True)
    st.caption("Green = above dataset average, red = below. Values are in standardized (z-score) units.")
        
        

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#535353; font-size:0.85rem; padding: 1rem 0;'>
    Built with Streamlit · Scikit-learn · Plotly &nbsp;|&nbsp; Dataset: Kaggle Spotify Tracks
</div>
""", unsafe_allow_html=True)
