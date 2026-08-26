import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from ml_utils import (
    AUDIO_FEATURES, RADAR_FEATURES, LOGIN_FEATURES,
    run_pca, get_recommendations, run_clustering, cluster_profile, find_outliers, top_loadings,
)
from spotify_utils import (
    sp_client, genre_counts, build_track_feature_df, genre_pct,
    nostalgia_index, taste_drift, discovery_vs_comfort, think_you_like_x,
    mood_by_time_of_day, music_age_gap, forgotten_library, music_bias,
    obsession_index, music_dna,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Musical Landscape",
    layout="wide",
    page_icon="🎧",
)

# ─────────────────────────────────────────────
# THEME  (light, minimalist)
# ─────────────────────────────────────────────
ACCENT = "#4A6CF7"
BG = "#FFFFFF"
CARD_BG = "#F7F7F9"
TEXT_MAIN = "#1A1A1A"
TEXT_SOFT = "#6B6B6F"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    html, body, .stApp {{ background-color: {BG}; font-family: 'Poppins', sans-serif; }}
    h1, h2, h3, h4 {{ color: {TEXT_MAIN}; font-family: 'Poppins', sans-serif; font-weight: 600; }}
    p, li, label {{ color: {TEXT_SOFT}; }}

    .stButton > button {{
        background-color: {TEXT_MAIN};
        color: #FFFFFF;
        border: none;
        border-radius: 999px;
        font-weight: 500;
        padding: 0.5rem 1.4rem;
    }}
    .stButton > button:hover {{ background-color: {ACCENT}; }}

    [data-testid="stFileUploadDropzone"] {{
        background-color: {CARD_BG};
        border: 1.5px dashed {ACCENT};
        border-radius: 12px;
    }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; background: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        background: {CARD_BG};
        border-radius: 999px;
        color: {TEXT_SOFT};
        padding: 6px 18px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{ background: {ACCENT} !important; color: #FFF !important; }}

    [data-testid="metric-container"] {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 12px 18px;
        border-left: 3px solid {ACCENT};
    }}

    .ml-badge {{
        display: inline-block; background: #E9EDFF; color: {ACCENT};
        font-size: 12px; padding: 3px 10px; border-radius: 999px; margin-left: 8px;
    }}
    hr {{ border-color: {CARD_BG}; }}
</style>
""", unsafe_allow_html=True)

PLOT_BG, PAPER_BG, GRID_CLR, TEXT_CLR = BG, BG, "#EAEAEE", TEXT_MAIN


def light_fig(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(ACCENT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5)
    return fig, ax


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; padding: 2rem 0 0.5rem;'>
    <p style='color:{TEXT_SOFT}; font-size:0.9rem; letter-spacing:1px; margin:0;'>MUSICAL LANDSCAPE</p>
    <h1 style='font-size:2.2rem; margin:6px 0 0;'>Your music, mapped</h1>
    <p style='font-size:1rem; margin-top:0.3rem;'>Connect Spotify to see what your listening says about you</p>
</div>
""", unsafe_allow_html=True)

sp = sp_client(st)

with st.expander("Or upload a CSV instead"):
    st.caption("Due to Spotify's developer restrictions, only a limited set of data is available through login. Upload the Kaggle Spotify Tracks dataset for the full audio-feature analysis.")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"], label_visibility="collapsed")
    st.markdown(
        "[Get the dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)"
    )

st.markdown("---")

# ═════════════════════════════════════════════
# LOGIN PATH — ML first, personality stats second
# ═════════════════════════════════════════════
if sp is not None:
    me = sp.current_user()
    st.success(f"Connected as {me['display_name']}")

    top = sp.current_user_top_tracks(limit=20, time_range="medium_term")["items"]
    saved_items = sp.current_user_saved_tracks(limit=20)["items"]
    saved = [item['track'] for item in saved_items]
    recent_items = sp.current_user_recently_played(limit=50)["items"]
    recent = [item['track'] for item in recent_items]

    top_df = build_track_feature_df(top)
    saved_df = build_track_feature_df(saved)
    recent_df = build_track_feature_df(recent)
    combined_df = pd.concat([top_df, saved_df, recent_df]).drop_duplicates(subset='track_id').reset_index(drop=True)

    # ── ML section ──
    st.markdown(f"### Analysis <span class='ml-badge'>ML powered</span>", unsafe_allow_html=True)

    if len(combined_df) < 5:
        st.info("Not enough track data yet to run the analysis. Listen to a bit more and check back.")
    else:
        scaled, pca_model, components = run_pca(combined_df, feature_cols=LOGIN_FEATURES)
        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df['track'] = combined_df['track_name'].values
        pca_df['artist'] = combined_df['artists'].values
        var1 = round(pca_model.explained_variance_ratio_[0] * 100, 1)
        var2 = round(pca_model.explained_variance_ratio_[1] * 100, 1)

        tab_pca, tab_cluster, tab_recommend, tab_outlier = st.tabs(
            ["Musical landscape", "Discover clusters", "Find similar tracks", "Outlier finder"]
        )

        with tab_pca:
            st.caption("Built from duration, explicit flag, and release year — Spotify no longer exposes audio features to new apps, so this runs on the metadata that's still available.")
            pc1_top, pc2_top = top_loadings(pca_model, LOGIN_FEATURES)
            st.info(f"PC1 is mostly driven by {', '.join(pc1_top)}. PC2 is mostly driven by {', '.join(pc2_top)}. Tracks close together on this chart share similar values across those features.")
            fig = px.scatter(
                pca_df, x="PC1", y="PC2", hover_data=['track', 'artist'],
                labels={"PC1": f"PC1 ({var1}%)", "PC2": f"PC2 ({var2}%)"},
                template="plotly_white", color_discrete_sequence=[ACCENT], opacity=0.7,
            )
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT_MAIN)
            st.plotly_chart(fig, use_container_width=True)

        with tab_cluster:
            k = st.slider("Number of clusters", 2, 6, 3, key="sp_k")
            labels, _ = run_clustering(scaled, k)
            cdf = pca_df.copy()
            cdf['cluster'] = labels.astype(str)
            fig = px.scatter(
                cdf, x="PC1", y="PC2", color="cluster",
                template="plotly_white", opacity=0.7,
            )
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT_MAIN)
            st.plotly_chart(fig, use_container_width=True)

        with tab_recommend:
            track_options = combined_df['track_name'] + " — " + combined_df['artists']
            chosen = st.selectbox("Choose a track", track_options, key="sp_rec")
            pos = track_options[track_options == chosen].index[0]
            n_recs = st.slider("Number of recommendations", 3, 10, 5, key="sp_rec_n")
            recs = get_recommendations(pos, scaled, combined_df, n=n_recs)
            st.dataframe(
                recs[['track_name', 'artists', 'similarity']].style.format({'similarity': '{:.3f}'}),
                use_container_width=True,
            )

        with tab_outlier:
            n_out = st.slider("Number of outliers", 3, 10, 5, key="sp_out_n")
            outliers = find_outliers(scaled, combined_df, n=n_out)
            st.dataframe(
                outliers[['track_name', 'artists', 'anomaly_score']].style.format({'anomaly_score': '{:.3f}'}),
                use_container_width=True,
            )

    st.markdown("---")

    # ── personality section ──
    st.markdown("### Your listening personality")

    top_ids = {t['id'] for t in top}
    saved_ids = [t['id'] for t in saved]
    recent_ids = [t['id'] for t in recent]

    top_artist_ids = [a['id'] for t in top for a in t['artists']]
    saved_artist_ids = [a['id'] for t in saved for a in t['artists']]
    recent_artist_ids = [a['id'] for t in recent for a in t['artists']]

    saved_genres = genre_counts(sp, saved_artist_ids)
    recent_genres = genre_counts(sp, recent_artist_ids)
    saved_pct = genre_pct(saved_genres)
    recent_pct = genre_pct(recent_genres)

    scores = {}

    tab_nost, tab_drift, tab_disc, tab_like, tab_mood, tab_age, tab_forgot, tab_bias, tab_dna = st.tabs([
        "Nostalgia", "Taste drift", "Discovery vs comfort", "Think you like X",
        "Mood by time of day", "Music age gap", "Forgotten library", "Music bias", "Music DNA",
    ])

    with tab_nost:
        if len(saved_df):
            n = nostalgia_index(saved_df)
            scores['nostalgia'] = n['score']
            st.metric("Nostalgia index", f"{n['score']}/100")
            st.write(f"Average track age: {n['avg_age_years']} years")
            st.write(f"Oldest track in your saved list: {n['oldest_track']} ({n['oldest_year']})")
        else:
            st.info("Save some tracks to see this.")

    with tab_drift:
        drift = taste_drift(saved_pct, recent_pct)
        scores['taste_drift'] = drift
        st.metric("Taste drift", f"{drift}/100")
        st.caption("Higher means your recent listening looks quite different from what you've saved.")

    with tab_disc:
        dvc = discovery_vs_comfort(recent, top_ids)
        scores['discovery_pct'] = dvc['discovery_pct']
        c1, c2 = st.columns(2)
        c1.metric("Discovery", f"{dvc['discovery_pct']}%")
        c2.metric("Comfort", f"{dvc['comfort_pct']}%")

    with tab_like:
        tylx = think_you_like_x(saved_pct, recent_pct)
        if tylx:
            if tylx['mismatch']:
                st.write(f"You save a lot of {tylx['saved_top_genre']}, but you're actually listening to {tylx['recent_top_genre']} the most.")
            else:
                st.write(f"Your saved and recent listening agree — mostly {tylx['saved_top_genre']}.")
        else:
            st.info("Not enough genre data yet.")

    with tab_mood:
        mood_df = mood_by_time_of_day(recent_items)
        if len(mood_df):
            st.dataframe(mood_df, use_container_width=True)
        else:
            st.info("Not enough recently-played data with timestamps yet.")

    with tab_age:
        gap = music_age_gap(top_df, saved_df, recent_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Top tracks", f"{gap['top']} yrs old" if gap['top'] is not None else "—")
        c2.metric("Saved tracks", f"{gap['saved']} yrs old" if gap['saved'] is not None else "—")
        c3.metric("Recently played", f"{gap['recent']} yrs old" if gap['recent'] is not None else "—")

    with tab_forgot:
        fg = forgotten_library(saved_ids, recent_ids, len(saved_ids))
        st.metric("Forgotten tracks", f"{fg['forgotten_count']} ({fg['forgotten_pct']}%)")
        st.caption("Saved tracks that haven't shown up in your recent listening.")

    with tab_bias:
        bias = music_bias(combined_df)
        if bias:
            st.write(f"Your biggest bias: {bias['pct']}% of your tracks fall into '{bias['label']}'.")
        else:
            st.info("Not enough data yet.")

    with tab_dna:
        obs = obsession_index(top)
        scores['obsession'] = obs['score']
        dna = music_dna(scores)
        if dna:
            for label, val in dna.items():
                st.metric(label, val)
        st.caption(f"Most played artist in your top tracks: {obs['top_artist']}")

else:
    st.info("Connect your Spotify account above to see your analysis.")

# ═════════════════════════════════════════════
# CSV PATH — secondary, full audio-feature pipeline
# ═════════════════════════════════════════════
if uploaded_file is not None:

    st.markdown("---")
    st.markdown("### CSV analysis")
    st.caption("Full audio-feature analysis on the uploaded dataset.")

    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df_clean = df.dropna(subset=AUDIO_FEATURES).copy()
        return df, df_clean

    df_raw, df = load_data(uploaded_file)
    missing = [f for f in AUDIO_FEATURES if f not in df.columns]

    if missing:
        st.error(f"These expected columns are missing: {missing}")
    else:
        has_genre = 'track_genre' in df.columns

        with st.expander("Filters"):
            sample_size = st.slider("Sample size (rows)", 500, min(10_000, len(df)), min(3_000, len(df)), step=500)
            genre_filter = []
            if has_genre:
                all_genres = sorted(df['track_genre'].dropna().unique().tolist())
                genre_filter = st.multiselect("Select genres to show (leave empty = all)", all_genres, default=[])

        df_work = df.copy()
        if has_genre and genre_filter:
            df_work = df_work[df_work['track_genre'].isin(genre_filter)]
        df_work = df_work.sample(n=min(sample_size, len(df_work)), random_state=42).reset_index(drop=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Tracks loaded", f"{len(df_work):,}")
        col2.metric("Features used", len(AUDIO_FEATURES))
        col3.metric("Genres in view", df_work['track_genre'].nunique() if has_genre else "N/A")

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

        (tab_heatmap, tab_variance, tab_landscape, tab_compare,
         tab_recommend_csv, tab_cluster_csv, tab_outlier_csv) = st.tabs([
            "Correlation heatmap", "Explained variance", "Musical landscape", "Genre comparison",
            "Find similar tracks", "Discover clusters", "Outlier finder",
        ])

        with tab_heatmap:
            corr_matrix = pd.DataFrame(scaled, columns=AUDIO_FEATURES).corr()
            fig, ax = light_fig(11, 8)
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlBu", ax=ax,
                        annot_kws={"size": 9}, linewidths=0.5, linecolor=BG, cbar_kws={"shrink": 0.8})
            ax.set_title("Audio feature correlations", fontsize=14, pad=15)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            fig.tight_layout()
            st.pyplot(fig)

        with tab_variance:
            pca_full = PCA().fit(scaled)
            var_ratios = pca_full.explained_variance_ratio_ * 100
            cumulative = np.cumsum(var_ratios)
            fig, ax = light_fig(10, 5)
            ax.bar(range(1, len(AUDIO_FEATURES) + 1), var_ratios, color=ACCENT, alpha=0.85, zorder=3)
            ax.plot(range(1, len(AUDIO_FEATURES) + 1), cumulative, color=TEXT_MAIN, marker='o', linewidth=2, zorder=4)
            ax.set_xlabel("Principal component")
            ax.set_ylabel("Explained variance (%)")
            fig.tight_layout()
            st.pyplot(fig)
            c1, c2 = st.columns(2)
            c1.metric("PC1 variance", f"{var1}%")
            c2.metric("PC2 variance", f"{var2}%")

        with tab_landscape:
            pc1_top, pc2_top = top_loadings(pca_model, AUDIO_FEATURES)
            st.info(f"PC1 is mostly driven by {', '.join(pc1_top)}. PC2 is mostly driven by {', '.join(pc2_top)}. Tracks close together on this chart share similar values across those features.")
            hover_data = {k: True for k in ['track', 'artist'] if k in pca_df.columns}
            fig = px.scatter(
                pca_df, x="PC1", y="PC2", color="genre" if has_genre else None,
                hover_data=hover_data or None,
                labels={"PC1": f"PC1 ({var1}%)", "PC2": f"PC2 ({var2}%)"},
                template="plotly_white", opacity=0.6, render_mode="webgl",
            )
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT_MAIN)
            st.plotly_chart(fig, use_container_width=True)

        with tab_compare:
            if not has_genre:
                st.warning("This dataset doesn't have a track_genre column.")
            else:
                all_g = sorted(df['track_genre'].dropna().unique().tolist())
                chosen = st.multiselect("Choose genres to compare (2-6 recommended)", all_g,
                                         default=all_g[:3] if len(all_g) >= 3 else all_g)
                if len(chosen) < 2:
                    st.info("Select at least 2 genres to compare.")
                else:
                    df_cmp = df[df['track_genre'].isin(chosen)].dropna(subset=AUDIO_FEATURES)
                    per_genre = min(500, df_cmp['track_genre'].value_counts().min())
                    df_cmp = df_cmp.groupby('track_genre', group_keys=False).apply(
                        lambda x: x.sample(per_genre, random_state=42))
                    sc_scaled, sc_pca, sc_comp = run_pca(df_cmp)
                    cmp_df = pd.DataFrame(sc_comp, columns=["PC1", "PC2"])
                    cmp_df['genre'] = df_cmp['track_genre'].values

                    fig = px.scatter(cmp_df, x="PC1", y="PC2", color="genre", template="plotly_white", opacity=0.65)
                    fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT_MAIN)
                    st.plotly_chart(fig, use_container_width=True)

                    means = df_cmp.groupby('track_genre')[RADAR_FEATURES].mean()
                    radar = go.Figure()
                    for genre_name in chosen:
                        if genre_name not in means.index:
                            continue
                        vals = means.loc[genre_name, RADAR_FEATURES].tolist()
                        vals += vals[:1]
                        radar.add_trace(go.Scatterpolar(
                            r=vals, theta=RADAR_FEATURES + [RADAR_FEATURES[0]],
                            fill='toself', name=genre_name, opacity=0.6,
                        ))
                    radar.update_layout(paper_bgcolor=BG, font_color=TEXT_MAIN)
                    st.plotly_chart(radar, use_container_width=True)

                    st.dataframe(means.style.format("{:.3f}"), use_container_width=True)

        with tab_recommend_csv:
            if 'track_name' not in df_work.columns:
                st.warning("Dataset needs a track_name column for this feature.")
            else:
                track_options = df_work['track_name'] + " — " + df_work.get('artists', '')
                chosen_track = st.selectbox("Choose a track", track_options, key="csv_rec")
                pos = df_work.index.get_loc(track_options[track_options == chosen_track].index[0])
                n_recs = st.slider("Number of recommendations", 5, 20, 10, key="csv_rec_n")
                recs = get_recommendations(pos, scaled, df_work, n=n_recs)
                cols_show = [c for c in ['track_name', 'artists', 'track_genre', 'similarity'] if c in recs.columns]
                st.dataframe(recs[cols_show].style.format({'similarity': '{:.3f}'}), use_container_width=True)

        with tab_cluster_csv:
            k = st.slider("Number of clusters", 2, 15, 8, key="csv_k")
            labels, _ = run_clustering(scaled, k)
            cdf = pca_df.copy()
            cdf['cluster'] = labels.astype(str)
            fig = px.scatter(cdf, x="PC1", y="PC2", color="cluster", template="plotly_white", opacity=0.6)
            fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT_MAIN)
            st.plotly_chart(fig, use_container_width=True)
            deviation = cluster_profile(scaled, AUDIO_FEATURES, labels)
            st.dataframe(deviation.style.background_gradient(cmap='RdYlBu', axis=None).format("{:.2f}"),
                         use_container_width=True)

        with tab_outlier_csv:
            n_out = st.slider("Number of outliers", 5, 30, 10, key="csv_out_n")
            outliers = find_outliers(scaled, df_work, n=n_out)
            cols_show = [c for c in ['track_name', 'artists', 'track_genre', 'anomaly_score'] if c in outliers.columns]
            st.dataframe(outliers[cols_show].style.format({'anomaly_score': '{:.3f}'}), use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:{TEXT_SOFT}; font-size:0.85rem; padding: 1rem 0;'>
    Built with Streamlit · Scikit-learn · Plotly · Spotify Web API
</div>
""", unsafe_allow_html=True)
