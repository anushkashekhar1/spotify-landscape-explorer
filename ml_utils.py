import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# full audio-feature set, used by the CSV path (Kaggle dataset has these columns)
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms',
]

# radar chart only needs the 0-1 scale features, tempo/loudness/duration are on different scales
RADAR_FEATURES = [
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
]

# thinner feature set used for logged-in users, since Spotify no longer exposes
# audio-feature data (danceability, energy, etc.) to new developer apps
LOGIN_FEATURES = ['duration_ms', 'explicit', 'release_year']


def run_pca(df_clean, feature_cols=AUDIO_FEATURES, n=2):
    """Scale the given feature columns and reduce to n components."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean[feature_cols])
    pca = PCA(n_components=n)
    components = pca.fit_transform(scaled)
    return scaled, pca, components


def get_recommendations(track_idx, scaled, df_ref, n=10):
    """Nearest neighbors to a given track's position in scaled feature space."""
    nn = NearestNeighbors(n_neighbors=n + 1, metric='cosine')
    nn.fit(scaled)
    distances, indices = nn.kneighbors(scaled[track_idx].reshape(1, -1))
    # first result is the track itself, skip it
    rec_idx = indices[0][1:]
    rec_dist = distances[0][1:]
    out = df_ref.iloc[rec_idx].copy()
    out['similarity'] = 1 - rec_dist
    return out


def run_clustering(scaled, k):
    """KMeans clustering on scaled feature data."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    return labels, km


def cluster_profile(scaled, feature_cols, labels):
    """Mean value per feature per cluster, compared to the overall mean.
    Positive = above average for that feature, negative = below."""
    df_feat = pd.DataFrame(scaled, columns=feature_cols)
    df_feat['cluster'] = labels
    cluster_means = df_feat.groupby('cluster')[feature_cols].mean()
    overall_mean = df_feat[feature_cols].mean()
    return cluster_means - overall_mean


def top_loadings(pca_model, feature_cols, n=3):
    """Which original features contribute most to PC1 and PC2, for a plain-
    language explanation of what the axes actually mean."""
    loadings = pca_model.components_
    pc1 = pd.Series(loadings[0], index=feature_cols).abs().sort_values(ascending=False).index[:n].tolist()
    pc2 = pd.Series(loadings[1], index=feature_cols).abs().sort_values(ascending=False).index[:n].tolist()
    return pc1, pc2


def find_outliers(feat_matrix, df_ref, n=10):
    """Most unusual rows, based on Isolation Forest anomaly score.
    Lower score = more unusual."""
    iso = IsolationForest(contamination='auto', random_state=42)
    iso.fit(feat_matrix)
    anomaly_score = iso.score_samples(feat_matrix)
    out = df_ref.copy()
    out['anomaly_score'] = anomaly_score
    return out.sort_values('anomaly_score').head(n)
