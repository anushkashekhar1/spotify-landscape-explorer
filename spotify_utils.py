import datetime
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

SP_SCOPES = "user-top-read user-library-read user-read-recently-played"


def sp_oauth(secrets):
    return SpotifyOAuth(
        client_id=secrets["spotify"]["client_id"],
        client_secret=secrets["spotify"]["client_secret"],
        redirect_uri=secrets["spotify"]["redirect_uri"],
        scope=SP_SCOPES,
        cache_path=None,
        show_dialog=True,
    )


def sp_client(st):
    """Handles the OAuth redirect and returns an authenticated spotipy client,
    or None if the user hasn't connected yet (renders a connect button either way)."""
    oauth = sp_oauth(st.secrets)
    qp = st.query_params
    if "code" in qp and "sp_token" not in st.session_state:
        token = oauth.get_access_token(qp["code"], as_dict=True, check_cache=False)
        st.session_state["sp_token"] = token["access_token"]
        st.query_params.clear()
    if "sp_token" not in st.session_state:
        st.link_button("Connect with Spotify", oauth.get_authorize_url())
        return None
    return spotipy.Spotify(auth=st.session_state["sp_token"])


def genre_counts(sp, artist_ids):
    """Get Several Artists was removed from Development Mode, so this loops
    single-artist lookups instead of batching."""
    ids = list(dict.fromkeys(artist_ids))
    counts = {}
    for artist_id in ids:
        artist = sp.artist(artist_id)
        for g in artist.get('genres', []):
            counts[g] = counts.get(g, 0) + 1
    return counts


def _parse_year(release_date):
    if release_date and release_date[:4].isdigit():
        return int(release_date[:4])
    return None


def build_track_feature_df(tracks):
    """Flatten a list of track objects into a dataframe with the fields
    still available under Spotify's restricted Dev Mode API."""
    rows = []
    for t in tracks:
        rows.append({
            'track_id': t['id'],
            'track_name': t['name'],
            'artists': ", ".join(a['name'] for a in t['artists']),
            'duration_ms': t['duration_ms'],
            'explicit': int(t.get('explicit', False)),
            'release_year': _parse_year(t.get('album', {}).get('release_date', '')),
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=['release_year'])


# ── personality metrics ──────────────────────────────────────────

def nostalgia_index(feat_df):
    """How old the music is relative to now, scaled roughly to 0-100."""
    current_year = datetime.datetime.now().year
    ages = current_year - feat_df['release_year']
    avg_age = ages.mean()
    score = min(100, round(avg_age / 30 * 100))
    oldest = feat_df.loc[feat_df['release_year'].idxmin()]
    return {
        'score': score,
        'avg_age_years': round(avg_age, 1),
        'oldest_track': oldest['track_name'],
        'oldest_year': int(oldest['release_year']),
    }


def genre_pct(genre_count_dict):
    total = sum(genre_count_dict.values())
    if total == 0:
        return {}
    return {g: c / total for g, c in genre_count_dict.items()}


def taste_drift(saved_genre_pct, recent_genre_pct):
    """Total variation distance between two genre distributions, 0-100 scale."""
    all_genres = set(saved_genre_pct) | set(recent_genre_pct)
    diff = sum(abs(saved_genre_pct.get(g, 0) - recent_genre_pct.get(g, 0)) for g in all_genres)
    score = round(min(100, diff / 2 * 100))
    return score


def discovery_vs_comfort(recent_tracks, top_track_ids):
    if not recent_tracks:
        return {'discovery_pct': 0, 'comfort_pct': 0}
    comfort = sum(1 for t in recent_tracks if t['id'] in top_track_ids)
    total = len(recent_tracks)
    comfort_pct = round(comfort / total * 100)
    return {'discovery_pct': 100 - comfort_pct, 'comfort_pct': comfort_pct}


def think_you_like_x(saved_genre_pct, recent_genre_pct):
    if not saved_genre_pct or not recent_genre_pct:
        return None
    saved_top = max(saved_genre_pct, key=saved_genre_pct.get)
    recent_top = max(recent_genre_pct, key=recent_genre_pct.get)
    return {
        'saved_top_genre': saved_top,
        'recent_top_genre': recent_top,
        'mismatch': saved_top != recent_top,
    }


def mood_by_time_of_day(recent_items):
    """Buckets recently-played tracks by hour, using the played_at timestamp."""
    buckets = {'Morning': [], 'Afternoon': [], 'Evening': [], 'Late night': []}
    for item in recent_items:
        played_at = item.get('played_at', '')
        if not played_at:
            continue
        hour = int(played_at[11:13])
        if 5 <= hour < 12:
            bucket = 'Morning'
        elif 12 <= hour < 17:
            bucket = 'Afternoon'
        elif 17 <= hour < 22:
            bucket = 'Evening'
        else:
            bucket = 'Late night'
        t = item['track']
        year = _parse_year(t.get('album', {}).get('release_date', ''))
        buckets[bucket].append({'duration_ms': t['duration_ms'], 'release_year': year})

    rows = []
    for bucket, tracks in buckets.items():
        if not tracks:
            continue
        df = pd.DataFrame(tracks).dropna(subset=['release_year'])
        rows.append({
            'time_of_day': bucket,
            'track_count': len(tracks),
            'avg_duration_min': round(pd.DataFrame(tracks)['duration_ms'].mean() / 60000, 1),
            'avg_release_year': round(df['release_year'].mean()) if len(df) else None,
        })
    return pd.DataFrame(rows)


def music_age_gap(top_feat_df, saved_feat_df, recent_feat_df):
    current_year = datetime.datetime.now().year
    out = {}
    for label, df in [('top', top_feat_df), ('saved', saved_feat_df), ('recent', recent_feat_df)]:
        if len(df):
            out[label] = round((current_year - df['release_year']).mean(), 1)
        else:
            out[label] = None
    return out


def forgotten_library(saved_ids, recent_ids, total_saved):
    if total_saved == 0:
        return {'forgotten_count': 0, 'forgotten_pct': 0}
    forgotten = len(set(saved_ids) - set(recent_ids))
    return {
        'forgotten_count': forgotten,
        'forgotten_pct': round(forgotten / total_saved * 100),
    }


def music_bias(feat_df):
    """Finds whichever feature deviates furthest from the dataset's own average,
    as a single automatically-generated observation."""
    checks = []
    if 'explicit' in feat_df.columns:
        explicit_pct = feat_df['explicit'].mean() * 100
        checks.append(('explicit content', explicit_pct, abs(explicit_pct - 50)))
    if 'release_year' in feat_df.columns:
        recent_pct = (feat_df['release_year'] >= feat_df['release_year'].max() - 5).mean() * 100
        checks.append(('recent-release concentration', recent_pct, abs(recent_pct - 50)))
    if 'duration_ms' in feat_df.columns:
        short_pct = (feat_df['duration_ms'] < 4 * 60_000).mean() * 100
        checks.append(('short tracks (under 4 min)', short_pct, abs(short_pct - 50)))
    if not checks:
        return None
    label, pct, _ = max(checks, key=lambda x: x[2])
    return {'label': label, 'pct': round(pct)}


def obsession_index(top_tracks):
    """How concentrated the top-tracks list is on a small number of artists."""
    artist_counts = {}
    for t in top_tracks:
        for a in t['artists']:
            artist_counts[a['name']] = artist_counts.get(a['name'], 0) + 1
    if not artist_counts:
        return {'score': 0, 'top_artist': None}
    total = sum(artist_counts.values())
    top3 = sorted(artist_counts.values(), reverse=True)[:3]
    score = round(sum(top3) / total * 100)
    top_artist = max(artist_counts, key=artist_counts.get)
    return {'score': score, 'top_artist': top_artist}


def music_dna(scores):
    """Combines the individual metrics into a small labeled profile.
    `scores` is a dict of whichever metrics were computed this session."""
    dna = {}
    if 'nostalgia' in scores:
        dna['Nostalgic'] = scores['nostalgia']
    if 'discovery_pct' in scores:
        dna['Explorer'] = scores['discovery_pct']
    if 'obsession' in scores:
        dna['Repeater'] = scores['obsession']
    if 'taste_drift' in scores:
        dna['Adaptive'] = scores['taste_drift']
    return dna
