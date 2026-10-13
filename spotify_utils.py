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


def build_artist_genre_map(sp, artist_ids):
    """One single-artist lookup per unique artist, reused everywhere else,
    since Get Several Artists (batch) was removed from Development Mode."""
    ids = list(dict.fromkeys(artist_ids))
    genre_map = {}
    for artist_id in ids:
        artist = sp.artist(artist_id)
        genre_map[artist_id] = artist.get('genres', [])
    return genre_map


def genres_for_tracks(tracks, genre_map):
    """Genre counts for a list of tracks, using an already-fetched genre map
    so this makes zero additional API calls."""
    counts = {}
    for t in tracks:
        seen = set()
        for a in t['artists']:
            for g in genre_map.get(a['id'], []):
                if g not in seen:
                    counts[g] = counts.get(g, 0) + 1
                    seen.add(g)
    return counts


def _parse_year(release_date):
    if release_date and release_date[:4].isdigit():
        return int(release_date[:4])
    return None


def _album_image(t):
    images = t.get('album', {}).get('images', [])
    return images[0]['url'] if images else None


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
            'album_image': _album_image(t),
        })
    df = pd.DataFrame(rows, columns=['track_id', 'track_name', 'artists', 'duration_ms', 'explicit', 'release_year', 'album_image'])
    return df.dropna(subset=['release_year'])


# ── personality metrics ──────────────────────────────────────────

NOSTALGIA_RANGES = [
    (25, "Very recent"), (45, "Modern leaning"), (65, "Balanced"),
    (80, "Nostalgic"), (100, "Strongly nostalgic"),
]


def nostalgia_label(score):
    for ceiling, label in NOSTALGIA_RANGES:
        if score <= ceiling:
            return label
    return "Strongly nostalgic"


def nostalgia_index(feat_df):
    """How old the music is relative to now, scaled roughly to 0-100."""
    current_year = datetime.datetime.now().year
    ages = current_year - feat_df['release_year']
    avg_age = ages.mean()
    score = min(100, round(avg_age / 30 * 100))
    oldest = feat_df.loc[feat_df['release_year'].idxmin()]
    newest = feat_df.loc[feat_df['release_year'].idxmax()]
    return {
        'score': score,
        'label': nostalgia_label(score),
        'avg_age_years': round(avg_age, 1),
        'oldest_track': oldest['track_name'],
        'oldest_year': int(oldest['release_year']),
        'oldest_image': oldest.get('album_image'),
        'newest_track': newest['track_name'],
        'newest_year': int(newest['release_year']),
        'newest_image': newest.get('album_image'),
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
    return round(min(100, diff / 2 * 100))


def genre_drift_table(saved_genre_pct, recent_genre_pct, top_n=6):
    """Biggest genre share changes between saved and recent listening,
    for a grouped bar chart."""
    all_genres = set(saved_genre_pct) | set(recent_genre_pct)
    rows = []
    for g in all_genres:
        saved_p = round(saved_genre_pct.get(g, 0) * 100, 1)
        recent_p = round(recent_genre_pct.get(g, 0) * 100, 1)
        rows.append({'genre': g, 'saved': saved_p, 'recent': recent_p, 'delta': recent_p - saved_p})
    df = pd.DataFrame(rows).sort_values('delta', key=abs, ascending=False)
    return df.head(top_n)


def discovery_vs_comfort(recent_tracks, top_track_ids, recent_genre_counts):
    if not recent_tracks:
        return None
    comfort = sum(1 for t in recent_tracks if t['id'] in top_track_ids)
    total = len(recent_tracks)
    comfort_pct = round(comfort / total * 100)

    artist_counts = {}
    for t in recent_tracks:
        for a in t['artists']:
            artist_counts[a['name']] = artist_counts.get(a['name'], 0) + 1
    rare_artists = sum(1 for c in artist_counts.values() if c == 1)

    years = [t.get('album', {}).get('release_date', '')[:4] for t in recent_tracks]
    years = [int(y) for y in years if y.isdigit()]
    year_spread = round(pd.Series(years).std(), 1) if len(years) > 1 else 0

    return {
        'discovery_pct': 100 - comfort_pct,
        'comfort_pct': comfort_pct,
        'unique_artists': len(artist_counts),
        'rare_artists': rare_artists,
        'unique_genres': len(recent_genre_counts),
        'release_year_spread': year_spread,
    }


def think_you_like_x(saved_genre_pct, recent_genre_pct, min_sample=5):
    """Returns None (hide the feature) when there isn't enough genre data
    to say anything meaningful, rather than showing an empty state."""
    if len(saved_genre_pct) < 2 or len(recent_genre_pct) < 2:
        return None
    saved_top = max(saved_genre_pct, key=saved_genre_pct.get)
    recent_top = max(recent_genre_pct, key=recent_genre_pct.get)
    return {
        'saved_top_genre': saved_top,
        'recent_top_genre': recent_top,
        'mismatch': saved_top != recent_top,
    }


def artist_concentration(tracks):
    """Share of tracks accounted for by your top N artists, by track count.
    Not a real play-percentage (Spotify doesn't expose that), just how
    concentrated this particular list is."""
    artist_counts = {}
    for t in tracks:
        for a in t['artists']:
            artist_counts[a['name']] = artist_counts.get(a['name'], 0) + 1
    total = sum(artist_counts.values())
    if total == 0:
        return None
    sorted_counts = sorted(artist_counts.values(), reverse=True)
    curve = {}
    for n in [1, 3, 5, 10, 20]:
        curve[n] = round(sum(sorted_counts[:n]) / total * 100)
    return curve


def mood_by_time_of_day(recent_items, genre_map=None):
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
        buckets[bucket].append(item['track'])

    rows = []
    for bucket, tracks in buckets.items():
        if not tracks:
            continue
        durations = [t['duration_ms'] for t in tracks]
        years = [_parse_year(t.get('album', {}).get('release_date', '')) for t in tracks]
        years = [y for y in years if y]
        row = {
            'time_of_day': bucket,
            'track_count': len(tracks),
            'avg_duration_min': round(sum(durations) / len(durations) / 60000, 1),
            'avg_release_year': round(sum(years) / len(years)) if years else None,
        }
        if genre_map:
            gc = genres_for_tracks(tracks, genre_map)
            row['dominant_genre'] = max(gc, key=gc.get) if gc else None
        rows.append(row)
    return pd.DataFrame(rows)


def music_age_gap(top_feat_df, saved_feat_df, recent_feat_df):
    current_year = datetime.datetime.now().year
    out = {}
    for label, df in [('top', top_feat_df), ('saved', saved_feat_df), ('recent', recent_feat_df)]:
        out[label] = round((current_year - df['release_year']).mean(), 1) if len(df) else None
    return out


def forgotten_library(saved, recent_ids):
    """Saved tracks that haven't shown up in recent listening, with artwork
    for display."""
    recent_id_set = set(recent_ids)
    forgotten = [t for t in saved if t['id'] not in recent_id_set]
    pct = round(len(forgotten) / len(saved) * 100) if saved else 0
    sample = [{
        'track_name': t['name'],
        'artists': ", ".join(a['name'] for a in t['artists']),
        'album_image': _album_image(t),
    } for t in forgotten[:10]]
    return {'forgotten_count': len(forgotten), 'forgotten_pct': pct, 'sample': sample}


def music_bias(feat_df):
    """Finds whichever feature deviates furthest from a 50/50 split, as a
    single automatically-generated observation."""
    checks = []
    if 'explicit' in feat_df.columns:
        explicit_pct = feat_df['explicit'].mean() * 100
        checks.append(('explicit content', explicit_pct, abs(explicit_pct - 50)))
    if 'release_year' in feat_df.columns:
        recent_pct = (feat_df['release_year'] >= feat_df['release_year'].max() - 5).mean() * 100
        checks.append(('releases from the last 5 years', recent_pct, abs(recent_pct - 50)))
    if 'duration_ms' in feat_df.columns:
        short_pct = (feat_df['duration_ms'] < 4 * 60_000).mean() * 100
        checks.append(('tracks under 4 minutes', short_pct, abs(short_pct - 50)))
    if not checks:
        return None
    label, pct, _ = max(checks, key=lambda x: x[2])
    return {'label': label, 'pct': round(pct)}


def obsession_index(top_tracks):
    artist_counts = {}
    for t in top_tracks:
        for a in t['artists']:
            artist_counts[a['name']] = artist_counts.get(a['name'], 0) + 1
    if not artist_counts:
        return {'score': 0, 'top_artist': None}
    total = sum(artist_counts.values())
    top3 = sorted(artist_counts.values(), reverse=True)[:3]
    score = round(sum(top3) / total * 100)
    return {'score': score, 'top_artist': max(artist_counts, key=artist_counts.get)}


def music_dna(scores):
    """Combines whichever metrics were computed this session into a small
    labeled profile — evidence-based dimensions, not a personality quiz."""
    dna = {}
    if 'nostalgia' in scores:
        dna['Nostalgic'] = scores['nostalgia']
    if 'discovery_pct' in scores:
        dna['Explorer'] = scores['discovery_pct']
    if 'obsession' in scores:
        dna['Repeat tendency'] = scores['obsession']
    if 'taste_drift' in scores:
        dna['Adaptive'] = scores['taste_drift']
    if 'genre_diversity' in scores:
        dna['Genre diversity'] = scores['genre_diversity']
    if 'concentration_top5' in scores:
        dna['Artist diversity'] = 100 - scores['concentration_top5']
    return dna


def key_findings(context):
    """3-5 short, data-backed observations, generated from whatever metrics
    were actually computed this session — never hardcoded."""
    findings = []

    age_gap = context.get('age_gap')
    if age_gap and age_gap.get('saved') is not None and age_gap.get('recent') is not None:
        diff = age_gap['saved'] - age_gap['recent']
        if abs(diff) >= 1:
            newer_older = "older" if diff > 0 else "newer"
            findings.append(f"Your saved library is {abs(diff):.1f} years {newer_older} than your recent listening.")

    concentration = context.get('concentration')
    if concentration:
        findings.append(f"Your top 5 artists account for {concentration[5]}% of this list.")

    bias = context.get('bias')
    if bias:
        findings.append(f"{bias['pct']}% of your tracks fall into '{bias['label']}' — your strongest measurable pattern.")

    drift = context.get('taste_drift')
    if drift is not None and drift >= 30:
        findings.append(f"Your recent listening looks quite different from what you've saved (drift score {drift}/100).")

    forgotten = context.get('forgotten')
    if forgotten and forgotten['forgotten_pct'] >= 30:
        findings.append(f"{forgotten['forgotten_pct']}% of your saved tracks haven't shown up in recent listening.")

    return findings[:5]
