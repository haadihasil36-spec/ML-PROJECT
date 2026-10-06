from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Flow Pattern Analysis",
    page_icon="🚦",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🚦 Traffic Flow Pattern Analysis Using K-Means and DBSCAN")

st.markdown(
    "Explore traffic observations, discover recurring traffic patterns, "
    "and compare two unsupervised machine-learning algorithms."
)


# ============================================================
# DATA PATH
# ============================================================

DEFAULT_PATH = (
    Path(__file__).parent
    / "data"
    / "traffic_sample.csv"
)


# ============================================================
# LOAD SAMPLE DATA
# ============================================================

@st.cache_data
def load_sample():
    return pd.read_csv(DEFAULT_PATH)


# ============================================================
# GENERATE DEMO DATA
# ============================================================

def make_demo_data(n=900, seed=42):
    """
    Create a reproducible illustrative traffic dataset.
    """

    rng = np.random.default_rng(seed)

    groups = [
        (
            "Free-flow",
            int(n * 0.38),
            72,
            8,
            24,
            7,
            0.12,
            0.04,
        ),
        (
            "Moderate",
            int(n * 0.34),
            48,
            7,
            58,
            12,
            0.31,
            0.07,
        ),
        (
            "Congested",
            int(n * 0.22),
            19,
            6,
            92,
            15,
            0.69,
            0.10,
        ),
        (
            "Incident-like",
            n
            - int(n * 0.38)
            - int(n * 0.34)
            - int(n * 0.22),
            12,
            10,
            28,
            13,
            0.48,
            0.17,
        ),
    ]

    rows = []

    timestamp = pd.date_range(
        "2026-01-01 06:00",
        periods=n,
        freq="5min",
    )

    idx = 0

    for (
        name,
        count,
        speed_mean,
        speed_std,
        volume_mean,
        volume_std,
        occupancy_mean,
        occupancy_std,
    ) in groups:

        for _ in range(count):

            rows.append(
                {
                    "timestamp": timestamp[idx],
                    "speed_kmh": max(
                        0,
                        rng.normal(
                            speed_mean,
                            speed_std,
                        ),
                    ),
                    "traffic_volume": max(
                        0,
                        rng.normal(
                            volume_mean,
                            volume_std,
                        ),
                    ),
                    "occupancy": float(
                        np.clip(
                            rng.normal(
                                occupancy_mean,
                                occupancy_std,
                            ),
                            0,
                            1,
                        )
                    ),
                    "road_segment": rng.choice(
                        ["A1", "A2", "B1", "B2"]
                    ),
                    "illustrative_state": name,
                }
            )

            idx += 1

    df = (
        pd.DataFrame(rows)
        .sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

def clean_numeric(df, features):

    x = df[features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = x.notna().all(axis=1)

    return (
        x.loc[valid].copy(),
        valid,
    )


# ============================================================
# SIDEBAR - DATA
# ============================================================

with st.sidebar:

    st.header("1. Data")

    uploaded = st.file_uploader(
        "Upload traffic CSV",
        type=["csv"],
    )

    if uploaded is not None:

        try:

            raw = pd.read_csv(uploaded)

            data_source = "Uploaded CSV"

        except Exception as e:

            st.error(
                f"Could not read CSV: {e}"
            )

            st.stop()

    else:

        try:

            raw = load_sample()

            data_source = "Included sample dataset"

        except Exception:

            raw = make_demo_data()

            data_source = "Generated demo dataset"

    st.caption(
        f"Source: {data_source}"
    )


    # ========================================================
    # FEATURES
    # ========================================================

    st.header("2. Features")

    numeric_cols = (
        raw
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    preferred = [
        c
        for c in [
            "speed_kmh",
            "traffic_volume",
            "occupancy",
        ]
        if c in numeric_cols
    ]

    if len(preferred) >= 2:

        default_features = preferred

    else:

        default_features = numeric_cols[
            :min(3, len(numeric_cols))
        ]

    features = st.multiselect(
        "Select at least 2 numeric features",
        options=numeric_cols,
        default=default_features,
    )


    # ========================================================
    # ALGORITHM
    # ========================================================

    algorithm = st.radio(
        "Clustering algorithm",
        [
            "Compare both",
            "K-Means",
            "DBSCAN",
        ],
    )


    # ========================================================
    # SCALING
    # ========================================================

    scale_data = st.checkbox(
        "Standardize features (recommended)",
        value=True,
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

if len(features) < 2:

    st.warning(
        "Select at least two numeric columns in the sidebar."
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

X_df, valid_mask = clean_numeric(
    raw,
    features,
)

if len(X_df) < 5:

    st.error(
        "At least 5 rows with valid numeric values are required."
    )

    st.stop()


dropped = (
    len(raw)
    - len(X_df)
)


# ============================================================
# SCALE DATA
# ============================================================

scaler = StandardScaler()

if scale_data:

    X = scaler.fit_transform(
        X_df
    )

else:

    X = X_df.to_numpy()


# ============================================================
# DATA SUMMARY
# ============================================================

st.caption(
    f"Rows: {len(raw):,} · "
    f"Rows used: {len(X_df):,} · "
    f"Invalid rows skipped: {dropped:,}"
)


# ============================================================
# PREVIEW DATA
# ============================================================

with st.expander("Preview data"):

    st.dataframe(
        raw.head(20),
        use_container_width=True,
    )

    st.download_button(
        "Download input data as CSV",
        raw.to_csv(index=False).encode("utf-8"),
        "traffic_input.csv",
        "text/csv",
    )


# ============================================================
# METRICS
# ============================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Observations used",
    f"{len(X_df):,}",
)

c2.metric(
    "Selected features",
    len(features),
)

c3.metric(
    "Features scaled",
    "Yes" if scale_data else "No",
)


# ============================================================
# TRAFFIC FEATURE RELATIONSHIPS
# ============================================================

st.subheader(
    "Traffic feature relationships"
)


# IMPORTANT FIX:
# X_df contains only numeric selected features.
# Therefore timestamp and road_segment must be added
# separately before using them as hover_data/color.

relationship_df = X_df.copy()


# Add timestamp if available

if "timestamp" in raw.columns:

    relationship_df["timestamp"] = (
        raw.loc[X_df.index, "timestamp"]
        .astype(str)
        .values
    )


# Add road segment if available

if "road_segment" in raw.columns:

    relationship_df["road_segment"] = (
        raw.loc[X_df.index, "road_segment"]
        .astype(str)
        .values
    )


# ============================================================
# CREATE RELATIONSHIP SCATTER PLOT
# ============================================================

color_column = (
    "road_segment"
    if "road_segment" in relationship_df.columns
    else None
)


hover_columns = [
    column
    for column in [
        "timestamp",
        "road_segment",
    ]
    if column in relationship_df.columns
]


fig = px.scatter(
    relationship_df,
    x=features[0],
    y=features[1],
    color=color_column,
    hover_data=hover_columns,
    title=f"{features[0]} vs {features[1]}",
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# MODEL SETTINGS
# ============================================================

st.sidebar.header(
    "3. Model settings"
)


# K-MEANS

max_k = min(
    10,
    max(
        2,
        len(X_df) - 1,
    ),
)


k = st.sidebar.slider(
    "K-Means clusters (K)",
    2,
    max_k,
    3,
)


# DBSCAN EPS

eps = st.sidebar.slider(
    "DBSCAN eps",
    0.1,
    3.0,
    0.65,
    0.05,
    help=(
        "Distance threshold in the feature space. "
        "If standardized, this is in standard-deviation units."
    ),
)


# DBSCAN MIN SAMPLES

min_samples = st.sidebar.slider(
    "DBSCAN min_samples",
    2,
    30,
    8,
)


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate(labels, data):

    labels = np.asarray(labels)

    non_noise = labels != -1

    unique = np.unique(
        labels[non_noise]
    )

    n_clusters = len(unique)

    noise_pct = float(
        np.mean(labels == -1)
        * 100
    )

    result = {
        "Clusters (excluding noise)": n_clusters,
        "Noise points (%)": round(
            noise_pct,
            2,
        ),
        "Silhouette": np.nan,
        "Davies–Bouldin": np.nan,
        "Calinski–Harabasz": np.nan,
    }


    # Metrics require at least two clusters.

    mask = non_noise

    if (
        n_clusters >= 2
        and mask.sum() > n_clusters
    ):

        try:

            result["Silhouette"] = round(
                float(
                    silhouette_score(
                        data[mask],
                        labels[mask],
                    )
                ),
                4,
            )

            result["Davies–Bouldin"] = round(
                float(
                    davies_bouldin_score(
                        data[mask],
                        labels[mask],
                    )
                ),
                4,
            )

            result[
                "Calinski–Harabasz"
            ] = round(
                float(
                    calinski_harabasz_score(
                        data[mask],
                        labels[mask],
                    )
                ),
                2,
            )

        except Exception:

            pass


    return result


# ============================================================
# RUN MODELS
# ============================================================

results = {}

models = {}


# ============================================================
# K-MEANS
# ============================================================

if algorithm in [
    "Compare both",
    "K-Means",
]:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    results["K-Means"] = (
        km.fit_predict(X)
    )

    models["K-Means"] = km


# ============================================================
# DBSCAN
# ============================================================

if algorithm in [
    "Compare both",
    "DBSCAN",
]:

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    results["DBSCAN"] = (
        db.fit_predict(X)
    )

    models["DBSCAN"] = db


# ============================================================
# CLUSTERING RESULTS
# ============================================================

st.subheader(
    "Clustering results"
)


metric_rows = []


for name, labels in results.items():

    metrics = evaluate(
        labels,
        X,
    )

    metrics["Algorithm"] = name

    metric_rows.append(
        metrics
    )


metrics_df = (
    pd.DataFrame(
        metric_rows
    )
    .set_index("Algorithm")
)


st.dataframe(
    metrics_df,
    use_container_width=True,
)


# ============================================================
# METRIC EXPLANATION
# ============================================================

if len(results) == 2:

    st.info(
        "Metric guide: higher Silhouette and "
        "Calinski–Harabasz values, and lower "
        "Davies–Bouldin values, often indicate "
        "better-separated clusters. DBSCAN marks "
        "outliers as -1; its scores exclude those "
        "noise points. Metrics are not directly "
        "definitive when cluster shapes differ."
    )


# ============================================================
# ADD CLUSTER LABELS TO OUTPUT DATA
# ============================================================

out = raw.loc[
    X_df.index
].copy()


for name, labels in results.items():

    column_name = (
        name
        .lower()
        .replace("-", "")
        .replace(" ", "_")
        + "_cluster"
    )

    out[column_name] = labels


# ============================================================
# CLUSTER VISUALIZATIONS
# ============================================================

st.subheader(
    "Cluster visualizations"
)


plot_features = features[:2]


for name, labels in results.items():

    plot_df = X_df.copy()

    plot_df["Cluster"] = (
        pd.Series(
            labels,
            index=plot_df.index,
        )
        .astype(str)
    )


    # Convert DBSCAN noise label

    plot_df["Cluster"] = (
        plot_df["Cluster"]
        .replace(
            {
                "-1": "Noise (-1)"
            }
        )
    )


    # Hover only with columns that actually exist

    cluster_hover = [
        column
        for column in features
        if column in plot_df.columns
    ]


    fig = px.scatter(
        plot_df,
        x=plot_features[0],
        y=plot_features[1],
        color="Cluster",
        title=(
            f"{name}: discovered traffic groups"
        ),
        hover_data=cluster_hover,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# CLUSTER PROFILE
# ============================================================

st.subheader(
    "Cluster profile"
)


selected_name = st.selectbox(
    "Profile algorithm",
    list(results.keys()),
)


profile_df = X_df.copy()

profile_df["cluster"] = (
    results[selected_name]
)


profile = (
    profile_df
    .groupby("cluster")[features]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
        ]
    )
    .round(3)
)


st.dataframe(
    profile,
    use_container_width=True,
)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.subheader(
    "Download results"
)


csv = out.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "⬇️ Download observations with cluster labels",
    csv,
    "traffic_cluster_results.csv",
    "text/csv",
)


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown("---")

st.markdown(
    """
### How to interpret

**K-Means** assigns every observation to one of K
centroid-based groups. It generally works well when
groups are relatively compact and separated.

**DBSCAN** groups dense regions of observations and
can identify isolated observations as noise. It does
not require choosing the number of clusters in advance.

The results depend on:

- Selected traffic features
- Feature scaling
- K-Means K value
- DBSCAN eps value
- DBSCAN min_samples value
"""
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Educational project demo. The included sample "
    "data is synthetic/illustrative, not live traffic data."
)