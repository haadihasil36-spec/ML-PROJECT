import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import (
    FEATURES,
    REFERENCE_COL,
    external_metrics,
    internal_metrics,
    k_sweep,
    run_models,
)

st.set_page_config(page_title="Traffic Flow Analysis", page_icon="🚦", layout="wide")
st.title("🚦 Traffic Flow Pattern Analysis Using K-Means and DBSCAN")
st.caption("Upload a CSV, choose cleaning options, explore patterns, and run clustering.")


@st.cache_data
def default_data():
    return pd.read_csv("data/traffic_sample.csv")


def numeric_candidates(df: pd.DataFrame) -> list[str]:
    """Include numeric columns and text columns that mostly contain numbers."""
    out = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        nonempty = df[col].notna().sum()
        if pd.api.types.is_numeric_dtype(df[col]) or (nonempty and converted.notna().sum() / nonempty >= 0.7):
            out.append(col)
    return out


def clean_uploaded_data(
    df: pd.DataFrame,
    features: list[str],
    missing_strategy: str,
    remove_duplicates: bool,
    trim_spaces: bool,
    remove_impossible: bool,
    constant_text: str,
):
    """Apply user-selected cleaning and return (cleaned, rejected, notes)."""
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    notes = []

    # Trim whitespace in text fields, if requested.
    if trim_spaces:
        for col in work.select_dtypes(include=["object", "string"]).columns:
            work[col] = work[col].map(lambda value: value.strip() if isinstance(value, str) else value)
        notes.append("Trimmed extra spaces from text cells")

    # Make selected model features numeric. Unparseable strings become missing values.
    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    notes.append("Converted selected clustering features to numeric")

    rejected_parts = []
    # Detect impossible values only for known traffic fields, not arbitrary numeric columns.
    if remove_impossible:
        impossible = pd.Series(False, index=work.index)
        if "speed_kmh" in features:
            impossible |= work["speed_kmh"] < 0
        if "traffic_volume" in features:
            impossible |= work["traffic_volume"] < 0
        if "occupancy" in features:
            impossible |= (work["occupancy"] < 0) | (work["occupancy"] > 1)
        if impossible.any():
            rejected_parts.append(work.loc[impossible].copy().assign(reason="impossible traffic value"))
            work = work.loc[~impossible].copy()
            notes.append(f"Removed {int(impossible.sum())} row(s) with impossible traffic values")

    if remove_duplicates:
        before = len(work)
        duplicate_mask = work.duplicated(keep="first")
        if duplicate_mask.any():
            rejected_parts.append(work.loc[duplicate_mask].copy().assign(reason="duplicate row"))
            work = work.loc[~duplicate_mask].copy()
        notes.append(f"Removed {before - len(work)} duplicate row(s)")

    if missing_strategy == "Remove rows with missing values":
        missing_mask = work.isna().any(axis=1)
        if missing_mask.any():
            rejected_parts.append(work.loc[missing_mask].copy().assign(reason="missing value"))
            work = work.loc[~missing_mask].copy()
        notes.append(f"Removed {int(missing_mask.sum())} row(s) containing missing values")
    elif missing_strategy == "Fill numeric columns with mean":
        for col in work.select_dtypes(include=np.number).columns:
            if work[col].isna().any() and work[col].notna().any():
                work[col] = work[col].fillna(work[col].mean())
        notes.append("Filled missing numeric values with the mean")
    elif missing_strategy == "Fill numeric columns with median":
        for col in work.select_dtypes(include=np.number).columns:
            if work[col].isna().any() and work[col].notna().any():
                work[col] = work[col].fillna(work[col].median())
        notes.append("Filled missing numeric values with the median")
    elif missing_strategy == "Fill missing values with mode":
        for col in work.columns:
            modes = work[col].mode(dropna=True)
            if work[col].isna().any() and not modes.empty:
                work[col] = work[col].fillna(modes.iloc[0])
        notes.append("Filled missing values with each column's mode")
    elif missing_strategy == "Fill missing values with a constant":
        try:
            fill_value = float(constant_text)
        except (TypeError, ValueError):
            fill_value = constant_text
        work = work.fillna(fill_value)
        notes.append(f"Filled missing values with {fill_value!r}")
    else:
        notes.append("Left missing values unchanged")

    if rejected_parts:
        rejected = pd.concat(rejected_parts, axis=0).sort_index()
        rejected = rejected.loc[~rejected.index.duplicated(keep="first")]
    else:
        rejected = pd.DataFrame(columns=list(work.columns) + ["reason"])
    return work.reset_index(drop=True), rejected.reset_index(drop=True), notes


# Sidebar: upload, feature selection, and cleaning choices.
st.sidebar.header("📂 Dataset")
uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
try:
    raw = pd.read_csv(uploaded) if uploaded is not None else default_data()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

raw.columns = [str(c).strip() for c in raw.columns]
if raw.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

candidates = numeric_candidates(raw)
if len(candidates) < 2:
    st.error("This dataset needs at least two numeric or numeric-like columns for clustering.")
    st.stop()
default_features = [c for c in FEATURES if c in candidates]
if len(default_features) < 2:
    default_features = candidates[:min(3, len(candidates))]
features = st.sidebar.multiselect("Numeric features for clustering", candidates, default=default_features)
if len(features) < 2:
    st.warning("Select at least two numeric features in the sidebar.")
    st.stop()

st.sidebar.divider()
st.sidebar.header("🧹 Cleaning options")
missing_strategy = st.sidebar.selectbox(
    "Handle missing values",
    [
        "Leave missing values unchanged",
        "Remove rows with missing values",
        "Fill numeric columns with mean",
        "Fill numeric columns with median",
        "Fill missing values with mode",
        "Fill missing values with a constant",
    ],
    index=0,
)
constant_text = "0"
if missing_strategy == "Fill missing values with a constant":
    constant_text = st.sidebar.text_input("Constant value", value="0")
remove_duplicates = st.sidebar.checkbox("Remove duplicate rows", value=True)
trim_spaces = st.sidebar.checkbox("Trim extra spaces in text", value=True)
remove_impossible = st.sidebar.checkbox("Remove impossible traffic values", value=True,
    help="Checks speed and volume for negative values and occupancy for values outside 0–1, when those features are selected.")

clean, rejected, cleaning_notes = clean_uploaded_data(
    raw, features, missing_strategy, remove_duplicates, trim_spaces, remove_impossible, constant_text
)

# Models require complete selected feature rows, even when the user chose to keep missing values in the cleaned export.
model_data = clean.copy()
for col in features:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna(subset=features).reset_index(drop=True)

st.subheader("Dataset summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Original rows", f"{len(raw):,}")
c2.metric("Rows after cleaning", f"{len(clean):,}")
c3.metric("Rows usable for clustering", f"{len(model_data):,}")
c4.metric("Columns", f"{len(clean.columns):,}")
st.caption("The app uses the selected cleaning options below. If you leave missing values unchanged, incomplete feature rows are retained in the cleaned CSV but skipped for clustering.")

# Tabs
tabs = st.tabs(["🔎 EDA", "🧹 Preprocessing", "🤖 Clustering", "📈 Evaluation", "📥 Results"])
has_time = "timestamp" in clean.columns and clean["timestamp"].notna().any()
run = None
result = pd.DataFrame()
model_error = None
if len(model_data) >= 3:
    k_max = min(8, len(model_data) - 1)
    k_default = min(3, k_max)
    with st.sidebar:
        st.divider()
        st.header("🤖 Model settings")
        k = st.slider("K-Means: K", 2, k_max, k_default)
        eps = st.slider("DBSCAN: eps", 0.10, 2.00, 0.65, 0.05)
        min_samples = st.slider("DBSCAN: min_samples", 2, min(15, len(model_data)), min(4, len(model_data)))
    try:
        run = run_models(model_data, features, k, eps, min_samples)
        result = run.result
    except Exception as exc:
        model_error = str(exc)
else:
    st.sidebar.info("At least 3 complete rows are needed for clustering.")

# EDA
with tabs[0]:
    st.header("Exploratory Data Analysis")
    st.write("### Dataset preview")
    st.dataframe(raw.head(15), use_container_width=True)
    if not clean.empty:
        st.write("### Descriptive statistics")
        st.dataframe(clean[features].describe().T, use_container_width=True)
        st.write("### Missing values")
        miss = clean.isna().sum()
        st.dataframe(pd.DataFrame({"Missing Count": miss, "Percentage": (miss / max(len(clean), 1) * 100).round(2)}), use_container_width=True)
        st.write("### Numeric distributions")
        plot_col = st.selectbox("Choose a numeric feature", features, key="eda_hist_feature")
        vals = pd.to_numeric(clean[plot_col], errors="coerce").dropna()
        st.plotly_chart(px.histogram(x=vals, nbins=20, title=f"Distribution of {plot_col}", labels={"x": plot_col}), use_container_width=True)
        numeric_clean = clean.select_dtypes(include=np.number)
        if len(numeric_clean.columns) >= 2:
            st.write("### Correlation heatmap")
            corr = numeric_clean.corr(numeric_only=True).round(2)
            st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlations", color_continuous_scale="RdBu_r"), use_container_width=True)
        if {"traffic_volume", "speed_kmh"}.issubset(clean.columns):
            plot_data = clean.copy()
            plot_data["traffic_volume"] = pd.to_numeric(plot_data["traffic_volume"], errors="coerce")
            plot_data["speed_kmh"] = pd.to_numeric(plot_data["speed_kmh"], errors="coerce")
            st.plotly_chart(px.scatter(plot_data, x="traffic_volume", y="speed_kmh", color=REFERENCE_COL if REFERENCE_COL in plot_data.columns else None,
                                       title="Traffic Volume vs Speed"), use_container_width=True)
    else:
        st.warning("No rows remain after the selected cleaning operations.")

# Preprocessing / selectable cleaning options
with tabs[1]:
    st.header("🧹 Data Preprocessing and Cleaning")
    st.write("Change cleaning options in the left sidebar. The cleaned preview updates automatically.")
    st.subheader("Before cleaning")
    a, b, c = st.columns(3)
    a.metric("Rows", len(raw))
    b.metric("Missing cells", int(raw.isna().sum().sum()))
    c.metric("Duplicate rows", int(raw.duplicated().sum()))
    st.dataframe(raw.head(20), use_container_width=True)
    st.subheader("After cleaning")
    a, b, c = st.columns(3)
    a.metric("Rows", len(clean), delta=len(clean) - len(raw))
    b.metric("Missing cells", int(clean.isna().sum().sum()))
    c.metric("Duplicate rows", int(clean.duplicated().sum()))
    st.dataframe(clean.head(50), use_container_width=True)
    with st.expander("Cleaning operations applied", expanded=True):
        for note in cleaning_notes:
            st.write("• " + note)
    st.download_button("⬇️ Download cleaned dataset", clean.to_csv(index=False).encode("utf-8"),
                       "cleaned_dataset.csv", "text/csv", type="primary", key="clean_download_preprocessing")
    st.subheader("Removed / rejected rows")
    if len(rejected):
        st.dataframe(rejected, use_container_width=True)
        st.download_button("⬇️ Download removed rows", rejected.to_csv(index=False).encode("utf-8"),
                           "removed_rows.csv", "text/csv", key="bad_download_preprocessing")
    else:
        st.success("No rows were rejected by the selected rules.")
    if len(model_data) != len(clean):
        st.info(f"{len(clean) - len(model_data)} row(s) with missing or non-numeric selected features remain in the cleaned file but are skipped by clustering.")

# Clustering
with tabs[2]:
    st.header("🤖 K-Means and DBSCAN")
    if model_error:
        st.error(f"Clustering could not run: {model_error}")
    elif run is None:
        st.warning("Not enough complete rows to cluster. Change cleaning settings or upload more data.")
    else:
        plotdf = pd.DataFrame({"PC1": run.pcs[:, 0], "PC2": run.pcs[:, 1],
                               "K-Means": run.km_labels.astype(str), "DBSCAN": run.db_labels.astype(str)})
        st.caption(f"PCA explains {run.explained_variance.sum() * 100:.1f}% of the variance in the two plotted components.")
        a, b = st.columns(2)
        a.plotly_chart(px.scatter(plotdf, x="PC1", y="PC2", color="K-Means", title="K-Means Clusters (PCA)"), use_container_width=True)
        b.plotly_chart(px.scatter(plotdf, x="PC1", y="PC2", color="DBSCAN", title="DBSCAN Clusters (PCA)"), use_container_width=True)
        if has_time and "timestamp" in result.columns and features:
            tdf = result.copy()
            tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
            tdf = tdf.dropna(subset=["timestamp"]).sort_values("timestamp")
            if len(tdf):
                tdf["KMeans_Cluster"] = tdf["KMeans_Cluster"].astype(str)
                st.plotly_chart(px.scatter(tdf, x="timestamp", y=features[0], color="KMeans_Cluster",
                                           title=f"{features[0]} over time, coloured by K-Means cluster"), use_container_width=True)
        st.write("### Clustered data")
        st.dataframe(result, use_container_width=True)
        st.write("### K-Means cluster profiles")
        st.dataframe(result.groupby("KMeans_Cluster")[features].agg(["count", "mean", "median"]), use_container_width=True)
        st.metric("DBSCAN noise / outliers", int(result["DBSCAN_Noise"].sum()))

# Evaluation
with tabs[3]:
    st.header("📈 Evaluation Metrics")
    if run is None:
        st.warning("Run clustering on at least three complete rows to see evaluation metrics.")
    else:
        km_m = internal_metrics(run.X, run.km_labels)
        db_m = internal_metrics(run.X, run.db_labels)
        st.dataframe(pd.DataFrame({"Metric": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score"],
                                   "K-Means": list(km_m), "DBSCAN": list(db_m)}).round(4), use_container_width=True)
        st.info("Silhouette: higher is better. Davies-Bouldin: lower is better. Calinski-Harabasz: higher is better. DBSCAN scores exclude noise points and can be unavailable when fewer than two clusters are found.")
        counts = pd.DataFrame({"K-Means": pd.Series(run.km_labels).value_counts().sort_index(),
                               "DBSCAN": pd.Series(run.db_labels).value_counts().sort_index()}).fillna(0)
        st.bar_chart(counts)
        sweep = k_sweep(run.X)
        if len(sweep):
            a, b = st.columns(2)
            a.plotly_chart(px.line(sweep, x="K", y="Inertia", markers=True, title="Elbow (inertia)"), use_container_width=True)
            b.plotly_chart(px.line(sweep, x="K", y="Silhouette", markers=True, title="Silhouette by K"), use_container_width=True)
        if REFERENCE_COL in result.columns:
            st.write("### Agreement with reference labels")
            st.caption(f"`{REFERENCE_COL}` is used only for evaluation, never to fit the clustering models.")
            ari_k, nmi_k = external_metrics(result[REFERENCE_COL], run.km_labels)
            ari_d, nmi_d = external_metrics(result[REFERENCE_COL], run.db_labels)
            st.dataframe(pd.DataFrame({"Metric": ["Adjusted Rand Index", "Normalized Mutual Info"],
                                       "K-Means": [ari_k, nmi_k], "DBSCAN": [ari_d, nmi_d]}).round(4), use_container_width=True)
            st.dataframe(pd.crosstab(result[REFERENCE_COL], result["KMeans_Cluster"]), use_container_width=True)

# Results downloads
with tabs[4]:
    st.header("📥 Download Results")
    st.download_button("⬇️ Download cleaned dataset", clean.to_csv(index=False).encode("utf-8"),
                       "cleaned_dataset.csv", "text/csv", key="clean_download_results")
    if run is not None:
        st.download_button("⬇️ Download clustering results", result.to_csv(index=False).encode("utf-8"),
                           "traffic_clustering_results.csv", "text/csv", key="cluster_download_results")
    else:
        st.info("Clustering results will be available after there are enough complete numeric rows.")
    if len(rejected):
        st.download_button("⬇️ Download removed rows", rejected.to_csv(index=False).encode("utf-8"),
                           "removed_rows.csv", "text/csv", key="bad_download_results")
    st.dataframe(result if run is not None else clean, use_container_width=True)

st.divider()
st.caption("Traffic Flow Pattern Analysis • Unsupervised Machine Learning")
