# Traffic Flow Pattern Analysis Using K-Means and DBSCAN

Streamlit project for unsupervised analysis of traffic-flow data: configurable cleaning, EDA, correlation matrix, K-Means, DBSCAN, outlier detection, PCA visualization, evaluation, cluster profiling and downloadable results.

## Run
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501

## Test
```powershell
python test_pipeline.py
```

## Project layout
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (models are fitted once and reused by every tab) |
| `pipeline.py` | Cleaning, clustering, metrics (no Streamlit, so it is testable) |
| `test_pipeline.py` | Tests for cleaning, metrics and edge cases |
| `data/traffic_sample.csv` | Small hand-made demo dataset |
| `PROJECT_REPORT.md` | Method, results and limitations |

## Flow
CSV → Cleaning → EDA → Correlation Matrix → Scaling → K-Means/DBSCAN → Evaluation → Visualization → Download

## Uploading your own CSV
Upload your own CSV in the sidebar. The app suggests numeric/numeric-like columns for clustering; select at least two.
Known traffic fields (`speed_kmh`, `traffic_volume`, `occupancy`) are selected by default when present. Optional: `timestamp` (enables time plots) and `illustrative_state` (a reference label used only to compare clusters, never for fitting).

## Cleaning options
In the sidebar you can choose to leave missing values unchanged, remove rows with missing values, fill numeric columns with mean or median, fill missing values with mode, or fill with a constant. You can also choose whether to remove duplicates, trim extra spaces, and remove impossible values in known traffic fields. The Preprocessing tab previews the cleaned data and lets you download cleaned and rejected rows. If missing values are left unchanged, rows with incomplete selected clustering features are skipped by the models but remain in the cleaned CSV.

## EDA graphs (EDA tab)
1. Traffic level distribution  2. Vehicle count distribution  3. Average speed distribution
4. Traffic by location  5. Vehicle count vs average speed  6. Vehicle count vs road occupancy
7. Correlation heatmap
Graphs 1 and 4 need the optional `illustrative_state` and `road_segment` columns respectively.
