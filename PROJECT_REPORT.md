# Traffic Flow Pattern Analysis Using K-Means and DBSCAN — Project Report

## Abstract
This project analyzes traffic-flow patterns with unsupervised machine learning. It performs exploratory
data analysis, data cleaning, feature standardization, K-Means clustering and DBSCAN clustering, and
evaluates cluster quality with Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Score.
DBSCAN additionally flags noise points, which are treated as candidate outliers or incidents.
Results are explored in an interactive Streamlit app.

## Introduction and aim
The title of this project is *Traffic Flow Pattern Analysis Using K-Means and DBSCAN*. Its main aim is to
analyze traffic data and identify different traffic patterns using machine learning clustering algorithms.

Traffic conditions depend on factors such as vehicle count, average speed, road occupancy, location and time.
Instead of analyzing these factors manually, the project uses clustering to find meaningful patterns in the data.
The bundled dataset contains vehicle count (`traffic_volume`), average speed (`speed_kmh`), road occupancy
(`occupancy`), location (`road_segment`) and time (`timestamp`). Weather is not part of the current dataset;
it could be added as an extra feature in future work.

## Scope
The project is **clustering / pattern discovery**, not flow forecasting. It groups observations into
traffic states; it does not predict future speed or volume.

## Data
Input features: `speed_kmh`, `traffic_volume`, `occupancy`. Optional: `timestamp`, `road_segment`.
The bundled `data/traffic_sample.csv` has 24 hand-made rows (3 states x 5 rows, 2 incident-like rows,
7 deliberately invalid rows). It is a **demonstration dataset**, not real sensor data.

## Method
1. **Cleaning** — rows with missing/non-numeric values or impossible values (negative speed or volume,
   occupancy outside 0–1) are removed and exported with a reason. Only the selected features are checked.
2. **EDA** — preview, descriptive statistics, missing values and seven core graphs: (1) traffic level
   distribution, (2) vehicle count distribution, (3) average speed distribution, (4) traffic by location,
   (5) vehicle count vs average speed, (6) vehicle count vs road occupancy, (7) correlation heatmap. Box plots
   and (if timestamps exist) feature-over-time plots are also provided.
3. **Standardization** — `StandardScaler`, so no feature dominates through its units.
4. **K-Means** — `random_state=42`, `n_init=10`; K chosen with the elbow and silhouette-by-K plots.
5. **DBSCAN** — `eps` and `min_samples` set from the sidebar; label `-1` is noise.
6. **Visualization** — 2-component PCA (about 99.9% of variance on the sample data).
7. **Evaluation** — internal metrics (noise excluded for DBSCAN) and, when a reference column exists,
   Adjusted Rand Index and Normalized Mutual Information.

## Results on the sample data (17 valid rows, K = 3, eps = 0.65, min_samples = 4)

| Metric | K-Means | DBSCAN |
|---|---|---|
| Silhouette (higher is better) | 0.656 | 0.863* |
| Davies-Bouldin (lower is better) | 0.573 | 0.162* |
| Calinski-Harabasz (higher is better) | 24.99 | 459.18* |
| ARI vs reference states | 0.792 | 1.000 |
| NMI vs reference states | 0.835 | 1.000 |

\*DBSCAN's internal scores exclude its 2 noise points, so they are flattering and not directly comparable
with K-Means, which must place every point in a cluster.

**K-Means (K = 3)** recovers the three main states (cluster means: speed 73.8 / volume 23.8 / occupancy 0.11
for free-flow, 41.2 / 54.0 / 0.29 for moderate, 31.3 / 107.5 / 0.75 for congested). The congested cluster's mean
speed is pulled down to 31 km/h by one incident-like row (speed 5), and one incident-like row is absorbed into
the moderate cluster.

**DBSCAN** finds the same three dense groups and labels the two incident-like rows as noise
(speed 5 / volume 25 / occupancy 0.05 and speed 90 / volume 115 / occupancy 0.85), matching the reference
states exactly. That is the useful property of density-based clustering for incident detection.

**Choosing K:** silhouette peaks at K = 5 (0.762), where K-Means gives the three states plus the two
incident rows their own clusters. This shows that K = 3 is a modelling choice and not a data fact.

## Interpretation
High speed with low volume and low occupancy indicates free-flow traffic. Medium values indicate moderate
traffic. Low speed with high volume and high occupancy indicates congestion. DBSCAN noise marks observations
that do not belong to a sufficiently dense group; here they are physically unusual combinations
(very low speed with almost no occupancy, or high speed with very high occupancy) and merit inspection as
possible sensor faults or incidents.

## Limitations
- The sample is tiny and hand-made, with well-separated groups by construction. The perfect DBSCAN agreement
  demonstrates the method, not real-world performance.
- Results are sensitive to `eps`, `min_samples` and K; they should be tuned per dataset.
- Three features and no temporal context. Time-of-day, road segment and neighbouring segments are not
  modelled.
- Clusters are labelled by interpreting their feature means, not by ground truth.

## Future work
Run on real detector data (for example a public loop-detector dataset), add time-of-day features, compare
with Gaussian mixtures or hierarchical clustering, and extend to short-term flow forecasting.

## Conclusion
The application provides an end-to-end, tested clustering workflow for traffic data. On the demonstration
data, DBSCAN separates normal traffic states from anomalous observations, while K-Means gives a cleaner
partition of the main states but is sensitive to outliers and to the choice of K.


## Interactive cleaning update
The Streamlit interface now provides selectable cleaning options for uploaded CSV files: keep missing values, remove rows with missing values, impute numeric values with mean or median, fill values with mode or a constant, remove duplicate rows, trim text whitespace, and remove impossible values in known traffic fields. The Preprocessing and Results tabs preview and export the cleaned dataset and rejected rows. Rows with missing selected features are excluded from clustering if the chosen cleaning method leaves them incomplete.
