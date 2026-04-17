# Customer Segmentation with Unsupervised Learning

## Business Problem
Understanding customer behavior is critical for any retail business. A one-size-fits-all marketing strategy wastes resources on low-value customers while under-investing in high-value ones. This project segments customers of a UK-based online retailer into distinct behavioral groups using unsupervised machine learning, enabling targeted marketing strategies for each segment.

## Key Findings
- **4 distinct customer segments** identified via K-means clustering on RFM (Recency, Frequency, Monetary) features
- **Champions (23.2% of customers) drive 76% of total revenue** — extreme revenue concentration requiring immediate retention focus
- **Top 23% of customers generate 80% of revenue** — confirming a near-Pareto distribution
- **K-means outperformed Hierarchical Clustering and DBSCAN** across all three evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- **At-Risk segment represents £2.67M in recoverable revenue** — highest priority for win-back campaigns

## Customer Segments

| Persona | % Customers | % Revenue | Avg Recency | Avg Frequency | Avg Monetary |
|---------|-------------|-----------|-------------|---------------|--------------|
| Champions | 23.2% | 76.0% | 30 days | 17.9 orders | £9,688 |
| At-Risk | 24.5% | 15.4% | 245 days | 4.7 orders | £1,856 |
| New/Promising | 20.3% | 5.3% | 30 days | 2.8 orders | £775 |
| Lost | 32.0% | 3.3% | 400 days | 1.3 orders | £302 |

## Methodology

```
Raw Data (1,067,371 rows)
        ↓
Data Cleaning (779,495 rows after removing nulls, returns, duplicates)
        ↓
EDA (revenue trends, skewness analysis, Pareto curve)
        ↓
RFM Feature Engineering (5,881 customers)
        ↓
Preprocessing (Winsorize → Log Transform → StandardScaler)
        ↓
K-Means Clustering (k=4, chosen via Elbow + Silhouette analysis)
        ↓
Model Comparison (K-Means vs Hierarchical vs DBSCAN)
        ↓
PCA Visualization + Customer Personas + Business Recommendations
```

## Model Comparison

| Method | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|--------|------------|-------------------|----------------|
| K-Means | 0.377 | 5536.2 | 0.919 |
| Hierarchical | 0.297 | 4245.6 | 1.148 |
| DBSCAN | 0.136 | 2550.6 | 2.319 |

K-Means was selected as the final model based on superior metrics and most balanced, interpretable segments.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda_and_cleaning.ipynb` | Data loading, cleaning pipeline, cleaning log |
| `02_eda.ipynb` | Revenue trends, distributions, Pareto analysis |
| `03_rfm_engineering.ipynb` | RFM computation and validation |
| `04_preprocessing.ipynb` | Winsorization, log transform, standardization |
| `05_clustering.ipynb` | K-Means, elbow/silhouette analysis, cluster profiling |
| `06_alternative_clustering.ipynb` | Hierarchical, DBSCAN, metrics comparison |
| `07_pca.ipynb` | PCA visualization of segments |
| `08_personas.ipynb` | Persona profiles and business recommendations |

## How to Reproduce

1. Clone this repository
2. Download the [Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) from UCI and place it in `data/raw/online_retail_II.xlsx`
3. Install dependencies: `pip install -r requirements.txt`
4. Run notebooks in order (01 → 08)

## Limitations
- **Temporal snapshot:** RFM computed at one point in time. In production, recompute monthly and track segment migration.
- **K-means assumptions:** Assumes spherical, equal-variance clusters.
- **No behavioral features:** Purchase category, browsing data, and returns could improve segmentation.
- **UK dominance:** 90%+ of revenue is UK-based. Geography may influence segments.
- **No ground truth:** Cluster quality evaluated via internal metrics only. A/B testing segment-specific campaigns would be the real validation.

## Tech Stack
Python, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, plotly, yellowbrick, kneed
