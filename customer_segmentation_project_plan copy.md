# Customer Segmentation with Unsupervised Learning — Senior DS Project Plan

> **Goal:** Build an end-to-end customer segmentation pipeline that demonstrates unsupervised learning mastery, business acumen, and production-quality thinking. This plan is written as if a senior DS is mentoring you through every decision.

---

## Phase 0: Project Setup & Mindset (Day 1)

Before touching data, set up like a professional.

### Repository Structure

```
customer-segmentation/
├── data/
│   ├── raw/                  # Original UCI dataset (never modify)
│   └── processed/            # Cleaned outputs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_rfm_engineering.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_evaluation_comparison.ipynb
│   └── 05_personas_and_recommendations.ipynb
├── src/
│   ├── data_cleaning.py      # Reusable cleaning functions
│   ├── rfm.py                # RFM computation
│   ├── clustering.py         # Model wrappers
│   └── visualization.py      # Plot helpers
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

**Why this matters:** Interviewers will look at your GitHub. A clean repo structure signals you've worked on real teams. The `src/` folder shows you know code shouldn't live only in notebooks.

### Key Libraries

```
pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, plotly, yellowbrick, kneed
```

---

## Phase 1: Data Understanding & Cleaning (Days 2–4)

This is where senior and junior diverge the most. Juniors rush to modeling. Seniors spend time here because garbage features produce garbage clusters.

### 1.1 Load and Inspect

```python
df = pd.read_excel('Online Retail.xlsx')
```

Immediately investigate:

- `df.shape` — you should see ~541K rows, 8 columns
- `df.info()` — check dtypes, notice CustomerID has nulls (important!)
- `df.describe()` — look for negative quantities and prices (they exist)
- `df.head(20)` — read the data like a human, not a machine

### 1.2 Critical Cleaning Decisions (Document Your Reasoning)

**This is the part most tutorials skip. A senior DS writes down WHY they made each choice.**

| Issue | What You'll Find | Decision to Make |
|-------|-----------------|-----------------|
| Missing CustomerID | ~135K rows have no customer | Drop them. You can't segment unknown customers. Log what % of revenue this represents. |
| Negative Quantities | These are returns/cancellations | Create a flag column `is_return` before dropping. Mention in your write-up that return behavior could be a feature in v2. |
| Zero/Negative Prices | Free items, adjustments | Drop rows with UnitPrice ≤ 0. Document the count. |
| Duplicates | Exact duplicate rows exist | Check and drop. Report the count. |
| Non-UK customers | Dataset is mostly UK | Keep all countries for now. Note the distribution. You may subset later if clusters are geography-dominated. |
| Outliers | A few customers with absurdly high spend | Don't remove yet. Flag them. Let the EDA guide your decision. |

**Senior move:** Create a `data_cleaning_log` — a simple dictionary or markdown that records every decision and its impact on row count. Present this in your final report. This is what consulting firms love.

```python
cleaning_log = {
    "raw_rows": len(df),
    "after_drop_null_customer": None,  # fill as you go
    "after_drop_cancellations": None,
    "after_drop_zero_price": None,
    "after_drop_duplicates": None,
}
```

### 1.3 Feature Engineering: The Revenue Column

```python
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
```

This is your monetary foundation. Verify it makes sense — spot-check a few invoices manually.

---

## Phase 2: Exploratory Data Analysis (Days 4–6)

EDA for clustering is different from EDA for prediction. You're looking for **natural groupings** and **distribution shapes** (because K-means assumes roughly spherical clusters).

### 2.1 Business-Level EDA (Tell a Story)

Produce these analyses — each one should have a plot AND a one-sentence business insight:

1. **Revenue over time** — line plot, monthly aggregation. Is there seasonality? (Spoiler: yes, Q4 spike — it's a UK gift retailer.)
2. **Top 10 countries by revenue** — bar chart. UK dominates. Note this.
3. **Revenue distribution per customer** — histogram. You'll see extreme right skew. This matters for clustering.
4. **Orders per customer** — histogram. Same skew pattern.
5. **Basket size distribution** — items per invoice. What does a "typical" order look like?
6. **Time between purchases** — for repeat customers. This previews your Recency feature.

### 2.2 Statistical EDA (Inform Modeling Choices)

- **Skewness of TotalPrice per customer:** Compute it. If skewness > 2, you'll need log transformation before K-means (K-means hates skewed features).
- **Correlation between potential features:** Will multicollinearity collapse your PCA components?
- **Outlier magnitude:** Use IQR or z-scores. How many customers are 3+ standard deviations out?

**Senior insight to mention in your report:** "K-means uses Euclidean distance, which is sensitive to feature scale and skewness. My EDA revealed all three RFM features are heavily right-skewed (skewness > 3), which motivated the log transformation in preprocessing."

---

## Phase 3: RFM Feature Engineering (Days 6–8)

RFM is the backbone. Get this right.

### 3.1 Define Your Reference Date

```python
# Use the day after the last transaction
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
```

### 3.2 Compute RFM Table

```python
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,   # Recency
    'InvoiceNo': 'nunique',                                      # Frequency
    'TotalPrice': 'sum'                                          # Monetary
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
})
```

### 3.3 Validate the RFM Table

- `rfm.describe()` — are the ranges sensible?
- Check for negative Monetary (returns exceeding purchases). Decide: drop or floor at 0.
- How many customers do you have? (~4,300 after cleaning — this is your sample size for clustering)
- Plot histograms of R, F, and M. Confirm the skewness you spotted in EDA.

### 3.4 (Bonus) Additional Features to Consider

A senior DS doesn't stop at basic RFM. Consider adding:

| Feature | Computation | Why |
|---------|------------|-----|
| Tenure | Days between first and last purchase | Distinguishes new vs. loyal |
| Avg Order Value | Monetary / Frequency | Separates "many small orders" from "few big orders" |
| Purchase Variability | Std dev of days between orders | Regularity of behavior |
| Return Rate | Cancelled orders / total orders | Risk signal |

**Start with just R, F, M.** Add extras only if your initial clustering feels under-differentiated. This is how senior DS work: iterate, don't over-engineer upfront.

---

## Phase 4: Preprocessing for Clustering (Days 8–9)

This phase is where most Kaggle notebooks fail. Preprocessing choices directly determine cluster quality.

### 4.1 Handle Outliers

**Strategy: Winsorize, don't drop.**

```python
from scipy.stats import mstats

for col in ['Recency', 'Frequency', 'Monetary']:
    rfm[col] = mstats.winsorize(rfm[col], limits=[0, 0.05])  # cap top 5%
```

**Why winsorize instead of drop:** Dropping outliers means losing your best customers (highest spenders). You want to segment them, not remove them. Winsorizing caps extreme values without losing rows.

### 4.2 Log Transformation

```python
import numpy as np

rfm_log = rfm.copy()
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])
rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
```

### 4.3 Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = pd.DataFrame(
    scaler.fit_transform(rfm_log),
    index=rfm_log.index,
    columns=rfm_log.columns
)
```

**Always scale after log-transforming.** The order matters.

### 4.4 Verify Preprocessing Worked

Plot distributions of `rfm_scaled`. They should look roughly normal and be on comparable scales. If one feature still dominates, it will dominate cluster assignment.

---

## Phase 5: K-Means Clustering (Days 9–12)

### 5.1 Choosing K: The Elbow Method + Silhouette Analysis

**Don't just do the elbow method.** Everyone does that. Do both, and add the Gap Statistic for extra credit.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, labels))
```

**Plot both on a side-by-side figure.** The elbow might suggest k=4. Silhouette might peak at k=3 or k=4. When they disagree, go with the one that produces **more interpretable business segments**. This is a judgment call — document your reasoning.

```python
# Automated elbow detection
kl = KneeLocator(list(K_range), inertias, curve='convex', direction='decreasing')
print(f"Optimal k (elbow): {kl.knee}")
```

### 5.2 Fit Final K-Means

```python
k_optimal = 4  # or whatever your analysis suggests
km_final = KMeans(n_clusters=k_optimal, n_init=25, random_state=42)
rfm['KMeans_Cluster'] = km_final.fit_predict(rfm_scaled)
```

### 5.3 Silhouette Plot (Per-Cluster Visualization)

```python
from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sv = SilhouetteVisualizer(KMeans(n_clusters=k_optimal, n_init=10, random_state=42), ax=ax)
sv.fit(rfm_scaled)
sv.show()
```

**What to look for:** Roughly equal-width silhouette blades. If one cluster has a very thin blade, it may be poorly defined. Mention this in your analysis.

### 5.4 Cluster Profiling

```python
cluster_summary = rfm.groupby('KMeans_Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'median', 'count']
}).round(1)
```

**This table is the heart of your project.** From it, you'll derive personas.

---

## Phase 6: Alternative Clustering Methods (Days 12–15)

### 6.1 Hierarchical (Agglomerative) Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Dendrogram (use a sample if >2K points for readability)
sample = rfm_scaled.sample(500, random_state=42)
Z = linkage(sample, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90)
plt.title('Dendrogram (Ward Linkage, 500-point sample)')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()
```

**Fit and compare:**

```python
agg = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')
rfm['Hierarchical_Cluster'] = agg.fit_predict(rfm_scaled)
```

**Try multiple linkages:** ward, complete, average. Ward usually wins for RFM data. Document which you tried and why you chose your final one.

### 6.2 DBSCAN

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Step 1: Find optimal eps using k-distance graph
nn = NearestNeighbors(n_neighbors=2 * rfm_scaled.shape[1])  # 2 * dimensions
nn.fit(rfm_scaled)
distances, _ = nn.kneighbors(rfm_scaled)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('Points (sorted)')
plt.ylabel('k-distance')
plt.title('k-Distance Graph for eps Selection')
plt.show()
# Look for the "elbow" in this curve — that's your eps

# Step 2: Fit DBSCAN
db = DBSCAN(eps=0.8, min_samples=10)  # tune these based on k-distance graph
rfm['DBSCAN_Cluster'] = db.fit_predict(rfm_scaled)

# Step 3: Examine results
n_clusters_db = len(set(rfm['DBSCAN_Cluster'])) - (1 if -1 in rfm['DBSCAN_Cluster'].values else 0)
n_noise = (rfm['DBSCAN_Cluster'] == -1).sum()
print(f"DBSCAN found {n_clusters_db} clusters and {n_noise} noise points ({n_noise/len(rfm)*100:.1f}%)")
```

**Senior insight for your write-up:** DBSCAN will likely perform poorly here — and that's the point. Explain WHY:

- DBSCAN finds density-based clusters. RFM features, even after scaling, tend to form one dense core with sparse tails — not well-separated dense regions.
- The "noise" label (-1) will capture many customers. Discuss whether these are genuinely anomalous or just a sign that density-based clustering isn't the right tool here.
- **This comparison is what makes your project strong.** Anyone can run K-means. Explaining why DBSCAN fails here shows algorithmic understanding.

### 6.3 Comparison Table

| Metric | K-Means | Hierarchical | DBSCAN |
|--------|---------|-------------|--------|
| Silhouette Score | | | |
| Calinski-Harabasz | | | |
| Davies-Bouldin | | | |
| Number of Clusters | | | |
| Noise Points | N/A | N/A | |
| Business Interpretability | | | |

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

for name, labels in [('KMeans', rfm['KMeans_Cluster']),
                      ('Hierarchical', rfm['Hierarchical_Cluster']),
                      ('DBSCAN', rfm['DBSCAN_Cluster'])]:
    mask = labels != -1  # exclude noise for DBSCAN
    if mask.sum() < 2:
        continue
    print(f"\n{name}:")
    print(f"  Silhouette:       {silhouette_score(rfm_scaled[mask], labels[mask]):.3f}")
    print(f"  Calinski-Harabasz:{calinski_harabasz_score(rfm_scaled[mask], labels[mask]):.1f}")
    print(f"  Davies-Bouldin:   {davies_bouldin_score(rfm_scaled[mask], labels[mask]):.3f}")
```

**Senior take:** Don't just report numbers. Say which method you'd recommend to a business stakeholder and why. Hint: K-means or Hierarchical usually wins for this data. The recommendation should weigh interpretability as much as metrics.

---

## Phase 7: PCA for Visualization & Insight (Days 15–16)

### 7.1 Fit PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
rfm_pca = pca.fit_transform(rfm_scaled)

print("Explained variance ratios:", pca.explained_variance_ratio_)
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))
```

### 7.2 Understand the Components

```python
components_df = pd.DataFrame(
    pca.components_,
    columns=rfm_scaled.columns,
    index=[f'PC{i+1}' for i in range(3)]
).round(3)
print(components_df)
```

**Interpret this!** For example:

- PC1 might load heavily on Frequency and Monetary → "Customer Value" axis
- PC2 might load on Recency → "Engagement Recency" axis
- Name your principal components. This shows business thinking.

### 7.3 2D and 3D Visualizations

```python
import plotly.express as px

viz_df = pd.DataFrame({
    'PC1': rfm_pca[:, 0],
    'PC2': rfm_pca[:, 1],
    'PC3': rfm_pca[:, 2],
    'Cluster': rfm['KMeans_Cluster'].astype(str)
})

# 2D — clean, publishable
fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster',
                 title='Customer Segments in PCA Space',
                 labels={'PC1': 'PC1 — Customer Value', 'PC2': 'PC2 — Engagement Recency'},
                 opacity=0.6)
fig.show()

# 3D — for exploration
fig3d = px.scatter_3d(viz_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                       title='3D Customer Segments', opacity=0.5)
fig3d.show()
```

### 7.4 Scree Plot

```python
pca_full = PCA().fit(rfm_scaled)
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_, alpha=0.7)
plt.step(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), where='mid', color='red')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Scree Plot')
plt.show()
```

---

## Phase 8: Customer Personas & Business Recommendations (Days 16–19)

**This is the phase that turns a data science project into a business deliverable.** Most people skip this. Don't.

### 8.1 Build Persona Profiles

For each K-means cluster, compute:

```python
persona_stats = rfm.groupby('KMeans_Cluster').agg(
    avg_recency=('Recency', 'mean'),
    avg_frequency=('Frequency', 'mean'),
    avg_monetary=('Monetary', 'mean'),
    median_monetary=('Monetary', 'median'),
    customer_count=('Recency', 'count'),
    pct_of_customers=('Recency', lambda x: len(x) / len(rfm) * 100),
    total_revenue=('Monetary', 'sum'),
    pct_of_revenue=('Monetary', lambda x: x.sum() / rfm['Monetary'].sum() * 100)
).round(1)
```

### 8.2 Name Your Personas

Here's the framework. Your actual names depend on your cluster profiles:

| Cluster | Likely Profile | Persona Name | Business Priority |
|---------|---------------|-------------|------------------|
| Low R, High F, High M | Recent, frequent, high spend | **Champions** | Retain at all costs |
| Low R, Low F, Low M | Recent but one-time, low spend | **New Customers** | Nurture to repeat |
| High R, High F, High M | Used to be great, gone quiet | **At-Risk VIPs** | Win-back urgently |
| High R, Low F, Low M | Long gone, barely engaged | **Lost / Hibernating** | Low priority or reactivation test |
| Medium R, Medium F, Medium M | Solid but not spectacular | **Loyalists** | Upsell, increase basket size |

### 8.3 Business Recommendations Per Segment

For each persona, write 2–3 specific, actionable recommendations. Example format:

> **Champions (Cluster 2, 12% of customers, 48% of revenue)**
>
> - **Retention:** Enroll in loyalty or VIP program. Offer early access to new products.
> - **Growth:** Cross-sell complementary product categories based on purchase history.
> - **Risk management:** Set up automated alerts if a Champion's recency exceeds 60 days — trigger a personalized re-engagement email.

> **At-Risk VIPs (Cluster 0, 8% of customers, 15% of revenue)**
>
> - **Win-back campaign:** Send a "We miss you" email with a time-limited discount.
> - **Root cause analysis:** Survey this segment to understand why they left. Was it product? Price? Service?
> - **Financial impact:** If we recover even 20% of this segment, the estimated revenue uplift is $X.

**Senior move:** Quantify the business impact wherever possible. "If we re-engage 20% of At-Risk VIPs, that's ~$X in recovered annual revenue." Back-of-envelope math is fine — it shows business thinking.

### 8.4 Revenue Concentration Analysis

```python
# What % of customers drive what % of revenue?
rfm_sorted = rfm.sort_values('Monetary', ascending=False)
rfm_sorted['cumulative_revenue_pct'] = rfm_sorted['Monetary'].cumsum() / rfm_sorted['Monetary'].sum() * 100
rfm_sorted['customer_pct'] = np.arange(1, len(rfm_sorted) + 1) / len(rfm_sorted) * 100

plt.figure(figsize=(8, 5))
plt.plot(rfm_sorted['customer_pct'], rfm_sorted['cumulative_revenue_pct'])
plt.xlabel('% of Customers')
plt.ylabel('% of Revenue')
plt.title('Revenue Concentration Curve (Pareto)')
plt.axhline(y=80, color='r', linestyle='--', alpha=0.5)
plt.show()
```

You'll likely find something close to 80/20 — 20% of customers drive 80% of revenue. State this prominently.

---

## Phase 9: Documentation & Portfolio Presentation (Days 19–21)

### 9.1 README.md Structure

```markdown
# Customer Segmentation Analysis

## Business Problem
[1 paragraph: what question are we answering and why it matters]

## Key Findings
- [Bullet 1: e.g., "4 distinct customer segments identified..."]
- [Bullet 2: e.g., "20% of customers drive 78% of revenue..."]
- [Bullet 3: e.g., "K-means outperformed DBSCAN for this data because..."]

## Methodology
[Brief pipeline description with a flow diagram]

## Results
[Link to notebook or summary with key visualizations]

## How to Reproduce
[Setup instructions]
```

### 9.2 Final Notebook Structure

Your `05_personas_and_recommendations.ipynb` is your "executive summary" notebook. It should:

1. Start with the business question (1 cell, markdown)
2. Show the final segmentation visualization (PCA scatter)
3. Present the persona table with stats
4. Give recommendations per segment
5. End with limitations and next steps

### 9.3 Limitations to Mention (Shows Maturity)

These are the limitations a senior DS would flag:

- **Temporal snapshot:** RFM is computed at one point in time. In production, you'd recompute monthly and track segment migration.
- **K-means assumptions:** Assumes spherical, equal-variance clusters. Your data may not satisfy this perfectly.
- **No behavioral features:** Purchase category, browsing data, and returns could improve segmentation but weren't available.
- **Country heterogeneity:** UK customers dominate. Mixed geography might create segments based on location rather than behavior. Consider subsetting to UK-only and comparing results.
- **Validation:** Without ground truth, cluster quality is evaluated via internal metrics only. A/B testing segment-specific campaigns would be the real validation.

---

## Interview-Ready Talking Points

Practice answering these questions about your project:

1. **"Why K-means over other methods?"** — Talk about the comparison you did. K-means gave the best silhouette score AND the most interpretable segments. Hierarchical was comparable but harder to scale. DBSCAN identified too many noise points because RFM data doesn't form density-separated clusters.

2. **"How did you choose K?"** — Elbow method suggested K=4, silhouette analysis confirmed it. I also checked K=3 and K=5 for interpretability. K=4 gave the most distinct, actionable personas.

3. **"What would you do differently in production?"** — Recompute segments monthly. Build a pipeline that assigns new customers to existing clusters (use `km.predict()`). Track segment transitions over time as a churn early-warning signal. Consider adding behavioral features beyond RFM.

4. **"How did you handle the preprocessing?"** — Log-transformed all features due to extreme right skew (showed skewness values). Standardized with StandardScaler. Winsorized top 5% instead of dropping to retain high-value customers in the analysis.

5. **"What business impact could this have?"** — Personalized marketing spend by segment: cut spend on Lost customers, invest heavily in retaining Champions and reactivating At-Risk VIPs. Estimated X% revenue recovery from win-back campaigns targeting At-Risk VIPs. Shift from one-size-fits-all email blasts to segment-specific messaging.

---

## Daily Execution Checklist

| Day | Phase | Key Deliverable |
|-----|-------|----------------|
| 1 | Setup | Repo created, data downloaded, environment ready |
| 2–3 | Cleaning | Clean dataset + data_cleaning_log |
| 4–6 | EDA | 6+ visualizations with business insights |
| 6–8 | RFM | Validated RFM table, distribution plots |
| 8–9 | Preprocessing | Scaled, transformed feature matrix |
| 9–11 | K-Means | Elbow + silhouette analysis, optimal K chosen |
| 12–14 | Alternatives | Hierarchical + DBSCAN fitted and compared |
| 15–16 | PCA | 2D/3D visualizations, component interpretation |
| 16–18 | Personas | Named segments with stats and recommendations |
| 19–21 | Polish | Clean notebooks, README, portfolio-ready |

---

*This plan was designed to build the skills a DS role requires: methodological rigor, preprocessing judgment, model comparison, and — most importantly — translating numbers into business decisions.*
