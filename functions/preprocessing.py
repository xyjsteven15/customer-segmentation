import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler


def preprocess_rfm(rfm, winsorize_limit=0.05):
    """
    Full preprocessing pipeline for RFM features:
    1. Winsorize top winsorize_limit of each feature
    2. Log transform (log1p)
    3. StandardScaler (z-score normalization)

    Parameters:
        rfm             : RFM DataFrame with columns [Recency, Frequency, Monetary]
        winsorize_limit : top percentile to cap (default 0.05 = top 5%)

    Returns:
        rfm_scaled : preprocessed DataFrame, same index as input
        scaler     : fitted StandardScaler (use scaler.transform() for new data)
    """
    # Step 1: Winsorize
    rfm_clean = rfm.copy()
    for col in ['Recency', 'Frequency', 'Monetary']:
        rfm_clean[col] = mstats.winsorize(rfm_clean[col], limits=[0, winsorize_limit])

    # Step 2: Log transform
    rfm_log = rfm_clean.copy()
    for col in ['Recency', 'Frequency', 'Monetary']:
        rfm_log[col] = np.log1p(rfm_log[col])

    # Step 3: Standardize
    scaler = StandardScaler()
    rfm_scaled = pd.DataFrame(
        scaler.fit_transform(rfm_log),
        index=rfm_log.index,
        columns=rfm_log.columns
    )

    return rfm_scaled, scaler
