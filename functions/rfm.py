import pandas as pd
import numpy as np


def compute_rfm(df, customer_col='Customer ID', date_col='InvoiceDate',
                invoice_col='Invoice', revenue_col='TotalPrice'):
    """
    Compute RFM (Recency, Frequency, Monetary) table from a cleaned transactions dataframe.

    Parameters:
        df            : cleaned transactions DataFrame
        customer_col  : column name for customer ID
        date_col      : column name for invoice date
        invoice_col   : column name for invoice number
        revenue_col   : column name for transaction revenue

    Returns:
        rfm : DataFrame with columns [Recency, Frequency, Monetary] indexed by customer_col
    """
    reference_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (reference_date - x.max()).days),
        Frequency=(invoice_col, 'nunique'),
        Monetary=(revenue_col, 'sum')
    ).round(2)

    return rfm


def validate_rfm(rfm):
    """
    Print a validation summary of the RFM table.
    Flags negative Monetary values and shows descriptive stats.
    """
    print(f"Customers: {len(rfm)}")
    print(f"Negative Monetary: {(rfm['Monetary'] < 0).sum()}")
    print(f"Zero Monetary: {(rfm['Monetary'] == 0).sum()}")
    print()
    print(rfm.describe().round(1))
