# Synthetic customer datasets

Five artificial customer files (1,000 to 1,470 rows each) for testing the app. Upload any of them on the website.

Each file has these columns:

| Column | Used for clustering? |
|---|---|
| Customer_ID, Name, Gender, Age, City | No (Age is shown in cluster profiles) |
| **Annual_Income** (k$), **Spending_Score** (1 to 100) | **Yes** |
| Visits_Per_Month, Membership_Years, Preferred_Channel | No (extra realistic columns) |
| **True_Segment** | No. Used only to calculate **accuracy** |

Because every customer was generated from a known segment, the website compares its clusters with `True_Segment` and shows **Accuracy**, **ARI** and **NMI**. Cluster numbers are arbitrary, so each cluster is first matched to the true segment it overlaps most (Hungarian matching) before counting correct customers.

## Results with the app's recommended K

| File | Rows | True segments | What makes it hard | Recommended K | Silhouette | Accuracy |
|---|---|---|---|---|---|---|
| 01_mall_customers_1200.csv | 1200 | 5 | Large central group, moderate overlap | 5 | 0.671 | 99.6% |
| 02_retail_chain_1470.csv | 1470 | 6 | Unequal sizes and spreads, 30 outliers | 6 | 0.666 | 97.7% |
| 03_ecommerce_1000.csv | 1000 | 4 | Stretched, tilted (correlated) segments | 4 | 0.735 | 99.7% |
| 04_supermarket_messy_1340.csv | 1340 (1300 valid) | 5 | Imbalanced, overlapping, 40 dirty rows to skip | 5 | 0.632 | 98.8% |
| 05_fashion_boutique_1050.csv | 1050 | 3 | Fewest segments (easy benchmark) | 3 | 0.759 | 100% |

If you move the K slider away from the true number of segments, accuracy drops. That is expected and is a good way to show why choosing K matters.

## Regenerate

```bash
python dataset/generate_datasets.py
```

The generator uses fixed random seeds, so it always produces the same files.
