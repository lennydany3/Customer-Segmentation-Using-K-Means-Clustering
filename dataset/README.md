# Tamil Nadu gift-shop customer datasets

Three synthetic customer files (70, 85 and 100 customers) for a shop that sells **Key Chains** and **Bouquets**. Upload any of them on the website.

| Column | Meaning | Used for clustering? |
|---|---|---|
| Name | Customer name | No |
| Age | Age in years | **Yes** |
| Gender | Female / Male | No (shown in the table) |
| District | One of 20 Tamil Nadu districts | No (top district shown per cluster) |
| Product_Type | Key Chains or Bouquet | No (top product shown per cluster) |
| Repeat_Purchases | Number of times the customer came back and bought again | **Yes** |
| True_Segment | The real group each customer was generated from | No. Used only to calculate **accuracy** |

Because every customer was generated from a known group, the website compares its clusters with `True_Segment` and shows **Accuracy**, **ARI** and **NMI**. Delete the `True_Segment` column if you want to upload a file without accuracy.

## Results with the app's recommended K

| File | Customers | True segments | Recommended K | Silhouette | Accuracy |
|---|---|---|---|---|---|
| tn_gift_shop_70.csv | 70 | 3 | 3 | 0.698 | 100% |
| tn_gift_shop_85.csv | 85 | 4 | 4 | 0.705 | 98.8% |
| tn_gift_shop_100.csv | 100 | 5 | 5 | 0.665 | 99.0% |

Segments used: Loyal Young Fans, Occasional Young Buyers, Regular Adult Customers, Occasional Senior Buyers and Loyal Senior Gifters. Young customers mostly buy Key Chains and older customers mostly buy Bouquets.

If you move the K slider away from the true number of segments, accuracy drops. That shows why choosing K matters.

## Regenerate

```bash
python dataset/generate_datasets.py
```

The generator uses fixed random seeds, so it always produces the same files.
