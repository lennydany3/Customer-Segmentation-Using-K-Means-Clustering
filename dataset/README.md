# Tamil Nadu gift-shop customer datasets

Three synthetic customer files (70, 85 and 100 customers) for a shop that sells **Key Chains**, **Bouquets** and **Miniature Toys**. Upload any of them on the website.

| Column | Meaning | Used for clustering? |
|---|---|---|
| Name | Customer name | No |
| Age | Age in years | **Yes** |
| Gender | Female / Male | No (shown in the table) |
| District | One of 20 Tamil Nadu districts | No (top district shown per cluster) |
| Product_Type | Key Chains, Bouquet or Miniature Toys | No (top product shown per cluster) |
| Repeat_Purchases | Number of times the customer came back and bought again | **Yes** |

The files have no answer column: the website has to discover the customer groups on its own. The dashboard shows the silhouette score; the Accuracy card appears only for files that include a `True_Segment` column.

## Results with the app's recommended K

| File | Customers | Hidden groups | Recommended K | Silhouette |
|---|---|---|---|---|
| tn_gift_shop_70.csv | 70 | 3 | 3 | 0.698 |
| tn_gift_shop_85.csv | 85 | 4 | 4 | 0.705 |
| tn_gift_shop_100.csv | 100 | 5 | 5 | 0.665 |

The data was generated from these groups: Loyal Young Fans, Occasional Young Buyers, Regular Adult Customers, Occasional Senior Buyers and Loyal Senior Gifters. Young customers mostly buy Key Chains and Miniature Toys, while older customers mostly buy Bouquets. On every file the website finds the right number of groups and gives them these same names.

## Regenerate

```bash
python dataset/generate_datasets.py
```

The generator uses fixed random seeds, so it always produces the same files.
