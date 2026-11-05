"""
Generates the synthetic customer datasets in this folder.

Every file has a True_Segment column holding the segment each customer was generated from.
The app ignores it for clustering and uses it only to score the clusters (accuracy, ARI, NMI).

Run:  python dataset/generate_datasets.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
FIRST = ["Aarav", "Diya", "Ishaan", "Meera", "Rohan", "Sneha", "Arjun", "Kavya", "Vikram", "Ananya",
         "Rahul", "Priya", "Karthik", "Lakshmi", "Nikhil", "Pooja", "Siddharth", "Divya", "Aditya", "Nisha"]
CITIES = ["Chennai", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Kolkata", "Pune", "Kochi", "Coimbatore", "Madurai"]
CHANNELS = ["In-store", "Online", "Mobile App", "Mixed"]


def segment(rng, n, income, spending, cov, age, name):
    """Draws n customers from a 2-D Gaussian (income, spending) with full covariance."""
    xy = rng.multivariate_normal([income, spending], cov, size=n)
    ages = np.clip(rng.normal(age[0], age[1], n), 18, 75).round()
    return pd.DataFrame({"Annual_Income": xy[:, 0], "Spending_Score": xy[:, 1], "Age": ages, "True_Segment": name})


def finish(rng, df, prefix, messy=0):
    """Shuffles, clips to realistic ranges, adds descriptive columns and optional dirty rows."""
    df = df.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    df["Annual_Income"] = df["Annual_Income"].clip(8, 200).round(1)
    df["Spending_Score"] = df["Spending_Score"].clip(1, 100).round().astype(int)
    n = len(df)
    df.insert(0, "Customer_ID", [f"{prefix}{i:05d}" for i in range(1, n + 1)])
    df.insert(1, "Name", [f"{FIRST[i % len(FIRST)]} {chr(65 + (i * 7) % 26)}." for i in range(n)])
    df.insert(2, "Gender", rng.choice(["Female", "Male"], n, p=[0.52, 0.48]))
    df["City"] = rng.choice(CITIES, n)
    df["Preferred_Channel"] = rng.choice(CHANNELS, n)
    df["Visits_Per_Month"] = np.clip(rng.poisson(1 + df["Spending_Score"] / 15), 0, 20)
    df["Membership_Years"] = np.clip(rng.exponential(3, n).round(1), 0, 25)
    df["Age"] = df["Age"].astype(int)
    df = df[["Customer_ID", "Name", "Gender", "Age", "City", "Annual_Income", "Spending_Score",
             "Visits_Per_Month", "Membership_Years", "Preferred_Channel", "True_Segment"]]
    if messy:
        # dirty rows the app must skip: blanks, text in numeric columns, impossible values
        df = df.astype(object)
        idx = rng.choice(n, messy, replace=False)
        for j, i in enumerate(idx):
            col = "Annual_Income" if j % 2 == 0 else "Spending_Score"
            df.at[i, col] = ["", "N/A", "unknown", "--"][j % 4]
    return df


def build():
    specs = {}

    # 1. Mall-style: five classic segments, moderate overlap
    rng = np.random.default_rng(101)
    specs["01_mall_customers_1200.csv"] = finish(rng, pd.concat([
        segment(rng, 470, 55, 50, [[60, 0], [0, 40]], (42, 12), "Standard"),
        segment(rng, 190, 88, 82, [[110, 0], [0, 45]], (32, 5), "Premium VIP"),
        segment(rng, 180, 88, 18, [[110, 0], [0, 45]], (41, 11), "Careful Saver"),
        segment(rng, 180, 25, 80, [[30, 0], [0, 50]], (25, 5), "Trendsetter"),
        segment(rng, 180, 26, 20, [[30, 0], [0, 50]], (45, 13), "Budget Seeker"),
    ]), "MC")

    # 2. Retail chain: six segments of very different size and spread, plus 2% outliers
    rng = np.random.default_rng(202)
    parts = [
        segment(rng, 420, 48, 45, [[50, 15], [15, 40]], (38, 11), "Mainstream"),
        segment(rng, 260, 100, 85, [[150, 20], [20, 25]], (36, 8), "Luxury Spender"),
        segment(rng, 230, 105, 22, [[160, -20], [-20, 40]], (47, 10), "Affluent Saver"),
        segment(rng, 200, 22, 82, [[18, 0], [0, 30]], (23, 4), "Young Trendsetter"),
        segment(rng, 210, 24, 15, [[20, 0], [0, 30]], (55, 12), "Price Sensitive"),
        segment(rng, 150, 65, 72, [[25, 0], [0, 18]], (30, 6), "Rising Professional"),
    ]
    df = pd.concat(parts, ignore_index=True)
    out = rng.choice(len(df), 30, replace=False)  # outliers: same label, pushed away from the centre
    df.loc[out, "Annual_Income"] += rng.normal(0, 18, 30)
    df.loc[out, "Spending_Score"] += rng.normal(0, 14, 30)
    specs["02_retail_chain_1470.csv"] = finish(rng, df, "RC", messy=0)

    # 3. E-commerce: four elongated (correlated) segments
    rng = np.random.default_rng(303)
    specs["03_ecommerce_1000.csv"] = finish(rng, pd.concat([
        segment(rng, 280, 40, 30, [[120, 70], [70, 70]], (34, 9), "Deal Hunter"),
        segment(rng, 240, 95, 85, [[140, 60], [60, 40]], (37, 7), "Power Buyer"),
        segment(rng, 250, 100, 25, [[150, -50], [-50, 45]], (44, 10), "Window Shopper"),
        segment(rng, 230, 30, 85, [[40, -15], [-15, 35]], (24, 4), "Impulse Buyer"),
    ]), "EC")

    # 4. Supermarket: five imbalanced, overlapping segments with 40 dirty rows to clean
    rng = np.random.default_rng(404)
    specs["04_supermarket_messy_1340.csv"] = finish(rng, pd.concat([
        segment(rng, 520, 52, 48, [[90, 10], [10, 70]], (40, 13), "Regular Shopper"),
        segment(rng, 150, 92, 80, [[120, 0], [0, 70]], (35, 7), "Premium Family"),
        segment(rng, 220, 95, 22, [[130, 10], [10, 70]], (46, 10), "Thrifty High Earner"),
        segment(rng, 130, 24, 82, [[40, 0], [0, 60]], (26, 6), "Student Splurger"),
        segment(rng, 320, 22, 20, [[40, 5], [5, 60]], (58, 11), "Senior Budget"),
    ]), "SM", messy=40)

    # 5. Fashion boutique: three well separated segments (easy benchmark)
    rng = np.random.default_rng(505)
    specs["05_fashion_boutique_1050.csv"] = finish(rng, pd.concat([
        segment(rng, 400, 35, 30, [[70, 0], [0, 70]], (45, 12), "Occasional Buyer"),
        segment(rng, 350, 110, 80, [[120, 0], [0, 60]], (34, 7), "Fashion Loyalist"),
        segment(rng, 300, 40, 85, [[60, 0], [0, 40]], (24, 4), "Trend Chaser"),
    ]), "FB")
    return specs


if __name__ == "__main__":
    for name, df in build().items():
        df.to_csv(OUT / name, index=False)
        print(f"{name}: {len(df)} rows, {df['True_Segment'].nunique()} segments")
