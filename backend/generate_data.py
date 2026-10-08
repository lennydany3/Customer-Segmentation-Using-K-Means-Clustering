import os
import numpy as np
import pandas as pd

# Directory setup: ensure backend/dataset/ exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_PATH = os.path.join(DATASET_DIR, "customers.csv")

def generate_customer_dataset(output_path: str = CSV_PATH, n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generates an artificial customer dataset with 200 records deliberately structured
    into 4 distinct market segments (50 customers each) based on:
      1. High Income, High Spending (VIP / Target)
      2. High Income, Low Spending (Careful Spenders)
      3. Low Income, High Spending (Trendsetters / Enthusiasts)
      4. Low Income, Low Spending (Budget Shoppers)

    Columns:
      - Customer_ID: Unique identifier (e.g., C001, C002, ...)
      - Age: Customer age (18 - 70)
      - Annual_Income: Annual income in thousands (e.g., $15k - $140k)
      - Spending_Score: Mall spending behavior score (1 - 100)
    """
    np.random.seed(seed)
    n_per_cluster = n_samples // 4

    # Group 1: High Income, High Spending
    # Typically younger professionals and affluent shoppers
    age_g1 = np.random.normal(loc=32, scale=6, size=n_per_cluster)
    income_g1 = np.random.normal(loc=95, scale=12, size=n_per_cluster)
    spending_g1 = np.random.normal(loc=82, scale=8, size=n_per_cluster)

    # Group 2: High Income, Low Spending
    # Typically mature, cautious, affluent savers
    age_g2 = np.random.normal(loc=46, scale=9, size=n_per_cluster)
    income_g2 = np.random.normal(loc=90, scale=12, size=n_per_cluster)
    spending_g2 = np.random.normal(loc=20, scale=8, size=n_per_cluster)

    # Group 3: Low Income, High Spending
    # Typically younger, aspirational trendsetters
    age_g3 = np.random.normal(loc=25, scale=5, size=n_per_cluster)
    income_g3 = np.random.normal(loc=28, scale=8, size=n_per_cluster)
    spending_g3 = np.random.normal(loc=78, scale=9, size=n_per_cluster)

    # Group 4: Low Income, Low Spending
    # Budget-conscious shoppers
    age_g4 = np.random.normal(loc=48, scale=10, size=n_per_cluster)
    income_g4 = np.random.normal(loc=30, scale=8, size=n_per_cluster)
    spending_g4 = np.random.normal(loc=22, scale=8, size=n_per_cluster)

    # Combine all 4 groups
    ages = np.concatenate([age_g1, age_g2, age_g3, age_g4])
    incomes = np.concatenate([income_g1, income_g2, income_g3, income_g4])
    spendings = np.concatenate([spending_g1, spending_g2, spending_g3, spending_g4])

    # Clip values to realistic domain ranges
    ages = np.clip(np.round(ages), 18, 70).astype(int)
    incomes = np.clip(np.round(incomes, 1), 15.0, 140.0)
    spendings = np.clip(np.round(spendings), 1, 100).astype(int)

    # Create formatted customer IDs: C001, C002, ..., C200
    customer_ids = [f"C{i+1:03d}" for i in range(n_samples)]

    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Age": ages,
        "Annual_Income": incomes,
        "Spending_Score": spendings
    })

    # Shuffle the dataset so clusters aren't ordered sequentially
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Re-assign IDs in shuffled order for clean presentation
    df["Customer_ID"] = [f"C{i+1:03d}" for i in range(n_samples)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path} ({len(df)} records)")
    return df

def get_or_create_dataset(output_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Returns the customer dataframe. If dataset doesn't exist, it is generated automatically.
    """
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    return generate_customer_dataset(output_path)

if __name__ == "__main__":
    generate_customer_dataset()

