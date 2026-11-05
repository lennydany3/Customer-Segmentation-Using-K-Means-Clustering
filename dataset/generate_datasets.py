"""
Generates the synthetic Tamil Nadu gift-shop customer datasets in this folder.

Columns: Name, Age, Gender, District, Product_Type, Repeat_Purchases
  - Age and Repeat_Purchases are the clustering features.
  - Each customer is drawn from one of the SEGMENTS below, but the segment is not written to the
    file: the app has to discover the groups on its own.

Run:  python dataset/generate_datasets.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent

DISTRICTS = ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Erode",
             "Vellore", "Thoothukudi", "Thanjavur", "Kanniyakumari", "Dindigul", "Kancheepuram",
             "Tiruppur", "Namakkal", "Karur", "Cuddalore", "Villupuram", "The Nilgiris", "Sivaganga"]
FEMALE = ["Priya", "Lakshmi", "Divya", "Kavya", "Meena", "Anitha", "Revathi", "Sangeetha", "Keerthana",
          "Nandhini", "Deepa", "Gayathri", "Harini", "Janani", "Kalaivani", "Abinaya", "Swetha", "Vaishnavi",
          "Yamuna", "Pavithra", "Malathi", "Saranya", "Bhuvana", "Indhu", "Shalini"]
MALE = ["Arun", "Karthik", "Vignesh", "Suresh", "Prakash", "Dinesh", "Bala", "Senthil", "Gokul",
        "Hari", "Manikandan", "Naveen", "Ramesh", "Sathish", "Tamilselvan", "Vasanth", "Ajith", "Surya",
        "Murugan", "Saravanan", "Ashwin", "Pranav", "Kishore", "Elango", "Raghav"]
INITIALS = list("ABCDEGJKMNPRSTV")
PRODUCTS = ["Key Chains", "Bouquet"]

# segment name: (mean age, sd age, mean repeats, sd repeats, share buying Key Chains)
SEGMENTS = {
    "Loyal Young Fans": (21, 2.5, 15, 2.0, 0.75),
    "Occasional Young Buyers": (23, 3.0, 3, 1.3, 0.60),
    "Regular Adult Customers": (37, 3.5, 9, 1.8, 0.35),
    "Occasional Senior Buyers": (56, 4.0, 3, 1.3, 0.15),
    "Loyal Senior Gifters": (55, 4.0, 15, 2.0, 0.10),
}


def make(seed, sizes):
    rng = np.random.default_rng(seed)
    rows = []
    for name, n in sizes.items():
        a_mu, a_sd, r_mu, r_sd, p_key = SEGMENTS[name]
        for _ in range(n):
            gender = rng.choice(["Female", "Male"])
            first = rng.choice(FEMALE if gender == "Female" else MALE)
            rows.append({
                "Name": f"{first} {rng.choice(INITIALS)}",
                "Age": int(np.clip(round(rng.normal(a_mu, a_sd)), 16, 75)),
                "Gender": gender,
                "District": rng.choice(DISTRICTS),
                "Product_Type": "Key Chains" if rng.random() < p_key else "Bouquet",
                "Repeat_Purchases": int(np.clip(round(rng.normal(r_mu, r_sd)), 0, 25)),
                "True_Segment": name,
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


DATASETS = {
    "tn_gift_shop_70.csv": (11, {"Loyal Young Fans": 22, "Regular Adult Customers": 25, "Occasional Senior Buyers": 23}),
    "tn_gift_shop_85.csv": (22, {"Loyal Young Fans": 20, "Occasional Young Buyers": 22,
                                 "Regular Adult Customers": 23, "Occasional Senior Buyers": 20}),
    "tn_gift_shop_100.csv": (33, {"Loyal Young Fans": 20, "Occasional Young Buyers": 20, "Regular Adult Customers": 22,
                                  "Occasional Senior Buyers": 19, "Loyal Senior Gifters": 19}),
}

if __name__ == "__main__":
    for file, (seed, sizes) in DATASETS.items():
        df = make(seed, sizes)
        segments = df["True_Segment"].nunique()
        df.drop(columns="True_Segment").to_csv(OUT / file, index=False)
        print(f"{file}: {len(df)} customers, {segments} hidden segments")
