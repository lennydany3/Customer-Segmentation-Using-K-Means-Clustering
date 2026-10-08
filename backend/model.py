"""
model.py: Machine Learning logic for Customer Segmentation using K-Means.
- Feature Selection: Annual_Income and Spending_Score
- Normalization: StandardScaler (zero mean, unit variance)
- Model: KMeans clustering with scikit-learn
- Metrics: Inertia (WCSS for Elbow method) and Silhouette Score
- Interpretation: Dynamic cluster labeling and marketing recommendations
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from generate_data import get_or_create_dataset

class CustomerSegmentationModel:
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.scaler: StandardScaler = StandardScaler()
        self.model: Optional[KMeans] = None
        self.current_k: int = 4
        self.cluster_metadata: Dict[int, Dict[str, Any]] = {}
        
        # Load dataset initially
        self.load_data()

    def load_data(self) -> pd.DataFrame:
        """Loads customer dataset from CSV or generates it if missing."""
        if self.dataset_path:
            self.df = get_or_create_dataset(self.dataset_path)
        else:
            self.df = get_or_create_dataset()
        return self.df

    def get_features(self) -> np.ndarray:
        """
        Extracts features used for clustering:
        1. Annual_Income ($k)
        2. Spending_Score (1-100)
        """
        if self.df.empty:
            self.load_data()
        return self.df[["Annual_Income", "Spending_Score"]].values

    def compute_elbow(self, k_min: int = 2, k_max: int = 8) -> Tuple[List[int], List[float]]:
        """
        Calculates Within-Cluster Sum of Squares (WCSS / Inertia) for candidate values of K.
        WCSS = sum of squared Euclidean distances between each point and its cluster centroid.
        """
        X = self.get_features()
        # Scale features before computing distances
        temp_scaler = StandardScaler()
        X_scaled = temp_scaler.fit_transform(X)

        k_values = list(range(k_min, k_max + 1))
        wcss = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            wcss.append(float(round(km.inertia_, 2)))

        return k_values, wcss

    def _generate_cluster_label_and_strategy(
        self,
        avg_income: float,
        avg_spending: float,
        mean_income: float,
        mean_spending: float,
        cluster_id: int,
        k: int
    ) -> Tuple[str, str, str]:
        """
        Derives an intuitive marketing persona, description, and actionable business strategy
        based on how the cluster's average income and spending compare with the overall store averages.
        """
        is_high_income = avg_income >= mean_income
        is_high_spending = avg_spending >= mean_spending

        if is_high_income and is_high_spending:
            label = "Premium VIP Shoppers"
            desc = "High income with high spending habits. Highly engaged luxury and priority buyers."
            strat = "Reward loyalty with VIP exclusive perks, early access to new collections, and personalized concierge offers."
        elif is_high_income and not is_high_spending:
            label = "Careful Savers"
            desc = "High financial capacity but disciplined and selective with discretionary expenditure."
            strat = "Emphasize premium product quality, long-term durability, and targeted value guarantees to convert savings into purchases."
        elif not is_high_income and is_high_spending:
            label = "Trendsetters / Enthusiasts"
            desc = "Lower or moderate income, but aggressive spenders who prioritize trends and lifestyle."
            strat = "Promote viral/trending items, flash sales, installment plans (BNPL), and social reward programs."
        else:
            label = "Budget / Value Seekers"
            desc = "Cost-sensitive customers with modest earnings and low discretionary spending frequency."
            strat = "Attract with seasonal clearance discounts, essential bundle deals, coupons, and everyday low prices."

        # If K > 4, append cluster identifier to avoid duplicate identical persona names in UI
        if k > 4:
            label = f"{label} (Group {cluster_id + 1})"

        return label, desc, strat

    def train(self, k: int = 4) -> Dict[str, Any]:
        """
        Trains K-Means clustering on the dataset with the chosen K:
        1. Scales Annual_Income and Spending_Score using StandardScaler.
        2. Fits KMeans(n_clusters=k, random_state=42, n_init=10).
        3. Computes centroids in original units using inverse_transform.
        4. Calculates per-cluster statistics (count, avg age, avg income, avg spending).
        5. Computes silhouette score to evaluate clustering cohesion and separation.
        """
        self.current_k = k
        X = self.get_features()

        # Step 1: Scale features (StandardScaler: z = (x - u) / s)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Step 2: Fit KMeans
        self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = self.model.fit_predict(X_scaled)
        self.df["Cluster"] = labels

        # Step 3: Compute Centroids in original feature space ($k, score)
        raw_centroids = self.model.cluster_centers_
        orig_centroids = self.scaler.inverse_transform(raw_centroids)

        # Step 4: Overall benchmarks for auto-labeling
        mean_income = float(self.df["Annual_Income"].mean())
        mean_spending = float(self.df["Spending_Score"].mean())
        total_customers = len(self.df)

        # Step 5: Calculate Silhouette Score (ranges from -1 to 1)
        sil_score = float(round(silhouette_score(X_scaled, labels), 3)) if k > 1 else 0.0

        # Step 6: Build per-cluster summaries
        self.cluster_metadata = {}
        clusters_summary = []
        centroids_summary = []

        for cid in range(k):
            c_mask = self.df["Cluster"] == cid
            c_df = self.df[c_mask]

            count = int(len(c_df))
            pct = round((count / total_customers) * 100, 1)
            avg_age = float(round(c_df["Age"].mean(), 1)) if count > 0 else 0.0
            avg_inc = float(round(c_df["Annual_Income"].mean(), 1)) if count > 0 else 0.0
            avg_sp = float(round(c_df["Spending_Score"].mean(), 1)) if count > 0 else 0.0
            
            centroid_income = float(round(orig_centroids[cid][0], 1))
            centroid_spending = float(round(orig_centroids[cid][1], 1))

            label, desc, strat = self._generate_cluster_label_and_strategy(
                avg_inc, avg_sp, mean_income, mean_spending, cid, k
            )

            # Store metadata for real-time predictions
            self.cluster_metadata[cid] = {
                "label": label,
                "description": desc,
                "strategy": strat,
                "centroid": [centroid_income, centroid_spending]
            }

            clusters_summary.append({
                "cluster_id": cid,
                "label": label,
                "customer_count": count,
                "percentage": pct,
                "avg_age": avg_age,
                "avg_income": avg_inc,
                "avg_spending": avg_sp,
                "centroid": [centroid_income, centroid_spending],
                "description": desc,
                "strategy": strat
            })

            centroids_summary.append({
                "cluster": cid,
                "income": centroid_income,
                "spending": centroid_spending,
                "label": label
            })

        # Format individual customer points for visualization
        points = [
            {
                "id": str(row["Customer_ID"]),
                "age": int(row["Age"]),
                "income": float(row["Annual_Income"]),
                "spending": int(row["Spending_Score"]),
                "cluster": int(row["Cluster"])
            }
            for _, row in self.df.iterrows()
        ]

        return {
            "k": k,
            "total_customers": total_customers,
            "silhouette_score": sil_score,
            "clusters": clusters_summary,
            "points": points,
            "centroids": centroids_summary
        }

    def predict(self, age: int, income: float, spending: int) -> Dict[str, Any]:
        """
        Classifies a new customer into one of the trained clusters:
        1. Scales [income, spending] using the fitted StandardScaler.
        2. Assigns nearest cluster using model.predict.
        3. Returns cluster ID, persona label, marketing strategy, and centroid.
        """
        if self.model is None:
            # If model hasn't been trained yet, train default k=4
            self.train(k=4)

        # Scale customer input features
        input_features = np.array([[income, spending]])
        scaled_input = self.scaler.transform(input_features)

        # Predict cluster index
        cluster_id = int(self.model.predict(scaled_input)[0])
        meta = self.cluster_metadata.get(cluster_id, {
            "label": f"Cluster {cluster_id}",
            "description": "Customer segment",
            "strategy": "Engage customer with standard marketing offers.",
            "centroid": [0.0, 0.0]
        })

        return {
            "cluster_id": cluster_id,
            "label": meta["label"],
            "description": meta["description"],
            "strategy": meta["strategy"],
            "income": float(income),
            "spending_score": int(spending),
            "age": int(age),
            "nearest_centroid": meta["centroid"]
        }

# Global singleton instance for FastAPI backend
ml_model = CustomerSegmentationModel()

if __name__ == "__main__":
    print("Testing ML Model...")
    model = CustomerSegmentationModel()
    k_vals, wcss_vals = model.compute_elbow()
    print("Elbow results:", list(zip(k_vals, wcss_vals)))
    result = model.train(k=4)
    print(f"Trained K=4: Silhouette Score = {result['silhouette_score']}")
    for c in result["clusters"]:
        print(f"Cluster {c['cluster_id']}: {c['label']} | {c['customer_count']} customers | Centroid: {c['centroid']}")
    pred = model.predict(age=28, income=95.0, spending=85)
    print("Prediction test:", pred)

