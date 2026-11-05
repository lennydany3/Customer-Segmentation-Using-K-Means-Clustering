"""
model.py: Machine Learning logic for Customer Segmentation using K-Means.
- Data: an uploaded customer CSV (nothing is bundled with the backend)
- Feature Selection: Annual_Income and Spending_Score
- Normalization: StandardScaler (zero mean, unit variance)
- Model: KMeans clustering with scikit-learn
- Metrics: Inertia (WCSS for Elbow method) and Silhouette Score
- Interpretation: Dynamic cluster labeling and marketing recommendations
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from scipy.optimize import linear_sum_assignment

MIN_ROWS = 10      # K-Means up to K=8 and the silhouette score need a meaningful number of customers
MAX_ROWS = 5000    # keeps the scatter plot and the JSON responses responsive
SILHOUETTE_SAMPLE = 2000  # silhouette is O(n^2): evaluate on a sample for larger datasets


class NoDatasetError(Exception):
    """Raised when a model operation is requested before any CSV has been uploaded."""


def _silhouette(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) < 2:
        return 0.0
    sample = SILHOUETTE_SAMPLE if len(X_scaled) > SILHOUETTE_SAMPLE else None
    return float(silhouette_score(X_scaled, labels, sample_size=sample, random_state=42))


# Normalized header names accepted as a ground-truth segment column (used only for scoring)
LABEL_COLUMNS = ("truesegment", "segment", "truelabel", "label", "class", "target",
                 "groundtruth", "actualsegment", "category")


def _cluster_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Clustering accuracy: cluster ids are arbitrary, so each cluster is matched one-to-one to the
    true segment it overlaps most (Hungarian algorithm), then the share of correctly grouped
    customers is returned. Unmatched clusters/segments count as errors.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    rows, cols = linear_sum_assignment(-cm)
    return float(cm[rows, cols].sum() / len(true_labels))


def _round_or_none(value: Any, digits: int = 1) -> Optional[float]:
    return None if pd.isna(value) else float(round(value, digits))


class CustomerSegmentationModel:
    def __init__(self):
        self.df: pd.DataFrame = pd.DataFrame()
        self.scaler: StandardScaler = StandardScaler()
        self.model: Optional[KMeans] = None
        self.current_k: int = 4
        self.cluster_metadata: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ data loading
    @staticmethod
    def _normalize_column(name: str) -> str:
        """Lowercases a column name and strips everything except letters and digits."""
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def _resolve_columns(self, columns: List[str]) -> Dict[str, Optional[str]]:
        """
        Maps uploaded CSV headers onto the features the model needs. Accepts the headers
        Customer_ID / Age / Annual_Income / Spending_Score as well as the popular Kaggle
        Mall_Customers.csv headers ("Annual Income (k$)", "Spending Score (1-100)").
        """
        resolved: Dict[str, Optional[str]] = {"id": None, "age": None, "income": None, "spending": None, "label": None}
        for col in columns:
            norm = self._normalize_column(col)
            if resolved["id"] is None and norm in ("customerid", "id", "custid"):
                resolved["id"] = col
            elif resolved["age"] is None and norm == "age":
                resolved["age"] = col
            elif resolved["income"] is None and (norm.startswith("annualincome") or norm == "income"):
                resolved["income"] = col
            elif resolved["spending"] is None and (norm.startswith("spendingscore") or norm == "spending"):
                resolved["spending"] = col
            elif resolved["label"] is None and norm in LABEL_COLUMNS:
                resolved["label"] = col
        return resolved

    def load_dataset(self, upload_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates an uploaded customer table and makes it the active dataset:
        1. Detects the Annual Income and Spending Score columns (Customer ID and Age optional).
        2. Drops rows whose features are missing or non-numeric.
        3. Replaces the current dataset and discards the previously trained model.
        Raises ValueError (with a user-facing message) if the file cannot be used.
        """
        cols = self._resolve_columns(list(upload_df.columns))
        missing = [
            label for key, label in (("income", "Annual_Income"), ("spending", "Spending_Score"))
            if cols[key] is None
        ]
        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Found columns: {', '.join(map(str, upload_df.columns))}"
            )

        total_rows = len(upload_df)
        work = pd.DataFrame({
            "Customer_ID": (
                upload_df[cols["id"]].astype(str).to_numpy()
                if cols["id"] else np.arange(1, total_rows + 1).astype(str)
            ),
            "Age": (
                pd.to_numeric(upload_df[cols["age"]], errors="coerce").to_numpy()
                if cols["age"] else np.full(total_rows, np.nan)
            ),
            "Annual_Income": pd.to_numeric(upload_df[cols["income"]], errors="coerce").to_numpy(),
            "Spending_Score": pd.to_numeric(upload_df[cols["spending"]], errors="coerce").to_numpy(),
            # Optional ground-truth segment: never used for clustering, only for scoring
            "True_Segment": (
                upload_df[cols["label"]].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).to_numpy()
                if cols["label"] else np.full(total_rows, np.nan, dtype=object)
            ),
        })
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["Annual_Income", "Spending_Score"])
        work = work.reset_index(drop=True)

        if len(work) < MIN_ROWS:
            raise ValueError(
                f"At least {MIN_ROWS} customers with numeric Annual Income and Spending Score are "
                f"required, but only {len(work)} valid row(s) were found."
            )
        if len(work) > MAX_ROWS:
            raise ValueError(f"Too many rows ({len(work)}). The maximum supported is {MAX_ROWS}.")

        self.df = work
        self.model = None
        self.cluster_metadata = {}

        return {
            "total_rows": total_rows,
            "processed_rows": len(work),
            "skipped_rows": total_rows - len(work),
            "detected_columns": {key: (str(val) if val is not None else None) for key, val in cols.items()},
        }

    def get_features(self) -> np.ndarray:
        """
        Extracts features used for clustering:
        1. Annual_Income
        2. Spending_Score
        """
        if self.df.empty:
            raise NoDatasetError("No dataset uploaded yet.")
        return self.df[["Annual_Income", "Spending_Score"]].values

    # ------------------------------------------------------------------ elbow
    def compute_elbow(self, k_min: int = 2, k_max: int = 8) -> Tuple[List[int], List[float], int]:
        """
        Calculates Within-Cluster Sum of Squares (WCSS / Inertia) for candidate values of K.
        WCSS = sum of squared Euclidean distances between each point and its cluster centroid.
        Also returns the K with the highest silhouette score as the recommended K.
        """
        X = self.get_features()
        X_scaled = StandardScaler().fit_transform(X)

        k_max = min(k_max, len(X) - 1)
        k_values = list(range(k_min, k_max + 1))
        wcss: List[float] = []
        silhouettes: List[float] = []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            wcss.append(float(round(km.inertia_, 2)))
            silhouettes.append(_silhouette(X_scaled, labels))

        recommended_k = k_values[int(np.argmax(silhouettes))]
        return k_values, wcss, recommended_k

    # ------------------------------------------------------------------ training
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
        based on how the cluster's average income and spending compare with the overall averages.
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
        Trains K-Means clustering on the uploaded dataset with the chosen K:
        1. Scales Annual_Income and Spending_Score using StandardScaler.
        2. Fits KMeans(n_clusters=k, random_state=42, n_init=10).
        3. Computes centroids in original units using inverse_transform.
        4. Calculates per-cluster statistics (count, avg age, avg income, avg spending).
        5. Computes silhouette score to evaluate clustering cohesion and separation.
        """
        X = self.get_features()
        if k >= len(X):
            raise ValueError(f"K must be smaller than the number of customers ({len(X)}).")

        # Fit into local objects and publish them only once fully trained, so concurrent
        # prediction requests never see a half-built (unfitted) model.
        # Step 1: Scale features (StandardScaler: z = (x - u) / s)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 2: Fit KMeans
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        self.df["Cluster"] = labels

        # Step 3: Compute Centroids in original feature space
        orig_centroids = scaler.inverse_transform(model.cluster_centers_)

        # Step 4: Overall benchmarks for auto-labeling
        mean_income = float(self.df["Annual_Income"].mean())
        mean_spending = float(self.df["Spending_Score"].mean())
        total_customers = len(self.df)

        # Step 5: Calculate Silhouette Score (ranges from -1 to 1)
        sil_score = float(round(_silhouette(X_scaled, labels), 3))

        # Step 5b: External accuracy, only when the upload carried a ground-truth segment column
        evaluation = self._evaluate_against_labels(labels)

        # Step 6: Build per-cluster summaries
        cluster_metadata: Dict[int, Dict[str, Any]] = {}
        clusters_summary = []
        centroids_summary = []

        for cid in range(k):
            c_df = self.df[self.df["Cluster"] == cid]

            count = int(len(c_df))
            pct = round((count / total_customers) * 100, 1)
            avg_age = _round_or_none(c_df["Age"].mean()) if count > 0 else None
            avg_inc = float(round(c_df["Annual_Income"].mean(), 1)) if count > 0 else 0.0
            avg_sp = float(round(c_df["Spending_Score"].mean(), 1)) if count > 0 else 0.0

            centroid_income = float(round(orig_centroids[cid][0], 1))
            centroid_spending = float(round(orig_centroids[cid][1], 1))

            label, desc, strat = self._generate_cluster_label_and_strategy(
                avg_inc, avg_sp, mean_income, mean_spending, cid, k
            )

            # Store metadata for real-time predictions
            cluster_metadata[cid] = {
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

        # Publish the trained model state together
        self.scaler, self.model, self.cluster_metadata, self.current_k = scaler, model, cluster_metadata, k

        # Format individual customer points for visualization
        points = [
            {
                "id": str(row["Customer_ID"]),
                "age": None if pd.isna(row["Age"]) else int(row["Age"]),
                "income": float(row["Annual_Income"]),
                "spending": float(row["Spending_Score"]),
                "cluster": int(row["Cluster"])
            }
            for _, row in self.df.iterrows()
        ]

        return {
            "k": k,
            "total_customers": total_customers,
            "silhouette_score": sil_score,
            **evaluation,
            "clusters": clusters_summary,
            "points": points,
            "centroids": centroids_summary
        }

    def _evaluate_against_labels(self, labels: np.ndarray) -> Dict[str, Any]:
        """Accuracy, ARI and NMI against the uploaded True_Segment column (None when absent)."""
        empty = {"accuracy": None, "ari": None, "nmi": None, "true_segments": None, "labeled_customers": 0}
        if "True_Segment" not in self.df.columns:
            return empty
        mask = self.df["True_Segment"].notna().to_numpy()
        if mask.sum() < 2:
            return empty
        truth = pd.factorize(self.df.loc[mask, "True_Segment"])[0]
        pred = labels[mask]
        return {
            "accuracy": round(_cluster_accuracy(truth, pred), 4),
            "ari": round(float(adjusted_rand_score(truth, pred)), 4),
            "nmi": round(float(normalized_mutual_info_score(truth, pred)), 4),
            "true_segments": int(len(set(truth))),
            "labeled_customers": int(mask.sum()),
        }

    # ------------------------------------------------------------------ inference
    def predict(self, age: Optional[int], income: float, spending: float) -> Dict[str, Any]:
        """
        Classifies a new customer into one of the trained clusters:
        1. Scales [income, spending] using the fitted StandardScaler.
        2. Assigns nearest cluster using model.predict.
        3. Returns cluster ID, persona label, marketing strategy, and centroid.
        """
        if self.model is None:
            # Dataset uploaded but not clustered yet: fall back to the default K=4
            self.train(k=4)
        scaler, model, metadata = self.scaler, self.model, self.cluster_metadata

        scaled_input = scaler.transform(np.array([[income, spending]]))
        cluster_id = int(model.predict(scaled_input)[0])
        meta = metadata.get(cluster_id, {
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
            "spending_score": float(spending),
            "age": age,
            "nearest_centroid": meta["centroid"]
        }


# Global singleton instance for FastAPI backend (starts empty until a CSV is uploaded)
ml_model = CustomerSegmentationModel()
