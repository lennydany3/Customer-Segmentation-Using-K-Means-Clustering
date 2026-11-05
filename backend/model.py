"""
model.py: Machine Learning logic for Customer Segmentation using K-Means.
- Data: an uploaded gift-shop customer CSV (nothing is bundled with the backend)
  Columns: Name, Age, Gender, District, Product_Type, Repeat_Purchases (+ optional True_Segment)
- Clustering features: Age and Repeat_Purchases
- Normalization: StandardScaler (zero mean, unit variance)
- Model: KMeans clustering with scikit-learn (k-means++, 10 restarts)
- Metrics: Inertia (WCSS for the Elbow method), Silhouette Score and, when the upload has a
  True_Segment column, accuracy / ARI / NMI against it
- Interpretation: persona, top product, top district and marketing strategy per cluster
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 10      # K-Means up to K=8 and the silhouette score need a meaningful number of customers
MAX_ROWS = 5000    # keeps the scatter plot and the JSON responses responsive
SILHOUETTE_SAMPLE = 2000  # silhouette is O(n^2): evaluate on a sample for larger datasets
FEATURES = ["Age", "Repeat_Purchases"]

# Normalized header names (lowercase, letters and digits only) accepted for each field
COLUMN_ALIASES = {
    "name": ("name", "customername", "customer"),
    "age": ("age",),
    "gender": ("gender", "genre", "sex"),
    "district": ("district", "city", "location"),
    "product": ("producttype", "product", "products"),
    "repeats": ("repeatpurchases", "repeats", "noofrepeats", "numberofrepeats", "repeatcount", "repeat"),
    "label": ("truesegment", "segment", "truelabel", "label", "class", "target", "groundtruth", "actualsegment"),
}


class NoDatasetError(Exception):
    """Raised when a model operation is requested before any CSV has been uploaded."""


def _silhouette(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) < 2:
        return 0.0
    sample = SILHOUETTE_SAMPLE if len(X_scaled) > SILHOUETTE_SAMPLE else None
    return float(silhouette_score(X_scaled, labels, sample_size=sample, random_state=42))


def _cluster_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Clustering accuracy: cluster ids are arbitrary, so each cluster is matched one-to-one to the
    true segment it overlaps most (Hungarian algorithm), then the share of correctly grouped
    customers is returned. Unmatched clusters/segments count as errors.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    rows, cols = linear_sum_assignment(-cm)
    return float(cm[rows, cols].sum() / len(true_labels))


def _text_column(upload_df: pd.DataFrame, col: Optional[str], n: int) -> np.ndarray:
    if col is None:
        return np.full(n, None, dtype=object)
    values = upload_df[col].astype(str).str.strip()
    return values.where(~values.isin(["", "nan", "None"]), None).to_numpy(dtype=object)


def _mode(series: pd.Series) -> Tuple[Optional[str], float]:
    s = series.dropna()
    if s.empty:
        return None, 0.0
    counts = s.value_counts()
    return str(counts.index[0]), float(round(counts.iloc[0] / len(s) * 100, 1))


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
        """Maps uploaded CSV headers onto the fields the app understands (first match wins)."""
        resolved: Dict[str, Optional[str]] = {key: None for key in COLUMN_ALIASES}
        for col in columns:
            norm = self._normalize_column(col)
            for key, aliases in COLUMN_ALIASES.items():
                if resolved[key] is None and norm in aliases:
                    resolved[key] = col
                    break
        return resolved

    def load_dataset(self, upload_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates an uploaded customer table and makes it the active dataset:
        1. Detects the Age and Repeat_Purchases columns (Name, Gender, District, Product_Type optional).
        2. Drops rows whose Age or Repeat_Purchases is missing, non-numeric or negative.
        3. Replaces the current dataset and discards the previously trained model.
        Raises ValueError (with a user-facing message) if the file cannot be used.
        """
        cols = self._resolve_columns(list(upload_df.columns))
        missing = [label for key, label in (("age", "Age"), ("repeats", "Repeat_Purchases")) if cols[key] is None]
        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing)}. "
                f"Found columns: {', '.join(map(str, upload_df.columns))}"
            )

        total_rows = len(upload_df)
        names = _text_column(upload_df, cols["name"], total_rows)
        work = pd.DataFrame({
            "Customer_ID": np.arange(1, total_rows + 1).astype(str),
            "Name": [n if n else f"Customer {i + 1}" for i, n in enumerate(names)],
            "Age": pd.to_numeric(upload_df[cols["age"]], errors="coerce").to_numpy(),
            "Gender": _text_column(upload_df, cols["gender"], total_rows),
            "District": _text_column(upload_df, cols["district"], total_rows),
            "Product_Type": _text_column(upload_df, cols["product"], total_rows),
            "Repeat_Purchases": pd.to_numeric(upload_df[cols["repeats"]], errors="coerce").to_numpy(),
            # Optional ground-truth segment: never used for clustering, only for scoring
            "True_Segment": _text_column(upload_df, cols["label"], total_rows),
        })
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
        work = work[(work["Age"] >= 0) & (work["Repeat_Purchases"] >= 0)].reset_index(drop=True)

        if len(work) < MIN_ROWS:
            raise ValueError(
                f"At least {MIN_ROWS} customers with numeric Age and Repeat_Purchases are "
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
        """Extracts the clustering features: Age and Repeat_Purchases."""
        if self.df.empty:
            raise NoDatasetError("No dataset uploaded yet.")
        return self.df[FEATURES].values.astype(float)

    # ------------------------------------------------------------------ elbow
    def compute_elbow(self, k_min: int = 2, k_max: int = 8) -> Tuple[List[int], List[float], int]:
        """
        Calculates Within-Cluster Sum of Squares (WCSS / Inertia) for candidate values of K and
        returns the K with the highest silhouette score as the recommended K.
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
    @staticmethod
    def _generate_cluster_label_and_strategy(
        avg_age: float, avg_repeats: float, age_stats: Tuple[float, float], repeat_stats: Tuple[float, float],
        top_product: Optional[str],
    ) -> Tuple[str, str, str]:
        """
        Derives a persona, description and strategy from the cluster's average age and repeat
        purchases. Bands are relative to the uploaded data: more than half a standard deviation
        below / above the mean is young / senior and occasional / loyal.
        """
        a_mu, a_sd = age_stats
        r_mu, r_sd = repeat_stats
        age_band = "young" if avg_age < a_mu - 0.5 * a_sd else ("senior" if avg_age > a_mu + 0.5 * a_sd else "adult")
        habit = "occasional" if avg_repeats < r_mu - 0.5 * r_sd else ("loyal" if avg_repeats > r_mu + 0.5 * r_sd else "regular")
        product = top_product or "our products"

        labels = {
            ("young", "loyal"): "Loyal Young Fans",
            ("young", "regular"): "Regular Young Buyers",
            ("young", "occasional"): "Occasional Young Buyers",
            ("adult", "loyal"): "Loyal Adult Customers",
            ("adult", "regular"): "Regular Adult Customers",
            ("adult", "occasional"): "Occasional Adult Buyers",
            ("senior", "loyal"): "Loyal Senior Gifters",
            ("senior", "regular"): "Regular Senior Customers",
            ("senior", "occasional"): "Occasional Senior Buyers",
        }
        who = {"young": "Young customers", "adult": "Working-age adults", "senior": "Older customers"}[age_band]
        how = {
            "loyal": "who come back again and again. The most engaged buyers.",
            "regular": "who return at a steady, moderate rate.",
            "occasional": "who have bought only a few times, usually for an occasion.",
        }[habit]
        strategy = {
            "loyal": f"Reward them with a loyalty card and referral bonus, and show new {product} designs to them first.",
            "regular": f"Turn them into loyal fans with festival combos, birthday reminders and member prices on {product}.",
            "occasional": f"Win a second visit with occasion reminders, first-repeat discounts and easy delivery of {product}.",
        }[habit]
        if age_band == "young" and habit != "loyal":
            strategy += " Instagram and WhatsApp offers work best for this age group."
        if age_band == "senior":
            strategy += " Reach them with phone calls, SMS and temple-festival or wedding season offers."
        return labels[(age_band, habit)], f"{who} {how}", strategy

    def _evaluate_against_labels(self, labels: np.ndarray) -> Dict[str, Any]:
        """Accuracy, ARI and NMI against the uploaded True_Segment column (None when absent)."""
        empty = {"accuracy": None, "ari": None, "nmi": None, "true_segments": None, "labeled_customers": 0}
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

    def train(self, k: int = 4) -> Dict[str, Any]:
        """
        Trains K-Means on the uploaded dataset with the chosen K:
        1. Scales Age and Repeat_Purchases using StandardScaler.
        2. Fits KMeans(n_clusters=k, random_state=42, n_init=10).
        3. Computes centroids in original units using inverse_transform.
        4. Builds per-cluster statistics, persona, top product and top district.
        5. Computes the silhouette score and, if available, accuracy against True_Segment.
        """
        X = self.get_features()
        if k >= len(X):
            raise ValueError(f"K must be smaller than the number of customers ({len(X)}).")

        # Fit into local objects and publish them only once fully trained, so concurrent
        # prediction requests never see a half-built (unfitted) model.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        self.df["Cluster"] = labels
        orig_centroids = scaler.inverse_transform(model.cluster_centers_)

        age_stats = (float(self.df["Age"].mean()), float(self.df["Age"].std(ddof=0)))
        repeat_stats = (float(self.df["Repeat_Purchases"].mean()), float(self.df["Repeat_Purchases"].std(ddof=0)))
        total_customers = len(self.df)
        sil_score = float(round(_silhouette(X_scaled, labels), 3))
        evaluation = self._evaluate_against_labels(labels)

        cluster_metadata: Dict[int, Dict[str, Any]] = {}
        clusters_summary = []
        centroids_summary = []

        for cid in range(k):
            c_df = self.df[self.df["Cluster"] == cid]
            count = int(len(c_df))
            avg_age = float(round(c_df["Age"].mean(), 1)) if count else 0.0
            avg_rep = float(round(c_df["Repeat_Purchases"].mean(), 1)) if count else 0.0
            centroid = [float(round(orig_centroids[cid][0], 1)), float(round(orig_centroids[cid][1], 1))]
            top_product, product_share = _mode(c_df["Product_Type"])
            top_district, _ = _mode(c_df["District"])
            female_share = (
                float(round((c_df["Gender"].str.lower() == "female").mean() * 100, 1))
                if c_df["Gender"].notna().any() else None
            )

            label, desc, strat = self._generate_cluster_label_and_strategy(
                avg_age, avg_rep, age_stats, repeat_stats, top_product
            )
            cluster_metadata[cid] = {"label": label, "description": desc, "strategy": strat, "centroid": centroid}

            clusters_summary.append({
                "cluster_id": cid,
                "label": label,
                "customer_count": count,
                "percentage": round((count / total_customers) * 100, 1),
                "avg_age": avg_age,
                "avg_repeats": avg_rep,
                "centroid": centroid,
                "top_product": top_product,
                "top_product_share": product_share,
                "top_district": top_district,
                "female_share": female_share,
                "description": desc,
                "strategy": strat,
            })
            centroids_summary.append({"cluster": cid, "age": centroid[0], "repeats": centroid[1], "label": label})

        # Two clusters can share a persona: number them so every label is unique on screen
        names = [c["label"] for c in clusters_summary]
        for c in clusters_summary:
            if names.count(c["label"]) > 1:
                c["label"] = f"{c['label']} (Group {c['cluster_id'] + 1})"
        for c in clusters_summary:
            cluster_metadata[c["cluster_id"]]["label"] = c["label"]
        for c in centroids_summary:
            c["label"] = cluster_metadata[c["cluster"]]["label"]

        # Publish the trained model state together
        self.scaler, self.model, self.cluster_metadata, self.current_k = scaler, model, cluster_metadata, k

        points = [
            {
                "id": str(row["Name"]),
                "age": float(row["Age"]),
                "repeats": float(row["Repeat_Purchases"]),
                "product": row["Product_Type"],
                "cluster": int(row["Cluster"]),
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
            "centroids": centroids_summary,
        }

    # ------------------------------------------------------------------ inference
    def predict(self, age: float, repeats: float) -> Dict[str, Any]:
        """
        Classifies a new customer into one of the trained clusters:
        1. Scales [age, repeat_purchases] using the fitted StandardScaler.
        2. Assigns the nearest cluster centroid using model.predict.
        """
        if self.model is None:
            # Dataset uploaded but not clustered yet: fall back to the default K=4
            self.train(k=min(4, len(self.df) - 1))
        scaler, model, metadata = self.scaler, self.model, self.cluster_metadata

        scaled_input = scaler.transform(np.array([[age, repeats]], dtype=float))
        cluster_id = int(model.predict(scaled_input)[0])
        meta = metadata[cluster_id]

        return {
            "cluster_id": cluster_id,
            "label": meta["label"],
            "description": meta["description"],
            "strategy": meta["strategy"],
            "age": float(age),
            "repeat_purchases": float(repeats),
            "nearest_centroid": meta["centroid"],
        }


# Global singleton instance for FastAPI backend (starts empty until a CSV is uploaded)
ml_model = CustomerSegmentationModel()
