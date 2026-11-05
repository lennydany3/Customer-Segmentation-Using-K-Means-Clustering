from typing import List, Optional
from pydantic import BaseModel, Field

# --- Upload / Dataset Schemas ---
class DetectedColumns(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    district: Optional[str] = None
    product: Optional[str] = None
    repeats: Optional[str] = None
    label: Optional[str] = None

class UploadResponse(BaseModel):
    total_rows: int
    processed_rows: int
    skipped_rows: int
    detected_columns: DetectedColumns

class Customer(BaseModel):
    customer_id: str = Field(..., description="Row number in the uploaded file")
    name: str = Field(..., description="Customer name (or 'Customer N' if the file has none)")
    age: float = Field(..., description="Customer age")
    gender: Optional[str] = None
    district: Optional[str] = None
    product_type: Optional[str] = None
    repeat_purchases: float = Field(..., description="Number of repeat purchases")
    cluster: Optional[int] = Field(None, description="Assigned cluster ID if clustered")

class DatasetResponse(BaseModel):
    total_customers: int
    customers: List[Customer]

# --- Elbow Method Schemas ---
class ElbowResponse(BaseModel):
    k_values: List[int] = Field(..., description="List of K candidate values (e.g., 2 to 8)")
    wcss: List[float] = Field(..., description="Within-Cluster Sum of Squares (Inertia) for each K")
    recommended_k: int = Field(..., description="K with the highest silhouette score")

# --- Clustering Schemas ---
class ClusterRequest(BaseModel):
    k: int = Field(4, ge=2, le=8, description="Number of clusters K (between 2 and 8)")

class ClusterSummary(BaseModel):
    cluster_id: int
    label: str
    customer_count: int
    percentage: float
    avg_age: float
    avg_repeats: float
    centroid: List[float] = Field(..., description="Centroid [age, repeat_purchases] in original units")
    top_product: Optional[str] = None
    top_product_share: float = 0.0
    top_district: Optional[str] = None
    female_share: Optional[float] = None
    description: str
    strategy: str

class Point(BaseModel):
    id: str
    age: float
    repeats: float
    product: Optional[str] = None
    cluster: int

class CentroidPoint(BaseModel):
    cluster: int
    age: float
    repeats: float
    label: str

class ClusterResponse(BaseModel):
    k: int
    total_customers: int
    silhouette_score: float
    accuracy: Optional[float] = Field(None, description="Clustering accuracy vs the uploaded true segment column (0 to 1)")
    ari: Optional[float] = Field(None, description="Adjusted Rand Index vs the true segments")
    nmi: Optional[float] = Field(None, description="Normalized Mutual Information vs the true segments")
    true_segments: Optional[int] = Field(None, description="Number of distinct true segments in the upload")
    labeled_customers: int = 0
    clusters: List[ClusterSummary]
    points: List[Point]
    centroids: List[CentroidPoint]

# --- Prediction Schemas ---
class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Customer age")
    repeat_purchases: float = Field(..., ge=0, description="Number of repeat purchases")

class PredictResponse(BaseModel):
    cluster_id: int
    label: str
    description: str
    strategy: str
    age: float
    repeat_purchases: float
    nearest_centroid: List[float]

# --- Health Check ---
class HealthResponse(BaseModel):
    status: str
    message: str
    customers_loaded: int
