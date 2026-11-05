from typing import List, Optional
from pydantic import BaseModel, Field

# --- Upload / Dataset Schemas ---
class DetectedColumns(BaseModel):
    id: Optional[str] = None
    age: Optional[str] = None
    income: Optional[str] = None
    spending: Optional[str] = None

class UploadResponse(BaseModel):
    total_rows: int
    processed_rows: int
    skipped_rows: int
    detected_columns: DetectedColumns

class Customer(BaseModel):
    customer_id: str = Field(..., description="Customer ID from the uploaded file (or the row number)")
    age: Optional[int] = Field(None, description="Customer age, if the file has an Age column")
    annual_income: float = Field(..., description="Annual Income")
    spending_score: float = Field(..., description="Spending Score")
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
    avg_age: Optional[float] = None
    avg_income: float
    avg_spending: float
    centroid: List[float] = Field(..., description="Centroid coordinates [income, spending] in original units")
    description: str
    strategy: str

class Point(BaseModel):
    id: str
    age: Optional[int] = None
    income: float
    spending: float
    cluster: int

class CentroidPoint(BaseModel):
    cluster: int
    income: float
    spending: float
    label: str

class ClusterResponse(BaseModel):
    k: int
    total_customers: int
    silhouette_score: float
    clusters: List[ClusterSummary]
    points: List[Point]
    centroids: List[CentroidPoint]

# --- Prediction Schemas ---
class PredictRequest(BaseModel):
    age: Optional[int] = Field(None, ge=0, le=120, description="Customer age (optional, not used for clustering)")
    income: float = Field(..., description="Annual Income, in the same unit as the uploaded dataset")
    spending_score: float = Field(..., description="Spending score, in the same unit as the uploaded dataset")

class PredictResponse(BaseModel):
    cluster_id: int
    label: str
    description: str
    strategy: str
    income: float
    spending_score: float
    age: Optional[int] = None
    nearest_centroid: List[float]

# --- Health Check ---
class HealthResponse(BaseModel):
    status: str
    message: str
    customers_loaded: int
