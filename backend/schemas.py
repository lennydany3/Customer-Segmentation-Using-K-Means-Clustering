from typing import List, Optional
from pydantic import BaseModel, Field

# --- Dataset Schemas ---
class Customer(BaseModel):
    customer_id: str = Field(..., description="Unique customer ID (e.g. C001)")
    age: int = Field(..., description="Customer age")
    annual_income: float = Field(..., description="Annual Income in $k")
    spending_score: int = Field(..., description="Spending Score between 1 and 100")
    cluster: Optional[int] = Field(None, description="Assigned cluster ID if clustered")

class DatasetResponse(BaseModel):
    total_customers: int
    customers: List[Customer]

# --- Elbow Method Schemas ---
class ElbowResponse(BaseModel):
    k_values: List[int] = Field(..., description="List of K candidate values (e.g., 2 to 8)")
    wcss: List[float] = Field(..., description="Within-Cluster Sum of Squares (Inertia) for each K")

# --- Clustering Schemas ---
class ClusterRequest(BaseModel):
    k: int = Field(4, ge=2, le=8, description="Number of clusters K (between 2 and 8)")

class ClusterSummary(BaseModel):
    cluster_id: int
    label: str
    customer_count: int
    percentage: float
    avg_age: float
    avg_income: float
    avg_spending: float
    centroid: List[float] = Field(..., description="Centroid coordinates [income, spending] in original units")
    description: str
    strategy: str

class Point(BaseModel):
    id: str
    age: int
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
    age: int = Field(..., ge=18, le=100, description="Customer age (18 - 100)")
    income: float = Field(..., ge=10.0, le=250.0, description="Annual Income in $k (10 - 250)")
    spending_score: int = Field(..., ge=1, le=100, description="Spending score (1 - 100)")

class PredictResponse(BaseModel):
    cluster_id: int
    label: str
    description: str
    strategy: str
    income: float
    spending_score: int
    age: int
    nearest_centroid: List[float]

# --- Health Check ---
class HealthResponse(BaseModel):
    status: str
    message: str
    customers_loaded: int

