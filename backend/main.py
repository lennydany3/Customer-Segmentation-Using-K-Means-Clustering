from typing import Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from model import ml_model
from schemas import (
    Customer,
    DatasetResponse,
    ElbowResponse,
    ClusterRequest,
    ClusterResponse,
    PredictRequest,
    PredictResponse,
    HealthResponse,
)

app = FastAPI(
    title="Customer Segmentation ML API",
    description="FastAPI service for K-Means Customer Clustering on synthetic customer behavioral dataset.",
    version="1.0.0",
)

# Enable CORS for Next.js frontend running locally
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """
    On server startup, ensure dataset is loaded and initialize default model with K=4.
    """
    try:
        ml_model.load_data()
        ml_model.train(k=4)
        print("Backend initialized: Dataset loaded and baseline K=4 model trained.")
    except Exception as e:
        print(f"Error during startup initialization: {e}")

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint returning system status and customer count.
    """
    return HealthResponse(
        status="healthy",
        message="Customer Segmentation ML API is running smoothly.",
        customers_loaded=len(ml_model.df),
    )

@app.get("/api/dataset", response_model=DatasetResponse, tags=["Dataset"])
def get_dataset(limit: Optional[int] = Query(None, ge=1, le=1000, description="Optional maximum number of records")):
    """
    Returns the synthetic customer dataset records.
    Optionally paginated / limited by query parameter ?limit=
    """
    df = ml_model.df
    if df.empty:
        df = ml_model.load_data()

    records = []
    has_cluster = "Cluster" in df.columns

    for _, row in df.iterrows():
        records.append(
            Customer(
                customer_id=str(row["Customer_ID"]),
                age=int(row["Age"]),
                annual_income=float(row["Annual_Income"]),
                spending_score=int(row["Spending_Score"]),
                cluster=int(row["Cluster"]) if has_cluster else None,
            )
        )

    if limit is not None:
        records = records[:limit]

    return DatasetResponse(
        total_customers=len(df),
        customers=records,
    )

@app.get("/api/elbow", response_model=ElbowResponse, tags=["Clustering"])
def get_elbow_curve(
    k_min: int = Query(2, ge=2, le=5, description="Minimum K value"),
    k_max: int = Query(8, ge=5, le=10, description="Maximum K value")
):
    """
    Computes Within-Cluster Sum of Squares (WCSS / Inertia) across K candidate values (2 to 8)
    to visually demonstrate the Elbow Method for optimal cluster selection.
    """
    try:
        k_values, wcss = ml_model.compute_elbow(k_min=k_min, k_max=k_max)
        return ElbowResponse(k_values=k_values, wcss=wcss)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate elbow curve: {str(e)}"
        )

@app.post("/api/cluster", response_model=ClusterResponse, tags=["Clustering"])
def run_kmeans_clustering(payload: ClusterRequest):
    """
    Trains K-Means clustering on the dataset with the user-selected K (2 <= K <= 8).
    Returns silhouette score, cluster statistics, marketing personas, centroids, and points.
    """
    if payload.k < 2 or payload.k > 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter K must be between 2 and 8."
        )

    try:
        result = ml_model.train(k=payload.k)
        return ClusterResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"K-Means training failed: {str(e)}"
        )

@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_customer_segment(payload: PredictRequest):
    """
    Predicts the market segment for a single customer given their Age, Annual Income, and Spending Score.
    Uses the currently fitted StandardScaler and KMeans model.
    """
    try:
        prediction = ml_model.predict(
            age=payload.age,
            income=payload.income,
            spending=payload.spending_score
        )
        return PredictResponse(**prediction)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

