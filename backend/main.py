import io
from typing import Optional
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
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
    UploadResponse,
    HealthResponse,
)

app = FastAPI(
    title="Customer Segmentation ML API",
    description="FastAPI service for K-Means Customer Clustering on an uploaded customer dataset.",
    version="2.0.0",
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

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


def require_dataset():
    """Dependency: every ML endpoint needs a dataset that was uploaded first."""
    if ml_model.df.empty:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No dataset uploaded yet. Upload a customer CSV file first."
        )


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint returning system status and the number of uploaded customers.
    """
    return HealthResponse(
        status="healthy",
        message="Customer Segmentation ML API is running smoothly.",
        customers_loaded=len(ml_model.df),
    )


@app.post("/api/upload", response_model=UploadResponse, tags=["Dataset"])
async def upload_dataset(request: Request):
    """
    Uploads a customer CSV file and makes it the active dataset (replacing any previous one).
    The request body is the raw CSV text (Content-Type: text/csv). Required columns are
    Age and Repeat_Purchases; Name, Gender, District and Product_Type are optional.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File is too large. Maximum upload size is 5 MB."
        )

    try:
        upload_df = pd.read_csv(io.StringIO(body.decode("utf-8-sig")))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read the file as CSV: {str(e)}"
        )

    try:
        summary = await run_in_threadpool(ml_model.load_dataset, upload_df)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return UploadResponse(**summary)


@app.get("/api/dataset", response_model=DatasetResponse, tags=["Dataset"], dependencies=[Depends(require_dataset)])
def get_dataset(limit: Optional[int] = Query(None, ge=1, le=10000, description="Optional maximum number of records")):
    """
    Returns the uploaded customer records (with their cluster once K-Means has been run).
    Optionally limited by query parameter ?limit=
    """
    df = ml_model.df
    has_cluster = "Cluster" in df.columns

    records = [
        Customer(
            customer_id=str(row["Customer_ID"]),
            name=str(row["Name"]),
            age=float(row["Age"]),
            gender=row["Gender"],
            district=row["District"],
            product_type=row["Product_Type"],
            repeat_purchases=float(row["Repeat_Purchases"]),
            cluster=int(row["Cluster"]) if has_cluster and pd.notna(row["Cluster"]) else None,
        )
        for _, row in df.iterrows()
    ]

    if limit is not None:
        records = records[:limit]

    return DatasetResponse(total_customers=len(df), customers=records)


@app.get("/api/elbow", response_model=ElbowResponse, tags=["Clustering"], dependencies=[Depends(require_dataset)])
def get_elbow_curve(
    k_min: int = Query(2, ge=2, le=5, description="Minimum K value"),
    k_max: int = Query(8, ge=5, le=10, description="Maximum K value")
):
    """
    Computes Within-Cluster Sum of Squares (WCSS / Inertia) across K candidate values (2 to 8)
    on the uploaded data to demonstrate the Elbow Method, plus the silhouette-optimal K.
    """
    try:
        k_values, wcss, recommended_k = ml_model.compute_elbow(k_min=k_min, k_max=k_max)
        return ElbowResponse(k_values=k_values, wcss=wcss, recommended_k=recommended_k)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate elbow curve: {str(e)}"
        )


@app.post("/api/cluster", response_model=ClusterResponse, tags=["Clustering"], dependencies=[Depends(require_dataset)])
def run_kmeans_clustering(payload: ClusterRequest):
    """
    Trains K-Means clustering on the uploaded dataset with the user-selected K (2 <= K <= 8).
    Returns silhouette score, cluster statistics, marketing personas, centroids, and points.
    """
    try:
        result = ml_model.train(k=payload.k)
        return ClusterResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"K-Means training failed: {str(e)}"
        )


@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"], dependencies=[Depends(require_dataset)])
def predict_customer_segment(payload: PredictRequest):
    """
    Predicts the market segment for a single customer given their Age and Repeat_Purchases.
    Uses the currently fitted StandardScaler and KMeans model.
    """
    try:
        prediction = ml_model.predict(
            age=payload.age,
            repeats=payload.repeat_purchases
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
