import numpy as np
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def build_csv(n_per_group: int = 25) -> str:
    """Builds an upload with four well-separated customer groups (Kaggle-style headers)."""
    rng = np.random.default_rng(0)
    rows = ["CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)"]
    groups = [(95, 82), (90, 20), (28, 78), (28, 22)]
    n = 1
    for income, spending in groups:
        for _ in range(n_per_group):
            rows.append(
                f"{n},Male,{rng.integers(20, 60)},"
                f"{rng.normal(income, 6):.1f},{np.clip(rng.normal(spending, 6), 1, 100):.0f}"
            )
            n += 1
    rows.append(f"{n},Female,,abc,40")  # invalid row that must be skipped
    return "\n".join(rows) + "\n"


def post_csv(text: str):
    return client.post("/api/upload", content=text, headers={"Content-Type": "text/csv"})


def test_all_endpoints():
    print("Testing endpoints before any upload (must be 409)...")
    assert client.get("/api/dataset").status_code == 409
    assert client.get("/api/elbow").status_code == 409
    assert client.post("/api/cluster", json={"k": 4}).status_code == 409
    assert client.post("/api/predict", json={"income": 90, "spending_score": 20}).status_code == 409

    print("\nTesting GET /api/health...")
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    assert r.json()["customers_loaded"] == 0

    print("\nTesting POST /api/upload...")
    r = post_csv(build_csv())
    assert r.status_code == 200, f"Upload failed: {r.text}"
    upload = r.json()
    assert upload["total_rows"] == 101
    assert upload["processed_rows"] == 100
    assert upload["skipped_rows"] == 1
    assert upload["detected_columns"]["income"] == "Annual Income (k$)"
    print("Upload summary:", upload)

    print("\nTesting GET /api/dataset...")
    r = client.get("/api/dataset?limit=5")
    assert r.status_code == 200, f"Dataset check failed: {r.text}"
    data = r.json()
    assert data["total_customers"] == 100
    assert len(data["customers"]) == 5

    print("\nTesting GET /api/elbow...")
    r = client.get("/api/elbow")
    assert r.status_code == 200, f"Elbow check failed: {r.text}"
    elbow = r.json()
    assert len(elbow["k_values"]) == len(elbow["wcss"])
    assert elbow["recommended_k"] == 4
    print("Elbow data:", elbow)

    print("\nTesting POST /api/cluster with k=4...")
    r = client.post("/api/cluster", json={"k": 4})
    assert r.status_code == 200, f"Cluster k=4 failed: {r.text}"
    cluster_res = r.json()
    assert cluster_res["k"] == 4
    assert len(cluster_res["clusters"]) == 4
    assert len(cluster_res["points"]) == 100
    assert sum(c["customer_count"] for c in cluster_res["clusters"]) == 100

    print("\nTesting POST /api/cluster validation error (k=10)...")
    r = client.post("/api/cluster", json={"k": 10})
    assert r.status_code in (400, 422), f"Validation failed to catch k=10: {r.status_code}"

    print("\nTesting POST /api/predict...")
    r = client.post("/api/predict", json={"age": 30, "income": 95.0, "spending_score": 85})
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred = r.json()
    assert pred["label"] == "Premium VIP Shoppers"

    print("\nTesting invalid uploads...")
    r = post_csv("name,city\nA,B\n")
    assert r.status_code == 400 and "Missing required column" in r.json()["detail"]
    r = post_csv("Annual_Income,Spending_Score\n1,2\n3,4\n")
    assert r.status_code == 400 and "At least" in r.json()["detail"]
    assert client.get("/api/dataset").json()["total_customers"] == 100  # previous dataset untouched

    print("\nAll FastAPI tests passed successfully!")


if __name__ == "__main__":
    test_all_endpoints()
