import sys
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_all_endpoints():
    print("Testing GET /api/health...")
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("Health response:", r.json())

    print("\nTesting GET /api/dataset...")
    r = client.get("/api/dataset?limit=5")
    assert r.status_code == 200, f"Dataset check failed: {r.text}"
    data = r.json()
    assert data["total_customers"] == 200
    assert len(data["customers"]) == 5
    print("Dataset 5 samples:", data["customers"])

    print("\nTesting GET /api/elbow...")
    r = client.get("/api/elbow")
    assert r.status_code == 200, f"Elbow check failed: {r.text}"
    elbow = r.json()
    assert "k_values" in elbow and "wcss" in elbow
    assert len(elbow["k_values"]) == len(elbow["wcss"])
    print("Elbow data:", elbow)

    print("\nTesting POST /api/cluster with k=4...")
    r = client.post("/api/cluster", json={"k": 4})
    assert r.status_code == 200, f"Cluster k=4 failed: {r.text}"
    cluster_res = r.json()
    assert cluster_res["k"] == 4
    assert len(cluster_res["clusters"]) == 4
    assert len(cluster_res["centroids"]) == 4
    assert len(cluster_res["points"]) == 200
    print(f"Clustering silhouette: {cluster_res['silhouette_score']}, clusters count: {len(cluster_res['clusters'])}")

    print("\nTesting POST /api/cluster validation error (k=10)...")
    r = client.post("/api/cluster", json={"k": 10})
    assert r.status_code == 422 or r.status_code == 400, f"Validation failed to catch k=10: {r.status_code}"
    print("Validation error caught successfully.")

    print("\nTesting POST /api/predict...")
    r = client.post("/api/predict", json={"age": 30, "income": 95.0, "spending_score": 85})
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred = r.json()
    assert "cluster_id" in pred
    assert "label" in pred
    assert "strategy" in pred
    print("Prediction response:", pred)

    print("\nAll FastAPI tests passed successfully!")

if __name__ == "__main__":
    test_all_endpoints()

