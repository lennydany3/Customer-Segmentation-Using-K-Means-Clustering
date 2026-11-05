from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# (mean age, mean repeat purchases, product, true segment)
GROUPS = [(21, 15, "Key Chains", "Loyal Young"), (23, 3, "Key Chains", "Occasional Young"),
          (55, 15, "Bouquet", "Loyal Senior"), (56, 3, "Bouquet", "Occasional Senior")]


def build_csv(n_per_group: int = 25, with_label: bool = False) -> str:
    """Builds an upload with four well-separated customer groups plus one invalid row."""
    rng = np.random.default_rng(0)
    header = "Name,Age,Gender,District,Product_Type,Repeat_Purchases" + (",True_Segment" if with_label else "")
    rows = [header]
    n = 1
    for age, repeats, product, segment in GROUPS:
        for _ in range(n_per_group):
            row = (f"Customer {n},{rng.normal(age, 2):.0f},{rng.choice(['Female', 'Male'])},Madurai,"
                   f"{product},{max(0, rng.normal(repeats, 1.5)):.0f}")
            rows.append(row + (f",{segment}" if with_label else ""))
            n += 1
    rows.append("Broken,abc,Male,Salem,Bouquet,4" + (",Loyal Young" if with_label else ""))  # must be skipped
    return "\n".join(rows) + "\n"


def post_csv(text: str):
    return client.post("/api/upload", content=text, headers={"Content-Type": "text/csv"})


def test_all_endpoints():
    print("Testing endpoints before any upload (must be 409)...")
    assert client.get("/api/dataset").status_code == 409
    assert client.get("/api/elbow").status_code == 409
    assert client.post("/api/cluster", json={"k": 4}).status_code == 409
    assert client.post("/api/predict", json={"age": 22, "repeat_purchases": 15}).status_code == 409

    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["customers_loaded"] == 0

    print("\nTesting POST /api/upload...")
    r = post_csv(build_csv())
    assert r.status_code == 200, f"Upload failed: {r.text}"
    upload = r.json()
    assert upload["total_rows"] == 101
    assert upload["processed_rows"] == 100
    assert upload["skipped_rows"] == 1
    assert upload["detected_columns"]["repeats"] == "Repeat_Purchases"
    assert upload["detected_columns"]["district"] == "District"
    assert upload["detected_columns"]["label"] is None

    r = client.get("/api/dataset?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert data["total_customers"] == 100 and len(data["customers"]) == 5
    assert data["customers"][0]["district"] == "Madurai"

    r = client.get("/api/elbow")
    assert r.status_code == 200
    elbow = r.json()
    assert len(elbow["k_values"]) == len(elbow["wcss"])
    assert elbow["recommended_k"] == 4

    r = client.post("/api/cluster", json={"k": 4})
    assert r.status_code == 200, f"Cluster k=4 failed: {r.text}"
    res = r.json()
    assert res["k"] == 4 and len(res["clusters"]) == 4 and len(res["points"]) == 100
    assert sum(c["customer_count"] for c in res["clusters"]) == 100
    assert res["accuracy"] is None  # no True_Segment column
    labels = {c["label"] for c in res["clusters"]}
    assert labels == {"Loyal Young Fans", "Loyal Senior Gifters", "Occasional Young Buyers", "Occasional Senior Buyers"}
    products = {c["top_product"] for c in res["clusters"]}
    assert products == {"Key Chains", "Bouquet"}

    r = client.post("/api/cluster", json={"k": 10})
    assert r.status_code in (400, 422)

    r = client.post("/api/predict", json={"age": 20, "repeat_purchases": 16})
    assert r.status_code == 200, r.text
    assert r.json()["label"] == "Loyal Young Fans"
    r = client.post("/api/predict", json={"age": 58, "repeat_purchases": 2})
    assert r.json()["label"] == "Occasional Senior Buyers"

    print("\nTesting invalid uploads...")
    r = post_csv("Name,District\nA,Salem\n")
    assert r.status_code == 400 and "Missing required column" in r.json()["detail"]
    r = post_csv("Age,Repeat_Purchases\n1,2\n3,4\n")
    assert r.status_code == 400 and "At least" in r.json()["detail"]
    assert client.get("/api/dataset").json()["total_customers"] == 100  # previous dataset untouched


def test_accuracy_with_true_segment_column():
    """Uploads with a True_Segment column get accuracy/ARI/NMI."""
    r = post_csv(build_csv(with_label=True))
    assert r.status_code == 200, r.text
    assert r.json()["detected_columns"]["label"] == "True_Segment"
    res = client.post("/api/cluster", json={"k": 4}).json()
    assert res["accuracy"] == 1.0 and res["ari"] == 1.0 and res["true_segments"] == 4
    assert res["labeled_customers"] == 100
    assert client.post("/api/cluster", json={"k": 2}).json()["accuracy"] <= 0.5


def test_bundled_datasets():
    """Every CSV in dataset/ uploads cleanly, has 70 to 100 customers and gets the right K."""
    files = sorted(DATASET_DIR.glob("*.csv"))
    assert files, "no datasets found"
    for f in files:
        r = post_csv(f.read_text())
        assert r.status_code == 200, (f.name, r.text)
        n = r.json()["processed_rows"]
        assert 70 <= n <= 100, (f.name, n)
        k = client.get("/api/elbow").json()["recommended_k"]
        res = client.post("/api/cluster", json={"k": k}).json()
        assert k == res["true_segments"], (f.name, k)
        assert res["accuracy"] >= 0.95, (f.name, res["accuracy"])


if __name__ == "__main__":
    test_all_endpoints()
    test_accuracy_with_true_segment_column()
    test_bundled_datasets()
    print("All FastAPI tests passed successfully!")
