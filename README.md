# Customer Segmentation Using K-Means Clustering on an Uploaded Dataset

A complete, full-stack Machine Learning mini-project built for college submissions and viva voce examinations. The project demonstrates unsupervised market segmentation using **K-Means Clustering** on a customer CSV that you upload with a **FastAPI + scikit-learn** backend and a responsive **Next.js (App Router, TypeScript, Tailwind CSS, Recharts)** dashboard.

---

## 🚀 Quick Start & Run Instructions

### 1. Backend Setup (FastAPI + scikit-learn)

Open a terminal and navigate to the `backend/` directory:

```bash
cd backend

# Create and activate a virtual environment (optional but recommended)
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
# source venv/bin/activate

# Install required Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```

- **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

### 2. Frontend Setup (Next.js App Router)

Open a second terminal at the project root directory:

```bash
# Install NPM packages (if not already installed)
npm install

# Run the Next.js development server
npm run dev
```

- **Interactive Web Dashboard:** [http://localhost:3000](http://localhost:3000)

*(Note: The frontend connects to `http://localhost:8000` by default. You can customize this by setting `NEXT_PUBLIC_API_URL` in an `.env.local` file).*

### 3. Using the App

The app starts empty: **there is no built-in dataset.** Upload a customer CSV and the dashboard builds everything from it:

1. **Upload CSV**: drag in a file with `Age` and `Repeat_Purchases` columns (`Name`, `Gender`, `District`, `Product_Type` optional; 10 to 5,000 customers, max 5 MB). Ready-made files are in [`dataset/`](dataset/). A loader shows the processing stages.
2. **Dataset / Elbow / K-Means / Predict** sections then appear, all computed on your file. K is picked automatically with the silhouette score and can be changed afterwards.
3. Download the segmented customers as a CSV.

If the file has a `True_Segment` column, the dashboard also shows **accuracy** (plus ARI and NMI) against it.

---

## 📁 Repository Structure

```
heshly_ml_project/
├── app/                          # Next.js App Router Frontend
│   ├── components/               # Small modular client components
│   │   ├── Navbar.tsx            # Header with live FastAPI status ping
│   │   ├── HeroSection.tsx       # Introduction & quick navigation
│   │   ├── UploadSection.tsx     # CSV upload, processing loader, results download
│   │   ├── DatasetSection.tsx    # Paginated uploaded-customer table with search & metrics
│   │   ├── ElbowSection.tsx      # WCSS vs K line chart & viva notes
│   │   ├── ClusteringControl.tsx # K parameter slider (2 to 8) & train trigger
│   │   ├── ResultsSection.tsx    # Stat cards, 2D Scatter plot with centroids, cluster cards
│   │   └── PredictSection.tsx    # Real-time customer classification form & presets
│   ├── lib/
│   │   ├── api.ts                # Typed API client with offline error handling
│   │   └── types.ts              # Strict TypeScript interfaces
│   ├── globals.css               # Tailwind CSS v4 styling
│   ├── layout.tsx                # App layout & meta title
│   └── page.tsx                  # Main interactive dashboard page
├── backend/                      # FastAPI ML Backend
│   ├── model.py                  # CSV loading/validation, StandardScaler, Elbow (WCSS), KMeans fit, prediction & labels
│   ├── schemas.py                # Pydantic request/response validation models
│   ├── main.py                   # FastAPI application, CORS, endpoints
│   ├── requirements.txt          # Python ML libraries
│   └── test_api.py               # Automated endpoint verification test suite
├── package.json                  # Next.js dependencies & scripts
├── tsconfig.json                 # TypeScript compiler configuration
└── README.md                     # Documentation & Viva notes
```

---

## 🧠 Machine Learning Pipeline Overview

1. **Dataset Upload (`POST /api/upload`)**:
   - The user uploads a CSV. Columns are detected case-insensitively (`Age`, `Repeat_Purchases` / `Repeats`, optional `Name`, `Gender`, `District`, `Product_Type`, `True_Segment`).
   - Rows with missing, negative or non-numeric age/repeat purchases are skipped and counted.

2. **Feature Preprocessing (`StandardScaler`)**:
   - `Age` and `Repeat_Purchases` are standardized using z-score normalization ($z = \frac{x - \mu}{\sigma}$) so the feature with the larger numeric range does not dominate Euclidean distances.

3. **Hyperparameter Selection (Elbow Method)**:
   - Calculates **Within-Cluster Sum of Squares (WCSS / Inertia)** for $K \in [2, 8]$ on the uploaded data.
   - The elbow shows where extra clusters stop paying off; the K with the highest silhouette score is recommended and used for the first training run.

4. **K-Means Clustering & Evaluation**:
   - Clusters data using scikit-learn's `KMeans(n_clusters=k, random_state=42, n_init=10)`.
   - Evaluates separation using **Silhouette Score** (closer to 1 means better cohesion and separation).
   - Inverses centroids back to human-readable units (`$k` and score) using `scaler.inverse_transform()`.
   - Automatically generates persona labels, descriptions, and actionable marketing strategies.

5. **Inference / Prediction**:
   - Accepts a new customer's Age and Repeat Purchases.
   - Scales the input using the fitted `StandardScaler` and assigns the nearest centroid.
   - Displays segment details and highlights the new customer point on the 2D Scatter plot in real-time.

---

## 🎓 Viva Voce Notes (Questions & Answers for Students)

### 1. What is Clustering and how does it differ from Classification?
- **Clustering** is an **Unsupervised Learning** technique that groups unlabelled data points based on feature similarity without predefined target categories.
- **Classification** is **Supervised Learning**, where the model learns from labeled training data containing explicit ground-truth class labels (e.g., Spam vs. Not Spam).

---

### 2. How does the K-Means Algorithm work step-by-step?
The K-Means algorithm partitions $N$ observations into $K$ clusters through an iterative 4-step process:
1. **Initialization:** Select $K$ initial cluster centroids (either randomly or using $k$-means++ for optimal dispersion).
2. **Assignment Step:** Assign each data point $x_i$ to its nearest centroid $\mu_j$ based on minimum Euclidean distance:
   $$\arg\min_j ||x_i - \mu_j||^2$$
3. **Update Step:** Recalculate the position of each centroid $\mu_j$ as the arithmetic mean of all points assigned to that cluster:
   $$\mu_j = \frac{1}{|C_j|} \sum_{x \in C_j} x$$
4. **Convergence:** Repeat steps 2 and 3 until centroids stop moving (convergence reached) or the maximum number of iterations is completed.

---

### 3. Why is Feature Scaling (`StandardScaler`) essential for K-Means?
- K-Means relies entirely on **Euclidean distance**:
  $$d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2}$$
- `Age` ranges from about 16 to 75 while `Repeat_Purchases` ranges from about 0 to 20, so without scaling the age feature would dominate the distance calculation simply due to scale difference.
- `StandardScaler` normalizes each feature to have a **mean of 0** and a **standard deviation of 1**, ensuring every feature contributes proportionally.

---

### 4. What is WCSS and how does the Elbow Method determine optimal K?
- **WCSS (Within-Cluster Sum of Squares)**, also known as **Inertia**, measures cluster compactness:
  $$\text{WCSS} = \sum_{j=1}^{K} \sum_{x_i \in C_j} ||x_i - \mu_j||^2$$
- As $K$ increases, WCSS monotonically decreases (reaching 0 when $K = N$).
- The **Elbow Method** plots WCSS against $K$. The point where the rate of decrease abruptly slows down (forming an &quot;elbow&quot; bend) represents the point of diminishing returns. On `dataset/tn_gift_shop_100.csv` the elbow appears at **$K = 5$**.

---

### 5. What is the Silhouette Score?
- The **Silhouette Score** measures how similar a data point is to its own cluster (cohesion $a$) compared to other clusters (separation $b$):
  $$s = \frac{b - a}{\max(a, b)}$$
- It ranges from **-1 to +1**:
  - **$+1$**: Points are well clustered and far from neighboring clusters.
  - **$0$**: Points are on or very close to the decision boundary between two clusters.
  - **$-1$**: Points may have been assigned to the incorrect cluster.
- On `dataset/tn_gift_shop_100.csv` the model achieves **$\approx 0.665$** with **99% accuracy** against the true segments.

---

### 6. What are the Limitations of K-Means Clustering?
1. **Pre-specifying K:** Requires the user to define the number of clusters $K$ in advance (resolved using the Elbow Method or Silhouette analysis).
2. **Spherical Cluster Assumption:** Assumes clusters are convex and spherical with similar variance; it struggles with arbitrary shapes (like concentric circles or crescents, where DBSCAN is preferred).
3. **Sensitivity to Outliers:** Because centroids are calculated as arithmetic means, extreme outliers can significantly distort centroid placement.
4. **Local Minima:** Random initialization can lead to suboptimal local optima (mitigated by using `k-means++` and multiple initializations `n_init=10`).

---

## 🛠️ API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns backend operational status and number of uploaded customers |
| `POST` | `/api/upload` | Uploads a customer CSV (raw `text/csv` body, max 5 MB) and makes it the active dataset |
| `GET` | `/api/dataset` | Returns the uploaded customer records (supports `?limit=`) |
| `GET` | `/api/elbow` | Computes WCSS values for $K \in [2, 8]$ plus the silhouette-optimal `recommended_k` |
| `POST` | `/api/cluster` | Trains KMeans with `{ k: int }` (2-8), returns metrics & centroids |
| `POST` | `/api/predict` | Classifies a customer from `{ age, repeat_purchases }` |

Every endpoint except `health` and `upload` returns `409` until a dataset has been uploaded.

---

## 👥 Authors & Academic Context
- Developed for **Academic Machine Learning Mini-Project / Viva Voce**.
- Monorepo architecture demonstrating end-to-end integration of statistical ML algorithms with modern web technology.
