# Customer Segmentation Using K-Means Clustering on an Artificial Dataset

A complete, full-stack Machine Learning mini-project built for college submissions and viva voce examinations. The project demonstrates unsupervised market segmentation using **K-Means Clustering** on customer behavioral data with a **FastAPI + scikit-learn** backend and a responsive **Next.js (App Router, TypeScript, Tailwind CSS, Recharts)** dashboard.

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

---

## 📁 Repository Structure

```
heshly_ml_project/
├── app/                          # Next.js App Router Frontend
│   ├── components/               # Small modular client components
│   │   ├── Navbar.tsx            # Header with live FastAPI status ping
│   │   ├── HeroSection.tsx       # Introduction & quick navigation
│   │   ├── DatasetSection.tsx    # Paginated customer table with search & metrics
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
│   ├── dataset/
│   │   └── customers.csv         # 200 synthetic customer records
│   ├── generate_data.py          # Synthetic dataset generator with 4 quadrants
│   ├── model.py                  # StandardScaler, Elbow (WCSS), KMeans fit, prediction & labels
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

1. **Synthetic Dataset Generation (`backend/generate_data.py`)**:
   - Synthesizes 200 customer samples across 4 distinct demographic quadrants:
     - *High Income, High Spending* (VIP / Target Shoppers)
     - *High Income, Low Spending* (Careful Savers)
     - *Low Income, High Spending* (Trendsetters / Enthusiasts)
     - *Low Income, Low Spending* (Budget / Value Seekers)
   - Features: `Customer_ID`, `Age` (18-70), `Annual_Income` ($15k-$140k), `Spending_Score` (1-100).

2. **Feature Preprocessing (`StandardScaler`)**:
   - `Annual_Income` and `Spending_Score` are standardized using z-score normalization ($z = \frac{x - \mu}{\sigma}$) so features with large numeric variance do not artificially dominate Euclidean distances.

3. **Hyperparameter Selection (Elbow Method)**:
   - Calculates **Within-Cluster Sum of Squares (WCSS / Inertia)** for $K \in [2, 8]$.
   - The elbow curve drops sharply until $K=4$ (WCSS drops from 215.2 to 31.58) and flattens thereafter, demonstrating that $K=4$ is the optimal number of clusters.

4. **K-Means Clustering & Evaluation**:
   - Clusters data using scikit-learn's `KMeans(n_clusters=k, random_state=42, n_init=10)`.
   - Evaluates separation using **Silhouette Score** ($\approx 0.714$ for $K=4$, representing excellent cluster cohesion and separation).
   - Inverses centroids back to human-readable units (`$k` and score) using `scaler.inverse_transform()`.
   - Automatically generates persona labels, descriptions, and actionable marketing strategies.

5. **Inference / Prediction**:
   - Accepts a new customer's Age, Income, and Spending Score.
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
- If `Annual_Income` is measured in thousands (e.g., $15,000 to $140,000) while `Spending_Score` ranges from $1$ to $100$, the income feature will dominate the distance calculation simply due to scale difference.
- `StandardScaler` normalizes each feature to have a **mean of 0** and a **standard deviation of 1**, ensuring every feature contributes proportionally.

---

### 4. What is WCSS and how does the Elbow Method determine optimal K?
- **WCSS (Within-Cluster Sum of Squares)**, also known as **Inertia**, measures cluster compactness:
  $$\text{WCSS} = \sum_{j=1}^{K} \sum_{x_i \in C_j} ||x_i - \mu_j||^2$$
- As $K$ increases, WCSS monotonically decreases (reaching 0 when $K = N$).
- The **Elbow Method** plots WCSS against $K$. The point where the rate of decrease abruptly slows down (forming an &quot;elbow&quot; bend) represents the point of diminishing returns. In this project, the elbow clearly appears at **$K = 4$**.

---

### 5. What is the Silhouette Score?
- The **Silhouette Score** measures how similar a data point is to its own cluster (cohesion $a$) compared to other clusters (separation $b$):
  $$s = \frac{b - a}{\max(a, b)}$$
- It ranges from **-1 to +1**:
  - **$+1$**: Points are well clustered and far from neighboring clusters.
  - **$0$**: Points are on or very close to the decision boundary between two clusters.
  - **$-1$**: Points may have been assigned to the incorrect cluster.
- Our baseline model achieves **$\approx 0.714$**, signifying distinct, well-separated market segments.

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
| `GET` | `/api/health` | Returns backend operational status and customer count |
| `GET` | `/api/dataset` | Returns customer dataset records (supports `?limit=`) |
| `GET` | `/api/elbow` | Computes WCSS values for $K \in [2, 8]$ |
| `POST` | `/api/cluster` | Trains KMeans with `{ k: int }` (2-8), returns metrics & centroids |
| `POST` | `/api/predict` | Classifies customer from `{ age, income, spending_score }` |

---

## 👥 Authors & Academic Context
- Developed for **Academic Machine Learning Mini-Project / Viva Voce**.
- Monorepo architecture demonstrating end-to-end integration of statistical ML algorithms with modern web technology.
