export interface Customer {
  customer_id: string;
  age: number | null;
  annual_income: number;
  spending_score: number;
  cluster?: number | null;
}

export interface DatasetResponse {
  total_customers: number;
  customers: Customer[];
}

export interface ElbowResponse {
  k_values: number[];
  wcss: number[];
  recommended_k: number;
}

export interface ClusterSummary {
  cluster_id: number;
  label: string;
  customer_count: number;
  percentage: number;
  avg_age: number | null;
  avg_income: number;
  avg_spending: number;
  centroid: [number, number];
  description: string;
  strategy: string;
}

export interface Point {
  id: string;
  age: number | null;
  income: number;
  spending: number;
  cluster: number;
}

export interface CentroidPoint {
  cluster: number;
  income: number;
  spending: number;
  label: string;
}

export interface ClusterResponse {
  k: number;
  total_customers: number;
  silhouette_score: number;
  clusters: ClusterSummary[];
  points: Point[];
  centroids: CentroidPoint[];
}

export interface PredictRequest {
  age?: number | null;
  income: number;
  spending_score: number;
}

export interface PredictResponse {
  cluster_id: number;
  label: string;
  description: string;
  strategy: string;
  income: number;
  spending_score: number;
  age: number | null;
  nearest_centroid: [number, number];
}

export interface UploadResponse {
  total_rows: number;
  processed_rows: number;
  skipped_rows: number;
  detected_columns: {
    id: string | null;
    age: string | null;
    income: string | null;
    spending: string | null;
  };
}

export interface HealthResponse {
  status: string;
  message: string;
  customers_loaded: number;
}

