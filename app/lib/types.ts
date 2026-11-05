export interface Customer {
  customer_id: string;
  name: string;
  age: number;
  gender: string | null;
  district: string | null;
  product_type: string | null;
  repeat_purchases: number;
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
  avg_age: number;
  avg_repeats: number;
  /** [age, repeat_purchases] */
  centroid: [number, number];
  top_product: string | null;
  top_product_share: number;
  top_district: string | null;
  female_share: number | null;
  description: string;
  strategy: string;
}

export interface Point {
  id: string;
  age: number;
  repeats: number;
  product: string | null;
  cluster: number;
}

export interface CentroidPoint {
  cluster: number;
  age: number;
  repeats: number;
  label: string;
}

export interface ClusterResponse {
  k: number;
  total_customers: number;
  silhouette_score: number;
  /** Present only when the upload has a ground-truth segment column (0 to 1). */
  accuracy?: number | null;
  ari?: number | null;
  nmi?: number | null;
  true_segments?: number | null;
  labeled_customers?: number;
  clusters: ClusterSummary[];
  points: Point[];
  centroids: CentroidPoint[];
}

export interface PredictRequest {
  age: number;
  repeat_purchases: number;
}

export interface PredictResponse {
  cluster_id: number;
  label: string;
  description: string;
  strategy: string;
  age: number;
  repeat_purchases: number;
  nearest_centroid: [number, number];
}

export interface UploadResponse {
  total_rows: number;
  processed_rows: number;
  skipped_rows: number;
  detected_columns: {
    name: string | null;
    age: string | null;
    gender: string | null;
    district: string | null;
    product: string | null;
    repeats: string | null;
    label: string | null;
  };
}

export interface HealthResponse {
  status: string;
  message: string;
  customers_loaded: number;
}

