import {
  DatasetResponse,
  ElbowResponse,
  ClusterResponse,
  PredictRequest,
  PredictResponse,
  HealthResponse,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      let errorMsg = `Server error: ${res.status} ${res.statusText}`;
      try {
        const errorJson = await res.json();
        if (errorJson.detail) {
          if (Array.isArray(errorJson.detail)) {
            errorMsg = errorJson.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join(", ");
          } else {
            errorMsg = String(errorJson.detail);
          }
        }
      } catch {
        // Fallback to status text
      }
      throw new Error(errorMsg);
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError") || err.message.includes("ECONNREFUSED")) {
        throw new Error(
          `Unable to connect to ML backend at ${BASE_URL}. Ensure FastAPI is running on port 8000 (cd backend && uvicorn main:app --reload).`
        );
      }
      throw err;
    }
    throw new Error("An unexpected error occurred while communicating with the server.");
  }
}

/** Check health status of FastAPI ML server */
export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

/** Fetch customer dataset records (optionally limited) */
export async function getDataset(limit?: number): Promise<DatasetResponse> {
  const query = limit ? `?limit=${limit}` : "";
  return request<DatasetResponse>(`/api/dataset${query}`);
}

/** Fetch Elbow method WCSS vs K values */
export async function getElbowCurve(): Promise<ElbowResponse> {
  return request<ElbowResponse>("/api/elbow");
}

/** Train K-Means clustering with given K (2 to 8) */
export async function runKMeans(k: number): Promise<ClusterResponse> {
  return request<ClusterResponse>("/api/cluster", {
    method: "POST",
    body: JSON.stringify({ k }),
  });
}

/** Classify a customer into one of the clusters */
export async function predictCustomer(payload: PredictRequest): Promise<PredictResponse> {
  return request<PredictResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
