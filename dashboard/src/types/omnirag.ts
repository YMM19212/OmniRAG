export type KBState = "disconnected" | "connecting" | "ready" | "error";

export interface ApiResponse<T> {
  success: boolean;
  message: string;
  data: T;
  error: string | null;
}

export interface KBConfig {
  milvus_uri?: string | null;
  milvus_host: string;
  milvus_port: number;
  collection_name: string;
  embedding_api_url: string;
  model_name: string;
  api_key: string;
  max_concurrent_embeds: number;
  enable_deduplication: boolean;
  dedup_mode: "semantic" | "strict";
  similarity_threshold: number;
}

export interface KBStatus {
  state: KBState;
  message: string;
  last_error: string | null;
  ready: boolean;
}

export interface CollectionStats {
  collection_name: string;
  entities_count: number;
  vector_dim: number;
  index_type: string;
  metric_type: string;
}

export interface SearchResult {
  id: string;
  distance?: number | null;
  text?: string | null;
  image_path?: string | null;
  video_path?: string | null;
  modality?: string | null;
  metadata?: Record<string, unknown> | null;
  thumbnail?: string | null;
  image_base64?: string | null;
  source?: "search" | "hybrid";
}

export interface SearchRequest {
  text?: string;
  top_k: number;
  distance_threshold: number;
  filter_modality?: string;
}

export interface HybridSearchRequest {
  text?: string;
  top_k: number;
  target_modality?: string;
}

export interface DocumentCreatePayload {
  text?: string;
  metadata?: string;
  store_image_base64?: boolean;
  extract_thumbnail?: boolean;
  skip_duplicate?: boolean;
  image?: File | null;
  video?: File | null;
}

export interface BatchDocumentPayload {
  files: File[];
  common_text?: string;
  metadata?: string;
  store_image_base64?: boolean;
  extract_thumbnail?: boolean;
  skip_duplicate?: boolean;
  max_concurrent?: number;
}
