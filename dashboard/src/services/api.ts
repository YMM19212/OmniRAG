import type {
  ApiResponse,
  BatchDocumentPayload,
  CollectionStats,
  DocumentCreatePayload,
  HybridSearchRequest,
  KBConfig,
  KBStatus,
  SearchRequest,
  SearchResult,
} from "../types/omnirag";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api";

export function buildMediaUrl(path?: string | null): string | null {
  if (!path) return null;
  return `${API_BASE}/media?path=${encodeURIComponent(path)}`;
}

async function parseResponse<T>(response: Response): Promise<ApiResponse<T>> {
  const payload = (await response.json()) as ApiResponse<T>;
  if (!response.ok || !payload.success) {
    throw new Error(payload.error || payload.message || "Request failed");
  }
  return payload;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  return (await parseResponse<T>(response)).data;
}

async function sendJson<T>(path: string, method: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return (await parseResponse<T>(response)).data;
}

async function sendForm<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
  });
  return (await parseResponse<T>(response)).data;
}

export const kbApi = {
  getStatus: () => get<KBStatus>("/kb/status"),
  getConfig: () => get<KBConfig>("/kb/config"),
  getStats: () => get<CollectionStats>("/kb/stats"),
  initialize: (config: KBConfig) =>
    sendJson<{ status: KBStatus; config: KBConfig; stats: CollectionStats }>(
      "/kb/initialize",
      "POST",
      config,
    ),
  createDocument: (payload: DocumentCreatePayload) => {
    const formData = new FormData();
    formData.append("text", payload.text ?? "");
    formData.append("metadata", payload.metadata ?? "{}");
    formData.append("store_image_base64", String(Boolean(payload.store_image_base64)));
    formData.append("extract_thumbnail", String(payload.extract_thumbnail ?? true));
    formData.append("skip_duplicate", String(payload.skip_duplicate ?? true));
    if (payload.image) formData.append("image", payload.image);
    if (payload.video) formData.append("video", payload.video);
    return sendForm<{ id: string }>("/kb/documents", formData);
  },
  createBatchDocuments: (payload: BatchDocumentPayload) => {
    const formData = new FormData();
    payload.files.forEach((file) => formData.append("files", file));
    formData.append("common_text", payload.common_text ?? "");
    formData.append("metadata", payload.metadata ?? "{}");
    formData.append("store_image_base64", String(Boolean(payload.store_image_base64)));
    formData.append("extract_thumbnail", String(payload.extract_thumbnail ?? true));
    formData.append("skip_duplicate", String(payload.skip_duplicate ?? true));
    formData.append("max_concurrent", String(payload.max_concurrent ?? 4));
    return sendForm<{ ids: string[]; count: number }>("/kb/documents/batch", formData);
  },
  search: (payload: SearchRequest, image?: File | null, video?: File | null) => {
    const formData = new FormData();
    formData.append("payload", JSON.stringify(payload));
    if (image) formData.append("image", image);
    if (video) formData.append("video", video);
    return sendForm<SearchResult[]>("/kb/search", formData);
  },
  hybridSearch: (payload: HybridSearchRequest, image?: File | null, video?: File | null) => {
    const formData = new FormData();
    formData.append("payload", JSON.stringify(payload));
    if (image) formData.append("image", image);
    if (video) formData.append("video", video);
    return sendForm<SearchResult[]>("/kb/search/hybrid", formData);
  },
  deleteDocuments: (ids: string[]) =>
    sendJson<{ deleted_ids: string[]; stats: CollectionStats }>("/kb/documents", "DELETE", {
      ids,
    }),
};
