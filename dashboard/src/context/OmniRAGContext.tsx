import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { kbApi } from "../services/api";
import type { CollectionStats, KBConfig, KBStatus, SearchResult } from "../types/omnirag";

const defaultConfig: KBConfig = {
  milvus_uri: "./data/multimodal_kb.db",
  milvus_host: "localhost",
  milvus_port: 19530,
  collection_name: "multimodal_kb",
  embedding_api_url: "https://api.jina.ai/v1/embeddings",
  model_name: "jina-embeddings-v4",
  api_key: "",
  max_concurrent_embeds: 8,
  enable_deduplication: true,
  dedup_mode: "semantic",
  similarity_threshold: 0.95,
};

const defaultStatus: KBStatus = {
  state: "disconnected",
  message: "Knowledge base is not initialized.",
  last_error: null,
  ready: false,
};

interface OmniRAGContextValue {
  status: KBStatus;
  config: KBConfig;
  stats: CollectionStats | null;
  recentResults: SearchResult[];
  selectedIds: string[];
  loading: boolean;
  error: string | null;
  setConfig: (config: KBConfig) => void;
  refreshStatus: () => Promise<void>;
  refreshStats: () => Promise<void>;
  initialize: (config: KBConfig) => Promise<void>;
  registerResults: (results: SearchResult[], source: "search" | "hybrid") => void;
  toggleSelected: (id: string) => void;
  clearSelection: () => void;
  deleteSelected: () => Promise<void>;
}

const OmniRAGContext = createContext<OmniRAGContextValue | undefined>(undefined);

export function OmniRAGProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<KBStatus>(defaultStatus);
  const [config, setConfig] = useState<KBConfig>(defaultConfig);
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [recentResults, setRecentResults] = useState<SearchResult[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = async () => {
    try {
      const [nextStatus, nextConfig] = await Promise.all([kbApi.getStatus(), kbApi.getConfig()]);
      setStatus(nextStatus);
      setConfig((current) => ({ ...current, ...nextConfig }));
      if (nextStatus.ready) {
        const nextStats = await kbApi.getStats();
        setStats(nextStats);
      } else {
        setStats(null);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh status");
    }
  };

  const refreshStats = async () => {
    try {
      const nextStats = await kbApi.getStats();
      setStats(nextStats);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh stats");
    }
  };

  const initialize = async (nextConfig: KBConfig) => {
    setLoading(true);
    setStatus({
      state: "connecting",
      message: "Initializing knowledge base connection...",
      last_error: null,
      ready: false,
    });
    try {
      const data = await kbApi.initialize(nextConfig);
      setConfig((current) => ({ ...current, ...data.config }));
      setStatus(data.status);
      setStats(data.stats);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Initialization failed";
      setStatus({
        state: "error",
        message: "Knowledge base initialization failed.",
        last_error: message,
        ready: false,
      });
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const registerResults = (results: SearchResult[], source: "search" | "hybrid") => {
    const tagged = results.map((result) => ({ ...result, source }));
    setRecentResults((current) => {
      const merged = [...tagged, ...current];
      const map = new Map<string, SearchResult>();
      merged.forEach((item) => map.set(item.id, item));
      return Array.from(map.values());
    });
  };

  const toggleSelected = (id: string) => {
    setSelectedIds((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
  };

  const clearSelection = () => setSelectedIds([]);

  const deleteSelected = async () => {
    if (selectedIds.length === 0) return;
    setLoading(true);
    try {
      const data = await kbApi.deleteDocuments(selectedIds);
      setRecentResults((current) => current.filter((item) => !selectedIds.includes(item.id)));
      setSelectedIds([]);
      setStats(data.stats);
      await refreshStatus();
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refreshStatus();
  }, []);

  const value = useMemo(
    () => ({
      status,
      config,
      stats,
      recentResults,
      selectedIds,
      loading,
      error,
      setConfig,
      refreshStatus,
      refreshStats,
      initialize,
      registerResults,
      toggleSelected,
      clearSelection,
      deleteSelected,
    }),
    [status, config, stats, recentResults, selectedIds, loading, error],
  );

  return <OmniRAGContext.Provider value={value}>{children}</OmniRAGContext.Provider>;
}

export function useOmniRAG() {
  const context = useContext(OmniRAGContext);
  if (!context) {
    throw new Error("useOmniRAG must be used within OmniRAGProvider");
  }
  return context;
}
