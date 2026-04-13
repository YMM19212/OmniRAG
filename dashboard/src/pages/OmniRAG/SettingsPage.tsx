import { useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import ComponentCard from "../../components/common/ComponentCard";
import Button from "../../components/ui/button/Button";
import PageIntro from "../../components/omnirag/PageIntro";
import StatusBanner from "../../components/omnirag/StatusBanner";
import { useOmniRAG } from "../../context/OmniRAGContext";
import type { KBConfig } from "../../types/omnirag";
import { useI18n } from "../../context/I18nContext";

export default function SettingsPage() {
  const { status, config, stats, setConfig, initialize, loading } = useOmniRAG();
  const { t } = useI18n();
  const [form, setForm] = useState<KBConfig>(config);
  const [message, setMessage] = useState<string | null>(null);

  const update = <K extends keyof KBConfig>(key: K, value: KBConfig[K]) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const handleSubmit = async () => {
    setMessage(null);
    try {
      await initialize(form);
      setConfig(form);
      setMessage(t("settings.readyMessage"));
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Initialization failed");
    }
  };

  return (
    <>
      <PageMeta title={`OmniRAG | ${t("settings.title")}`} description={t("settings.desc")} />
      <PageIntro
        title={t("settings.title")}
        description={t("settings.desc")}
      />

      <div className="space-y-6">
        <StatusBanner status={status} />

        <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <ComponentCard title={t("settings.runtime")} desc={t("settings.runtimeDesc")}>
            <div className="grid gap-4 md:grid-cols-2">
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.milvusUri")}</span>
                <input
                  value={form.milvus_uri ?? ""}
                  onChange={(e) => update("milvus_uri", e.target.value)}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.collectionName")}</span>
                <input
                  value={form.collection_name}
                  onChange={(e) => update("collection_name", e.target.value)}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.milvusHost")}</span>
                <input
                  value={form.milvus_host}
                  onChange={(e) => update("milvus_host", e.target.value)}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.milvusPort")}</span>
                <input
                  type="number"
                  value={form.milvus_port}
                  onChange={(e) => update("milvus_port", Number(e.target.value))}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.embeddingUrl")}</span>
                <input
                  value={form.embedding_api_url}
                  onChange={(e) => update("embedding_api_url", e.target.value)}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.embeddingModel")}</span>
                <input
                  value={form.model_name}
                  onChange={(e) => update("model_name", e.target.value)}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
              <label className="space-y-2 md:col-span-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.embeddingApiKey")}</span>
                <input
                  type="password"
                  value={form.api_key}
                  onChange={(e) => update("api_key", e.target.value)}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.maxConcurrentEmbeds")}</span>
                <input
                  type="number"
                  value={form.max_concurrent_embeds}
                  onChange={(e) => update("max_concurrent_embeds", Number(e.target.value))}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.dedupMode")}</span>
                <select
                  value={form.dedup_mode}
                  onChange={(e) => update("dedup_mode", e.target.value as KBConfig["dedup_mode"])}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                >
                  <option value="semantic">semantic</option>
                  <option value="strict">strict</option>
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("settings.similarityThreshold")}</span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={form.similarity_threshold}
                  onChange={(e) => update("similarity_threshold", Number(e.target.value))}
                  className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
                />
              </label>
            </div>

            <label className="inline-flex items-center gap-3 text-sm text-gray-700 dark:text-gray-300">
              <input
                type="checkbox"
                checked={form.enable_deduplication}
                onChange={(e) => update("enable_deduplication", e.target.checked)}
                className="h-4 w-4 rounded border-gray-300 text-brand-500 focus:ring-brand-500"
              />
              {t("settings.enableDedup")}
            </label>

            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={() => void handleSubmit()} disabled={loading}>
                {loading ? t("common.initializing") : t("common.initialize")}
              </Button>
              {message ? (
                <span className="text-sm text-gray-500 dark:text-gray-400">{message}</span>
              ) : null}
            </div>
          </ComponentCard>

          <ComponentCard title={t("settings.currentWorkspace")} desc={t("settings.currentWorkspaceDesc")}>
            <div className="space-y-4">
              <div className="rounded-xl bg-gray-50 p-4 dark:bg-gray-800/70">
                <p className="text-xs uppercase tracking-wide text-gray-400">{t("settings.statsPayload")}</p>
                <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-sm text-gray-700 dark:text-gray-300">
                  {JSON.stringify(stats ?? {}, null, 2)}
                </pre>
              </div>
              <div className="rounded-xl bg-gray-50 p-4 dark:bg-gray-800/70">
                <p className="text-xs uppercase tracking-wide text-gray-400">{t("settings.configPayload")}</p>
                <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-sm text-gray-700 dark:text-gray-300">
                  {JSON.stringify(form, null, 2)}
                </pre>
              </div>
            </div>
          </ComponentCard>
        </div>
      </div>
    </>
  );
}
