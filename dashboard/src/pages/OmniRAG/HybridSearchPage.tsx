import { useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import ComponentCard from "../../components/common/ComponentCard";
import Button from "../../components/ui/button/Button";
import FilePreview from "../../components/omnirag/FilePreview";
import PageIntro from "../../components/omnirag/PageIntro";
import ResultCard from "../../components/omnirag/ResultCard";
import StatusBanner from "../../components/omnirag/StatusBanner";
import { useOmniRAG } from "../../context/OmniRAGContext";
import { useI18n } from "../../context/I18nContext";
import { kbApi } from "../../services/api";
import type { SearchResult } from "../../types/omnirag";

export default function HybridSearchPage() {
  const { status, registerResults } = useOmniRAG();
  const { t, locale } = useI18n();
  const [text, setText] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [topK, setTopK] = useState(5);
  const [targetModality, setTargetModality] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await kbApi.hybridSearch(
        {
          text,
          top_k: topK,
          target_modality: targetModality || undefined,
        },
        image,
        video,
      );
      setResults(data);
      registerResults(data, "hybrid");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Hybrid search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PageMeta title={`OmniRAG | ${t("hybrid.title")}`} description={t("hybrid.desc")} />
      <PageIntro
        title={t("hybrid.title")}
        description={t("hybrid.desc")}
      />

      <div className="space-y-6">
        <StatusBanner status={status} />

        <ComponentCard title={t("hybrid.panel")} desc={t("hybrid.panelDesc")}>
          <div className="grid gap-4 md:grid-cols-2">
            <label className="space-y-2 md:col-span-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("hybrid.queryText")}</span>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
              />
            </label>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("hybrid.refImage")}</span>
              <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] ?? null)} />
            </label>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("hybrid.refVideo")}</span>
              <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files?.[0] ?? null)} />
            </label>
            <FilePreview file={image} kind="image" />
            <FilePreview file={video} kind="video" />
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("search.topK")}</span>
              <input
                type="number"
                value={topK}
                min={1}
                max={20}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
              />
            </label>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("hybrid.target")}</span>
              <select
                value={targetModality}
                onChange={(e) => setTargetModality(e.target.value)}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
              >
                <option value="">{locale === "zh" ? "全部" : "all"}</option>
                <option value="image">image</option>
                <option value="video">video</option>
                <option value="text">text</option>
                <option value="visual">visual</option>
              </select>
            </label>
          </div>
          <Button onClick={() => void runSearch()} disabled={loading || !status.ready}>
            {loading ? t("hybrid.running") : t("hybrid.run")}
          </Button>
          {error ? <p className="text-sm text-error-600 dark:text-error-400">{error}</p> : null}
        </ComponentCard>

        <div className="space-y-4">
          {results.map((result) => (
            <ResultCard key={result.id} result={result} />
          ))}
          {results.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-gray-300 px-6 py-10 text-center text-sm text-gray-500 dark:border-gray-700 dark:text-gray-400">
              {t("hybrid.empty")}
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
}
