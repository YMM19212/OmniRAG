import { useEffect, useState } from "react";
import PageMeta from "../../components/common/PageMeta";
import ComponentCard from "../../components/common/ComponentCard";
import Button from "../../components/ui/button/Button";
import FilePreview from "../../components/omnirag/FilePreview";
import PageIntro from "../../components/omnirag/PageIntro";
import StatusBanner from "../../components/omnirag/StatusBanner";
import { useOmniRAG } from "../../context/OmniRAGContext";
import { useI18n } from "../../context/I18nContext";
import { kbApi } from "../../services/api";

export default function IngestPage() {
  const { status, refreshStats, refreshStatus } = useOmniRAG();
  const { t } = useI18n();
  const [text, setText] = useState("");
  const [metadata, setMetadata] = useState("{}");
  const [image, setImage] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [storeImageBase64, setStoreImageBase64] = useState(false);
  const [extractThumbnail, setExtractThumbnail] = useState(true);
  const [skipDuplicate, setSkipDuplicate] = useState(true);
  const [files, setFiles] = useState<File[]>([]);
  const [commonText, setCommonText] = useState("");
  const [batchMetadata, setBatchMetadata] = useState("{}");
  const [maxConcurrent, setMaxConcurrent] = useState(4);
  const [parquetFile, setParquetFile] = useState<File | null>(null);
  const [parquetMaxRows, setParquetMaxRows] = useState(100);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Refresh once on page entry so stale ready state does not block imports.
    void refreshStatus();
  }, []);

  const ensureReady = async () => {
    const latestStatus = await kbApi.getStatus();
    if (latestStatus.ready) {
      await refreshStatus();
      return true;
    }

    setFeedback(t("ingest.initRequired"));
    await refreshStatus();
    return false;
  };

  const submitSingle = async () => {
    setLoading(true);
    setFeedback(null);
    try {
      const ready = await ensureReady();
      if (!ready) {
        setLoading(false);
        return;
      }

      const result = await kbApi.createDocument({
        text,
        metadata,
        store_image_base64: storeImageBase64,
        extract_thumbnail: extractThumbnail,
        skip_duplicate: skipDuplicate,
        image,
        video,
      });
      setFeedback(`${t("ingest.addDocument")}. ID: ${result.id}`);
      setImage(null);
      setVideo(null);
      setLoading(false);
      void refreshStats();
    } catch (err) {
      setFeedback(err instanceof Error ? err.message : "Import failed");
      setLoading(false);
    }
  };

  const submitBatch = async () => {
    setLoading(true);
    setFeedback(null);
    try {
      const ready = await ensureReady();
      if (!ready) {
        setLoading(false);
        return;
      }

      const result = await kbApi.createBatchDocuments({
        files,
        common_text: commonText,
        metadata: batchMetadata,
        store_image_base64: storeImageBase64,
        extract_thumbnail: extractThumbnail,
        skip_duplicate: skipDuplicate,
        max_concurrent: maxConcurrent,
      });
      setFeedback(`${t("ingest.runBatch")}. ${result.count}`);
      setFiles([]);
      setLoading(false);
      void refreshStats();
    } catch (err) {
      setFeedback(err instanceof Error ? err.message : "Batch import failed");
      setLoading(false);
    }
  };

  const submitParquet = async () => {
    if (!parquetFile) return;

    setLoading(true);
    setFeedback(null);
    try {
      const ready = await ensureReady();
      if (!ready) {
        setLoading(false);
        return;
      }

      const result = await kbApi.createParquetDocuments({
        parquet: parquetFile,
        max_rows: parquetMaxRows > 0 ? parquetMaxRows : undefined,
        store_image_base64: storeImageBase64,
        skip_duplicate: skipDuplicate,
        max_concurrent: maxConcurrent,
      });
      const summary = `${t("ingest.parquetSuccess")}. ${result.count}/${result.parsed_rows}`;
      const errorDetail =
        result.failed_rows > 0 && result.errors.length > 0
          ? ` Failed: ${result.failed_rows}. ${result.errors.slice(0, 3).join(" | ")}`
          : "";
      setFeedback(`${summary}${errorDetail}`);
      setParquetFile(null);
      setLoading(false);
      void refreshStats();
    } catch (err) {
      setFeedback(err instanceof Error ? err.message : "Parquet import failed");
      setLoading(false);
    }
  };

  return (
    <>
      <PageMeta title={`OmniRAG | ${t("ingest.title")}`} description={t("ingest.desc")} />
      <PageIntro
        title={t("ingest.title")}
        description={t("ingest.desc")}
      />

      <div className="space-y-6">
        <StatusBanner status={status} />

        <div className="grid gap-6 xl:grid-cols-2">
          <ComponentCard title={t("ingest.single")} desc={t("ingest.singleDesc")}>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.text")}</span>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={5}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
              />
            </label>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.metadata")}</span>
              <textarea
                value={metadata}
                onChange={(e) => setMetadata(e.target.value)}
                rows={5}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 font-mono text-sm dark:border-gray-700 dark:bg-gray-900"
              />
            </label>
            <div className="grid gap-4 md:grid-cols-2">
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.image")}</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setImage(e.target.files?.[0] ?? null)}
                  className="w-full text-sm text-gray-500"
                />
              </label>
              <label className="space-y-2">
                <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.video")}</span>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setVideo(e.target.files?.[0] ?? null)}
                  className="w-full text-sm text-gray-500"
                />
              </label>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <FilePreview file={image} kind="image" />
              <FilePreview file={video} kind="video" />
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              <label className="inline-flex items-center gap-3 text-sm text-gray-700 dark:text-gray-300">
                <input
                  type="checkbox"
                  checked={storeImageBase64}
                  onChange={(e) => setStoreImageBase64(e.target.checked)}
                />
                {t("ingest.storeBase64")}
              </label>
              <label className="inline-flex items-center gap-3 text-sm text-gray-700 dark:text-gray-300">
                <input
                  type="checkbox"
                  checked={extractThumbnail}
                  onChange={(e) => setExtractThumbnail(e.target.checked)}
                />
                {t("ingest.extractThumbnail")}
              </label>
              <label className="inline-flex items-center gap-3 text-sm text-gray-700 dark:text-gray-300">
                <input
                  type="checkbox"
                  checked={skipDuplicate}
                  onChange={(e) => setSkipDuplicate(e.target.checked)}
                />
                {t("ingest.skipDuplicate")}
              </label>
            </div>
            {!status.ready ? (
              <p className="text-sm text-amber-600 dark:text-amber-400">{t("ingest.initRequired")}</p>
            ) : null}
            <Button onClick={() => void submitSingle()} disabled={loading}>
              {loading ? t("ingest.submitting") : t("ingest.addDocument")}
            </Button>
          </ComponentCard>

          <ComponentCard title={t("ingest.batch")} desc={t("ingest.batchDesc")}>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.files")}</span>
              <input
                type="file"
                multiple
                accept="image/*,video/*"
                onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
                className="w-full text-sm text-gray-500"
              />
            </label>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.sharedText")}</span>
              <textarea
                value={commonText}
                onChange={(e) => setCommonText(e.target.value)}
                rows={4}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
              />
            </label>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.sharedMetadata")}</span>
              <textarea
                value={batchMetadata}
                onChange={(e) => setBatchMetadata(e.target.value)}
                rows={5}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 font-mono text-sm dark:border-gray-700 dark:bg-gray-900"
              />
            </label>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.maxConcurrent")}</span>
              <input
                type="number"
                value={maxConcurrent}
                min={1}
                max={8}
                onChange={(e) => setMaxConcurrent(Number(e.target.value))}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
              />
            </label>
            <div className="rounded-xl bg-gray-50 p-4 text-sm text-gray-700 dark:bg-gray-800/70 dark:text-gray-300">
              {files.length === 0
                ? t("ingest.noFiles")
                : `${files.length} files selected: ${files.map((file) => file.name).join(", ")}`}
            </div>
            {!status.ready ? (
              <p className="text-sm text-amber-600 dark:text-amber-400">{t("ingest.initRequired")}</p>
            ) : null}
            <Button onClick={() => void submitBatch()} disabled={loading || files.length === 0}>
              {loading ? t("ingest.importing") : t("ingest.runBatch")}
            </Button>
          </ComponentCard>

          <ComponentCard title={t("ingest.parquet")} desc={t("ingest.parquetDesc")}>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.parquetFile")}</span>
              <input
                type="file"
                accept=".parquet"
                onChange={(e) => setParquetFile(e.target.files?.[0] ?? null)}
                className="w-full text-sm text-gray-500"
              />
            </label>
            <label className="space-y-2">
              <span className="text-sm text-gray-700 dark:text-gray-300">{t("ingest.parquetLimit")}</span>
              <input
                type="number"
                value={parquetMaxRows}
                min={1}
                onChange={(e) => setParquetMaxRows(Number(e.target.value))}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm dark:border-gray-700 dark:bg-gray-900"
              />
            </label>
            <div className="rounded-xl bg-gray-50 p-4 text-sm text-gray-700 dark:bg-gray-800/70 dark:text-gray-300">
              {t("ingest.parquetHint")}
              {parquetFile ? ` Selected: ${parquetFile.name}` : ""}
            </div>
            {!status.ready ? (
              <p className="text-sm text-amber-600 dark:text-amber-400">{t("ingest.initRequired")}</p>
            ) : null}
            <Button onClick={() => void submitParquet()} disabled={loading || !parquetFile}>
              {loading ? t("ingest.importing") : t("ingest.parquetRun")}
            </Button>
          </ComponentCard>
        </div>

        {feedback ? (
          <div className="rounded-2xl border border-gray-200 bg-white px-5 py-4 text-sm text-gray-700 dark:border-gray-800 dark:bg-white/[0.03] dark:text-gray-300">
            {feedback}
          </div>
        ) : null}
      </div>
    </>
  );
}
