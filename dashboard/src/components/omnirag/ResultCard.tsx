import { useState } from "react";
import Badge from "../ui/badge/Badge";
import type { SearchResult } from "../../types/omnirag";
import { useI18n } from "../../context/I18nContext";
import { buildMediaUrl } from "../../services/api";

function getSimilarity(distance?: number | null) {
  if (distance === null || distance === undefined) return null;
  return Math.max(0, 1 - distance).toFixed(4);
}

function getPreview(result: SearchResult) {
  if (result.image_base64) {
    return `data:image/jpeg;base64,${result.image_base64}`;
  }
  if (result.thumbnail) {
    return `data:image/jpeg;base64,${result.thumbnail}`;
  }
  return null;
}

export default function ResultCard({
  result,
  selected,
  onToggle,
}: {
  result: SearchResult;
  selected?: boolean;
  onToggle?: (id: string) => void;
}) {
  const { t } = useI18n();
  const similarity = getSimilarity(result.distance);
  const preview = getPreview(result);
  const [imageLoadFailed, setImageLoadFailed] = useState(false);
  const imageUrl = !preview && !imageLoadFailed ? buildMediaUrl(result.image_path) : null;

  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-theme-xs dark:border-gray-800 dark:bg-white/[0.03]">
      <div className="flex flex-col gap-5 xl:flex-row">
        <div className="flex h-48 w-full items-center justify-center overflow-hidden rounded-2xl bg-gray-100 xl:w-72 dark:bg-gray-800">
          {preview ? (
            <img src={preview} alt={result.id} className="h-full w-full object-cover" />
          ) : imageUrl ? (
            <img
              src={imageUrl}
              alt={result.id}
              className="h-full w-full object-cover"
              onError={() => setImageLoadFailed(true)}
            />
          ) : (
            <div className="px-4 text-center text-sm text-gray-500 dark:text-gray-400">
              {result.video_path ? t("result.videoPreviewUnavailable") : t("result.textOnly")}
            </div>
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div className="min-w-0">
              <p className="text-sm text-gray-500 dark:text-gray-400">ID</p>
              <p className="truncate font-mono text-sm text-gray-900 dark:text-white">
                {result.id}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge color="info">{(result.modality || t("result.unknown")).toUpperCase()}</Badge>
              {result.source ? <Badge color="light">{result.source.toUpperCase()}</Badge> : null}
              {similarity ? <Badge color="success">SIM {similarity}</Badge> : null}
            </div>
          </div>

          {result.text ? (
            <p className="mt-4 text-sm leading-6 text-gray-700 dark:text-gray-300">{result.text}</p>
          ) : null}

          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div className="rounded-xl bg-gray-50 p-3 dark:bg-gray-800/70">
              <p className="text-xs uppercase tracking-wide text-gray-400">{t("result.media")}</p>
              <p className="mt-2 break-all text-sm text-gray-700 dark:text-gray-300">
                {result.video_path || result.image_path || t("result.inlineTextOnly")}
              </p>
            </div>
            <div className="rounded-xl bg-gray-50 p-3 dark:bg-gray-800/70">
              <p className="text-xs uppercase tracking-wide text-gray-400">{t("result.metadata")}</p>
              <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-words text-sm text-gray-700 dark:text-gray-300">
                {JSON.stringify(result.metadata ?? {}, null, 2)}
              </pre>
            </div>
          </div>

          {onToggle ? (
            <label className="mt-4 inline-flex items-center gap-3 text-sm text-gray-700 dark:text-gray-300">
              <input
                type="checkbox"
                checked={Boolean(selected)}
                onChange={() => onToggle(result.id)}
                className="h-4 w-4 rounded border-gray-300 text-brand-500 focus:ring-brand-500"
              />
              {t("result.markDelete")}
            </label>
          ) : null}
        </div>
      </div>
    </div>
  );
}
