import PageMeta from "../../components/common/PageMeta";
import ComponentCard from "../../components/common/ComponentCard";
import Button from "../../components/ui/button/Button";
import PageIntro from "../../components/omnirag/PageIntro";
import ResultCard from "../../components/omnirag/ResultCard";
import StatusBanner from "../../components/omnirag/StatusBanner";
import { useOmniRAG } from "../../context/OmniRAGContext";
import { useI18n } from "../../context/I18nContext";

export default function RecordsPage() {
  const {
    status,
    recentResults,
    selectedIds,
    toggleSelected,
    clearSelection,
    deleteSelected,
    refreshStats,
    stats,
    loading,
  } = useOmniRAG();
  const { t } = useI18n();

  return (
    <>
      <PageMeta title={`OmniRAG | ${t("records.title")}`} description={t("records.desc")} />
      <PageIntro title={t("records.title")} description={t("records.desc")} />

      <div className="space-y-6">
        <StatusBanner status={status} />

        <ComponentCard title={t("records.actions")} desc={t("records.actionsDesc")}>
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              {selectedIds.length} {t("records.selected")} • {recentResults.length}{" "}
              {t("records.cachedResults")} •{" "}
              {stats ? `${stats.entities_count} ${t("records.entitiesInCollection")}` : "stats unavailable"}
            </div>
            <div className="flex flex-wrap gap-3">
              <Button variant="outline" onClick={() => void refreshStats()}>
                {t("records.refreshStats")}
              </Button>
              <Button variant="outline" onClick={clearSelection}>
                {t("records.clearSelection")}
              </Button>
              <Button onClick={() => void deleteSelected()} disabled={loading || selectedIds.length === 0}>
                {t("records.deleteSelected")}
              </Button>
            </div>
          </div>
        </ComponentCard>

        <div className="space-y-4">
          {recentResults.map((result) => (
            <ResultCard
              key={result.id}
              result={result}
              selected={selectedIds.includes(result.id)}
              onToggle={toggleSelected}
            />
          ))}
          {recentResults.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-gray-300 px-6 py-10 text-center text-sm text-gray-500 dark:border-gray-700 dark:text-gray-400">
              {t("records.empty")}
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
}
