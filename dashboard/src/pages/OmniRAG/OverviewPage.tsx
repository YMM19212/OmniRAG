import PageMeta from "../../components/common/PageMeta";
import ComponentCard from "../../components/common/ComponentCard";
import Button from "../../components/ui/button/Button";
import PageIntro from "../../components/omnirag/PageIntro";
import StatCard from "../../components/omnirag/StatCard";
import StatusBanner from "../../components/omnirag/StatusBanner";
import { useOmniRAG } from "../../context/OmniRAGContext";
import { useI18n } from "../../context/I18nContext";

export default function OverviewPage() {
  const { status, stats, recentResults, refreshStatus, refreshStats, loading } = useOmniRAG();
  const { t, locale } = useI18n();

  const capabilityKeys = [
    "overview.cap1",
    "overview.cap2",
    "overview.cap3",
    "overview.cap4",
    "overview.cap5",
    "overview.cap6",
  ];

  return (
    <>
      <PageMeta title={`OmniRAG | ${t("overview.title")}`} description={t("overview.desc")} />
      <PageIntro
        title={t("overview.title")}
        description={t("overview.desc")}
        eyebrow={t("overview.heroBadge")}
      />

      <div className="space-y-6">
        <section className="relative overflow-hidden rounded-[28px] border border-gray-200 bg-white p-6 shadow-theme-lg dark:border-gray-800 dark:bg-white/[0.03] md:p-8">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(70,95,255,0.18),transparent_38%),radial-gradient(circle_at_bottom_right,rgba(11,165,236,0.16),transparent_34%)]" />
          <div className="relative grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
            <div>
              <div className="mb-4 inline-flex rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-brand-600 dark:border-brand-500/20 dark:bg-brand-500/10 dark:text-brand-300">
                {t("overview.heroBadge")}
              </div>
              <h1 className="max-w-3xl text-3xl font-semibold leading-tight text-gray-950 dark:text-white md:text-5xl">
                {t("overview.heroTitle")}
              </h1>
              <p className="mt-5 max-w-2xl text-base leading-7 text-gray-600 dark:text-gray-300">
                {t("overview.heroBody")}
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <Button variant="outline" onClick={() => void refreshStatus()}>
                  {t("common.refreshStatus")}
                </Button>
                <Button onClick={() => void refreshStats()} disabled={loading || !status.ready}>
                  {t("common.refreshStats")}
                </Button>
              </div>
            </div>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-1">
              <div className="rounded-2xl border border-white/70 bg-white/80 p-5 backdrop-blur dark:border-white/10 dark:bg-gray-900/70">
                <p className="text-xs uppercase tracking-[0.22em] text-gray-400">
                  {locale === "zh" ? "工作区状态" : "Workspace state"}
                </p>
                <p className="mt-3 text-xl font-semibold text-gray-900 dark:text-white">
                  {status.message}
                </p>
                <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                  {stats
                    ? `${stats.collection_name} • ${stats.metric_type}`
                    : locale === "zh"
                      ? "等待初始化后显示实时数据。"
                      : "Live metrics appear after initialization."}
                </p>
              </div>
              <div className="rounded-2xl border border-white/70 bg-gray-950 p-5 text-white shadow-theme-md">
                <p className="text-xs uppercase tracking-[0.22em] text-white/50">
                  {locale === "zh" ? "当前焦点" : "Current focus"}
                </p>
                <p className="mt-3 text-xl font-semibold">
                  {locale === "zh" ? "多模态知识摄取与检索" : "Multimodal ingestion and retrieval"}
                </p>
                <p className="mt-2 text-sm text-white/70">
                  {locale === "zh"
                    ? "把配置、导入、检索和治理收敛在一个可持续维护的控制台里。"
                    : "Unify setup, ingestion, retrieval and governance in one maintainable control plane."}
                </p>
              </div>
            </div>
          </div>
        </section>

        <StatusBanner status={status} />

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard
            label={t("overview.collection")}
            value={stats?.collection_name ?? t("overview.notReady")}
            hint={t("overview.currentCollection")}
            accent="from-brand-500/20 to-transparent"
          />
          <StatCard
            label={t("overview.entities")}
            value={stats?.entities_count ?? 0}
            hint={t("overview.recordHint")}
            accent="from-blue-light-500/20 to-transparent"
          />
          <StatCard
            label={t("overview.vectorDim")}
            value={stats?.vector_dim ?? 0}
            hint={t("overview.vectorHint")}
            accent="from-success-500/20 to-transparent"
          />
          <StatCard
            label={t("overview.index")}
            value={stats?.index_type ?? "Unknown"}
            hint={stats?.metric_type ? `Metric: ${stats.metric_type}` : t("overview.awaitingInit")}
            accent="from-warning-500/20 to-transparent"
          />
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
          <ComponentCard title={t("overview.capabilities")} desc={t("overview.capabilitiesDesc")}>
            <div className="grid gap-4 md:grid-cols-2">
              {capabilityKeys.map((key) => (
                <div
                  key={key}
                  className="rounded-2xl border border-gray-200 bg-gray-50 px-4 py-4 text-sm text-gray-700 shadow-theme-xs dark:border-gray-800 dark:bg-gray-800/60 dark:text-gray-300"
                >
                  {t(key)}
                </div>
              ))}
            </div>
          </ComponentCard>

          <ComponentCard title={t("overview.recent")} desc={t("overview.recentDesc")}>
            <div className="space-y-3">
              {recentResults.slice(0, 5).map((result) => (
                <div
                  key={result.id}
                  className="rounded-2xl border border-gray-200 px-4 py-3 dark:border-gray-800"
                >
                  <p className="truncate font-mono text-sm text-gray-900 dark:text-white">
                    {result.id}
                  </p>
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    {(result.modality ?? t("result.unknown"))}
                    {result.source ? ` • ${result.source}` : ""}
                  </p>
                </div>
              ))}
              {recentResults.length === 0 ? (
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {t("overview.recentEmpty")}
                </p>
              ) : null}
            </div>
          </ComponentCard>
        </div>
      </div>
    </>
  );
}
