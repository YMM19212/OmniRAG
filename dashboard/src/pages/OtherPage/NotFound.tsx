import GridShape from "../../components/common/GridShape";
import { Link } from "react-router";
import PageMeta from "../../components/common/PageMeta";
import { useI18n } from "../../context/I18nContext";

export default function NotFound() {
  const { t } = useI18n();

  return (
    <>
      <PageMeta title={`OmniRAG | ${t("notfound.title")}`} description={t("notfound.desc")} />
      <div className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden p-6">
        <GridShape />
        <div className="mx-auto w-full max-w-[520px] rounded-[28px] border border-gray-200 bg-white/90 p-10 text-center shadow-theme-lg backdrop-blur dark:border-gray-800 dark:bg-gray-900/85">
          <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-3xl bg-brand-500 text-2xl font-semibold text-white">
            O
          </div>
          <h1 className="mb-4 text-5xl font-semibold text-gray-900 dark:text-white">404</h1>
          <p className="mb-8 text-base leading-7 text-gray-600 dark:text-gray-300">
            {t("notfound.desc")}
          </p>
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-lg border border-gray-300 bg-white px-5 py-3.5 text-sm font-medium text-gray-700 shadow-theme-xs hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300"
          >
            {t("notfound.back")}
          </Link>
        </div>
      </div>
    </>
  );
}
