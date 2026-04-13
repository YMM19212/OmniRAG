import Badge from "../ui/badge/Badge";
import type { KBStatus } from "../../types/omnirag";
import { useI18n } from "../../context/I18nContext";

const colorMap = {
  disconnected: "warning",
  connecting: "info",
  ready: "success",
  error: "error",
} as const;

export default function StatusBanner({
  status,
  extra,
}: {
  status: KBStatus;
  extra?: React.ReactNode;
}) {
  const { t } = useI18n();
  const labelMap = {
    disconnected: t("status.disconnected"),
    connecting: t("status.connecting"),
    ready: t("status.ready"),
    error: t("status.error"),
  };
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-theme-xs dark:border-gray-800 dark:bg-white/[0.03]">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-3">
            <Badge color={colorMap[status.state]}>{labelMap[status.state]}</Badge>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {t("brand.subtitle")}
            </span>
          </div>
          <p className="text-base font-medium text-gray-800 dark:text-white/90">
            {status.message}
          </p>
          {status.last_error ? (
            <p className="mt-2 text-sm text-error-600 dark:text-error-400">{status.last_error}</p>
          ) : null}
        </div>
        {extra ? <div className="shrink-0">{extra}</div> : null}
      </div>
    </div>
  );
}
