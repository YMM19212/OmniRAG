interface StatCardProps {
  label: string;
  value: string | number;
  hint: string;
  accent?: string;
}

export default function StatCard({ label, value, hint, accent = "from-brand-500/15 to-blue-light-400/5" }: StatCardProps) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-gray-200 bg-white p-5 shadow-theme-xs dark:border-gray-800 dark:bg-white/[0.03]">
      <div className={`pointer-events-none absolute inset-x-0 top-0 h-20 bg-gradient-to-br ${accent}`} />
      <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
      <p className="relative mt-3 text-3xl font-semibold tracking-tight text-gray-900 dark:text-white">
        {value}
      </p>
      <p className="relative mt-2 text-sm text-gray-500 dark:text-gray-400">{hint}</p>
    </div>
  );
}
