import PageBreadcrumb from "../common/PageBreadCrumb";

interface PageIntroProps {
  title: string;
  description: string;
  eyebrow?: string;
}

export default function PageIntro({ title, description, eyebrow }: PageIntroProps) {
  return (
    <div className="mb-6">
      <PageBreadcrumb pageTitle={title} />
      {eyebrow ? (
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.24em] text-brand-500">
          {eyebrow}
        </p>
      ) : null}
      <p className="max-w-3xl text-sm text-gray-500 dark:text-gray-400">{description}</p>
    </div>
  );
}
