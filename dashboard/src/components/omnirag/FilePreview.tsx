export default function FilePreview({
  file,
  kind,
}: {
  file: File | null;
  kind: "image" | "video";
}) {
  if (!file) return null;
  const url = URL.createObjectURL(file);

  return (
    <div className="rounded-2xl border border-dashed border-gray-300 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800/50">
      <p className="mb-3 text-sm font-medium text-gray-700 dark:text-gray-300">
        Selected {kind}: {file.name}
      </p>
      {kind === "image" ? (
        <img src={url} alt={file.name} className="max-h-56 rounded-xl object-cover" />
      ) : (
        <video src={url} controls className="max-h-56 rounded-xl" />
      )}
    </div>
  );
}
