import { api } from '../api';

interface Props {
  breadcrumbs?: { label: string; to?: string }[];
}

export default function TopAppBar({ breadcrumbs }: Props) {
  const handleRunPipeline = async () => {
    try {
      await api.runPipeline();
      alert('Pipeline started! Check status on the Dashboard.');
    } catch (err) {
      alert(`Failed to start pipeline: ${err}`);
    }
  };

  return (
    <header className="bg-surface border-b border-outline-variant sticky top-0 z-10 flex justify-between items-center w-full px-8 py-4 h-16">
      <div className="flex-1">
        {breadcrumbs && breadcrumbs.length > 0 ? (
          <nav className="flex items-center gap-2 text-label-md text-on-surface-variant">
            {breadcrumbs.map((crumb, i) => (
              <span key={i} className="flex items-center gap-2">
                {i > 0 && <span className="text-outline">/</span>}
                {crumb.to ? (
                  <a href={crumb.to} className="hover:text-primary transition-colors">
                    {crumb.label}
                  </a>
                ) : (
                  <span className="text-navy font-semibold">{crumb.label}</span>
                )}
              </span>
            ))}
          </nav>
        ) : (
          <span className="hidden">RankAI</span>
        )}
      </div>
      <div className="flex items-center gap-4">
        <button
          onClick={() => api.exportCSV()}
          className="px-4 py-2 text-primary-container border border-primary-container rounded text-label-md active:scale-95 transition-transform hover:bg-surface-container-low"
        >
          Export CSV
        </button>
        <button
          onClick={handleRunPipeline}
          className="px-4 py-2 bg-primary-container text-on-primary rounded text-label-md active:scale-95 transition-transform hover:bg-primary-container/90 shadow-none"
        >
          Run New Pipeline
        </button>
      </div>
    </header>
  );
}
