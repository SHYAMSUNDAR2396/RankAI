import { useEffect, useState } from 'react';
import { api } from '../api';
import type { AuditReport } from '../types';
import TopAppBar from '../components/TopAppBar';

export default function AuditReportPage() {
  const [audit, setAudit] = useState<AuditReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [showClean, setShowClean] = useState(false);

  useEffect(() => {
    api
      .getAudit()
      .then(setAudit)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex flex-col">
        <TopAppBar breadcrumbs={[{ label: 'Audit Report' }]} />
        <main className="p-8">
          <div className="animate-pulse-subtle space-y-6">
            <div className="h-8 w-48 bg-surface-variant rounded" />
            <div className="grid grid-cols-4 gap-6">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-20 bg-surface-variant rounded-lg" />
              ))}
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (!audit) {
    return (
      <div className="flex-1 flex flex-col">
        <TopAppBar breadcrumbs={[{ label: 'Audit Report' }]} />
        <main className="p-8 flex items-center justify-center">
          <p className="text-on-surface-variant text-body-lg">No audit data available.</p>
        </main>
      </div>
    );
  }

  const totalAudited = audit.total_candidates_audited + (audit.audit_failures?.length ?? 0);
  const cleanCount = audit.clean_candidates_count;
  const flagRate = ((audit.flagged_count / Math.max(totalAudited, 1)) * 100).toFixed(1);

  // Build clean candidates list from audit_failures that are not flagged
  const cleanCandidates = (audit.audit_failures || []).filter((f) => !f.bias_flag);

  const stats = [
    { label: 'Candidates Audited', value: totalAudited, color: 'text-navy' },
    { label: 'Bias Flags Raised', value: audit.flagged_count, color: 'text-error' },
    { label: 'Clean Results', value: cleanCount, color: 'text-primary-container' },
    { label: 'Flag Rate', value: `${flagRate}%`, color: 'text-tertiary-container' },
  ];

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <TopAppBar />
      <main className="flex-1 overflow-y-auto p-8">
        {/* Header */}
        <div className="mb-6 animate-fade-in" style={{ opacity: 0 }}>
          <h2 className="text-headline-md font-bold text-navy mb-2">Fairness Audit Report</h2>
          <p className="text-body-md text-on-surface-variant">
            Counterfactual fairness analysis · {totalAudited} candidates processed
          </p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          {stats.map((stat, i) => (
            <div
              key={stat.label}
              className={`card p-4 animate-fade-in animate-fade-in-delay-${i + 1}`}
              style={{ opacity: 0 }}
            >
              <p className="text-label-md text-on-surface-variant uppercase mb-2">{stat.label}</p>
              <p className={`text-display-lg ${stat.color}`}>{stat.value}</p>
            </div>
          ))}
        </div>

        {/* Two Column Layout */}
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left Column (60%) */}
          <div className="w-full lg:w-[60%] flex flex-col gap-6">
            {/* Flagged Candidates Table */}
            <div className="card overflow-hidden flex flex-col animate-fade-in" style={{ opacity: 0, animationDelay: '0.2s' }}>
              <div className="p-4 border-b border-card-border">
                <h3 className="text-headline-sm font-semibold text-navy">Flagged Candidates</h3>
              </div>
              <div className="w-full overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-card-border text-label-md uppercase bg-background">
                      <th className="p-4 font-semibold w-1"></th>
                      <th className="p-4 font-semibold">Candidate</th>
                      <th className="p-4 font-semibold">Original Score</th>
                      <th className="p-4 font-semibold">CF Score</th>
                      <th className="p-4 font-semibold">Delta</th>
                      <th className="p-4 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {audit.flagged_candidates.map((fc, i) => (
                      <tr
                        key={fc.candidate_id}
                        className={`border-b border-card-border relative group table-row-hover ${
                          i % 2 === 1 ? 'bg-row-alt' : ''
                        }`}
                      >
                        <td className="w-1 p-0">
                          <div className="bg-error w-1 h-full opacity-0 group-hover:opacity-100 transition-opacity" />
                        </td>
                        <td className="p-4 text-body-md font-medium text-navy">{fc.name}</td>
                        <td className="p-4 text-body-md text-on-surface-variant">
                          {fc.original_score?.toFixed(1) ?? 'N/A'}
                        </td>
                        <td className="p-4 text-body-md text-on-surface-variant">
                          {fc.cf_score?.toFixed(2) ?? 'N/A'}
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            <span className="text-body-md font-medium text-error">
                              {fc.delta?.toFixed(2) ?? 'N/A'}
                            </span>
                            {fc.delta != null && (
                              <div className="w-16 h-1.5 bg-surface-variant rounded-full overflow-hidden">
                                <div
                                  className="bg-error h-full"
                                  style={{ width: `${Math.min(fc.delta * 100, 100)}%` }}
                                />
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="p-4">
                          <span className="badge badge-flagged">Flagged</span>
                        </td>
                      </tr>
                    ))}
                    {audit.flagged_candidates.length === 0 && (
                      <tr>
                        <td colSpan={6} className="p-4 text-center text-on-surface-variant text-body-md">
                          No flagged candidates.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              {audit.flagged_candidates.length > 0 && (
                <div className="p-4 bg-amber-50 border-t border-card-border">
                  <p className="text-body-md text-tertiary flex items-center gap-2">
                    <span className="material-symbols-outlined text-[18px]">info</span>
                    Candidates highlighted above showed a scoring delta &gt;{' '}
                    {audit.bias_flag_threshold} across counterfactual variations, indicating potential
                    bias in evaluation. Manual review required.
                  </p>
                </div>
              )}
            </div>

            {/* Clean Candidates (Collapsible) */}
            <button
              onClick={() => setShowClean((v) => !v)}
              className="w-full card p-4 flex justify-between items-center hover:bg-surface-container-low transition-colors animate-fade-in"
              style={{ opacity: 0, animationDelay: '0.25s' }}
            >
              <span className="text-headline-sm font-semibold text-navy">
                Clean Candidates ({cleanCandidates.length})
              </span>
              <span className="material-symbols-outlined text-on-surface-variant">
                {showClean ? 'expand_less' : 'expand_more'}
              </span>
            </button>

            {showClean && (
              <div className="card overflow-hidden">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-card-border text-label-md uppercase bg-background">
                      <th className="p-4 font-semibold">Candidate</th>
                      <th className="p-4 font-semibold">Score</th>
                      <th className="p-4 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cleanCandidates.map((c, i) => (
                      <tr
                        key={c.candidate_id}
                        className={`border-b border-card-border/50 ${i % 2 === 1 ? 'bg-row-alt' : ''}`}
                      >
                        <td className="p-4 text-body-md text-navy">{c.name}</td>
                        <td className="p-4 text-body-md text-on-surface-variant">
                          {c.original_score?.toFixed(1) ?? 'N/A'}
                        </td>
                        <td className="p-4">
                          {c.audit_failure ? (
                            <span className="badge bg-surface-variant text-on-surface-variant">
                              Audit Error
                            </span>
                          ) : (
                            <span className="badge bg-green-100 text-green-800">Clean</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Right Column (38%) */}
          <div className="w-full lg:w-[38%] flex flex-col gap-6">
            {/* What was tested */}
            <div className="card p-5 animate-fade-in" style={{ opacity: 0, animationDelay: '0.15s' }}>
              <h3 className="text-headline-sm font-semibold text-navy mb-4">What was tested</h3>
              <div className="flex flex-col gap-3">
                {['Name Swaps', 'Pronoun Variations', 'Institution Masking'].map((test) => (
                  <div key={test} className="flex items-center gap-3">
                    <span className="material-symbols-outlined text-primary-container">check_circle</span>
                    <span className="text-body-md text-navy">{test}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Threshold Analysis */}
            <div className="card p-5 animate-fade-in" style={{ opacity: 0, animationDelay: '0.2s' }}>
              <h3 className="text-headline-sm font-semibold text-navy mb-6">Threshold Analysis</h3>
              <div className="relative pt-6 pb-2">
                {/* Gradient Axis */}
                <div className="h-1 bg-linear-to-r from-primary-fixed-dim via-surface-variant to-error-container rounded-full w-full" />
                {/* Threshold Line */}
                <div
                  className="absolute top-2 bottom-0 w-px bg-outline border-r border-dashed border-outline"
                  style={{ left: '75%' }}
                />
                <span
                  className="absolute top-0 text-label-sm text-outline -translate-x-1/2"
                  style={{ left: '75%' }}
                >
                  {audit.bias_flag_threshold} Limit
                </span>

                {/* Data Points */}
                {audit.flagged_candidates.map((fc) => {
                  const pos = Math.min((fc.delta ?? 0) * 100, 100);
                  return (
                    <div
                      key={fc.candidate_id}
                      className="absolute top-1/2 w-4 h-4 rounded-full bg-error border-2 border-surface-container-lowest -translate-x-1/2 -translate-y-1/2 group"
                      style={{ left: `${pos}%` }}
                    >
                      <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 bg-inverse-surface text-inverse-on-surface text-label-sm px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10 pointer-events-none">
                        {fc.name} ({fc.delta?.toFixed(2)})
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="flex justify-between mt-4 text-label-md text-on-surface-variant uppercase">
                <span>Safe Zone</span>
                <span>Review Zone</span>
              </div>
            </div>

            {/* Methodology Note */}
            <div className="card p-5 animate-fade-in" style={{ opacity: 0, animationDelay: '0.25s' }}>
              <h3 className="text-headline-sm font-semibold text-navy mb-3">Methodology</h3>
              <p className="text-body-md text-on-surface-variant leading-relaxed">
                {audit.methodology_note}
              </p>
            </div>

            {/* Actions */}
            <div className="flex gap-4 animate-fade-in" style={{ opacity: 0, animationDelay: '0.3s' }}>
              <button
                onClick={() => {
                  window.open('/api/export/csv', '_blank');
                }}
                className="flex-1 border border-primary-container text-primary-container hover:bg-surface-container-low transition-colors px-4 py-3 rounded text-label-md text-center active:scale-95 transition-transform"
              >
                Download Full Report
              </button>
              <button
                onClick={async () => {
                  try {
                    await api.runPipeline();
                    alert('Audit re-run started!');
                  } catch {
                    alert('Failed to start audit');
                  }
                }}
                className="flex-1 bg-primary-container text-on-primary hover:bg-surface-tint transition-colors px-4 py-3 rounded text-label-md text-center active:scale-95 transition-transform"
              >
                Re-run Audit
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
