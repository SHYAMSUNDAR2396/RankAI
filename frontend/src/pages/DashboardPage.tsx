import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { Candidate } from '../types';
import TopAppBar from '../components/TopAppBar';
import EmptyStatePage from './EmptyStatePage';

function getVerdictBadge(verdict: string) {
  const v = verdict?.toLowerCase().replace(/[^a-z_]/g, '');
  switch (v) {
    case 'strong_yes':
      return <span className="badge badge-strong-yes">Strong Yes</span>;
    case 'yes':
      return <span className="badge badge-yes">Yes</span>;
    case 'maybe':
      return <span className="badge badge-maybe">Maybe</span>;
    case 'no':
      return <span className="badge badge-no">No</span>;
    default:
      return <span className="badge bg-surface-variant text-on-surface-variant">{verdict}</span>;
  }
}

export default function DashboardPage() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const pageSize = 8;
  const navigate = useNavigate();

  useEffect(() => {
    api
      .getCandidates()
      .then((data) => setCandidates(data.candidates))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const totalCandidates = candidates.length;
  const requireReview = candidates.filter((c) => c.requires_human_review).length;
  const biasFlags = candidates.filter((c) => c.bias_flag).length;
  const avgScore = totalCandidates
    ? candidates.reduce((s, c) => s + (c.composite_score ?? 0), 0) / totalCandidates
    : 0;
  const pipelineConfidence = Math.round((avgScore / 10) * 100);

  const pagedCandidates = candidates.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(totalCandidates / pageSize);

  const stats = [
    { label: 'Candidates Ranked', value: totalCandidates, color: 'text-on-surface' },
    { label: 'Require Human Review', value: requireReview, color: 'text-req-amber' },
    { label: 'Bias Flags Raised', value: biasFlags, color: 'text-req-red' },
    { label: 'Pipeline Confidence', value: `${pipelineConfidence}%`, color: 'text-primary-container' },
  ];

  if (loading) {
    return (
      <div className="flex-1 flex flex-col">
        <TopAppBar />
        <main className="p-8 flex flex-col gap-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="card p-6 animate-pulse-subtle">
                <div className="h-3 w-24 bg-surface-variant rounded mb-3" />
                <div className="h-8 w-16 bg-surface-variant rounded" />
              </div>
            ))}
          </div>
          <div className="card p-6 animate-pulse-subtle">
            <div className="h-5 w-40 bg-surface-variant rounded mb-4" />
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-10 bg-surface-variant rounded" />
              ))}
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (candidates.length === 0) {
    return (
      <div className="flex-1 flex flex-col animate-fade-in">
        <TopAppBar />
        <EmptyStatePage />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      <TopAppBar />
      <main className="p-8 flex flex-col gap-6">
        {/* Stats Row */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, i) => (
            <div
              key={stat.label}
              className={`card p-6 flex flex-col gap-2 animate-fade-in animate-fade-in-delay-${i + 1}`}
              style={{ opacity: 0 }}
            >
              <span className="text-label-sm text-on-surface-variant uppercase tracking-wider">
                {stat.label}
              </span>
              <span className={`text-display-lg ${stat.color}`}>{stat.value}</span>
            </div>
          ))}
        </section>

        {/* Ranked Candidates Table */}
        <section className="card overflow-hidden flex flex-col animate-fade-in" style={{ opacity: 0, animationDelay: '0.25s' }}>
          <div className="p-4 border-b border-card-border">
            <h2 className="text-headline-sm text-on-surface">Ranked Candidates</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-lowest border-b border-outline-variant">
                  <th className="py-3 px-4 text-label-md text-secondary uppercase w-16">Rank</th>
                  <th className="py-3 px-4 text-label-md text-secondary uppercase">Name</th>
                  <th className="py-3 px-4 text-label-md text-secondary uppercase w-48">Score</th>
                  <th className="py-3 px-4 text-label-md text-secondary uppercase">Verdict</th>
                  <th className="py-3 px-4 text-label-md text-secondary uppercase w-24 text-center">
                    Bias Flag
                  </th>
                  <th className="py-3 px-4 text-label-md text-secondary uppercase w-24 text-center">
                    Data Gap
                  </th>
                  <th className="py-3 px-4 text-label-md text-secondary uppercase w-24 text-right">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="text-body-md text-on-surface">
                {pagedCandidates.map((c, i) => {
                  const score = c.composite_score ?? 0;
                  const pct = Math.min(Math.round(score * 10), 100);
                  const isAlt = i % 2 === 1;
                  const hasDataGap =
                    (c.panel_variance ?? 0) > 2 || (c.concerns?.length ?? 0) > 4;

                  return (
                    <tr
                      key={c.candidate_id}
                      className={`table-row-hover border-b border-outline-variant/30 cursor-pointer ${
                        isAlt ? 'bg-row-alt' : 'bg-surface-container-lowest'
                      }`}
                      onClick={() => navigate(`/candidates/${c.candidate_id}`)}
                    >
                      <td className="py-3 px-4 font-bold">#{c.rank}</td>
                      <td className="py-3 px-4">{c.name}</td>
                      <td className="py-3 px-4">
                        <div className="flex flex-col gap-1 w-full max-w-[120px]">
                          <span>{score.toFixed(1)}/10</span>
                          <div className="progress-bar">
                            <div
                              className="progress-bar-fill bg-primary-container"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4">{getVerdictBadge(c.verdict_consensus)}</td>
                      <td className="py-3 px-4 text-center">
                        {c.bias_flag ? (
                          <span
                            className="material-symbols-outlined text-req-amber text-lg"
                            title="Bias Warning"
                          >
                            warning
                          </span>
                        ) : (
                          <span className="text-on-surface-variant">No</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-center">
                        {hasDataGap ? (
                          <span
                            className="material-symbols-outlined text-req-red text-lg"
                            title="Data Gap Flag"
                          >
                            flag
                          </span>
                        ) : (
                          <span className="text-on-surface-variant">No</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          className="text-primary-container hover:underline text-label-md"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/candidates/${c.candidate_id}`);
                          }}
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="p-4 border-t border-outline-variant flex justify-between items-center bg-surface-container-lowest">
            <span className="text-body-md text-on-surface-variant">
              Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, totalCandidates)} of{' '}
              {totalCandidates}
            </span>
            <div className="flex gap-2">
              <button
                className="p-1 text-on-surface-variant hover:bg-surface-container-low rounded transition-colors disabled:opacity-50"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                <span className="material-symbols-outlined">chevron_left</span>
              </button>
              <button
                className="p-1 text-on-surface hover:bg-surface-container-low rounded transition-colors disabled:opacity-50"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                <span className="material-symbols-outlined">chevron_right</span>
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
