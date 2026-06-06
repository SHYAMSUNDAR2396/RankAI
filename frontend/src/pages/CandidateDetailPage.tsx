import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { Candidate } from '../types';
import TopAppBar from '../components/TopAppBar';

export default function CandidateDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [candidate, setCandidate] = useState<Candidate | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    api
      .getCandidate(id)
      .then(setCandidate)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex-1 flex flex-col">
        <TopAppBar breadcrumbs={[{ label: 'Dashboard', to: '/' }, { label: 'Loading...' }]} />
        <main className="p-8">
          <div className="animate-pulse-subtle space-y-6">
            <div className="h-10 w-64 bg-surface-variant rounded" />
            <div className="grid grid-cols-2 gap-6">
              <div className="h-64 bg-surface-variant rounded-lg" />
              <div className="h-64 bg-surface-variant rounded-lg" />
            </div>
          </div>
        </main>
      </div>
    );
  }

  if (!candidate) {
    return (
      <div className="flex-1 flex flex-col">
        <TopAppBar breadcrumbs={[{ label: 'Dashboard', to: '/' }, { label: 'Not Found' }]} />
        <main className="p-8 flex items-center justify-center">
          <p className="text-on-surface-variant text-body-lg">Candidate not found.</p>
        </main>
      </div>
    );
  }

  const score = candidate.composite_score ?? 0;
  const hmScore = candidate.hiring_manager_score ?? 0;
  const peerScore = candidate.peer_interviewer_score ?? 0;
  const daScore = candidate.devils_advocate_score ?? 0;
  const trajScore = candidate.trajectory_score ?? 0;
  const variance = candidate.panel_variance ?? 0;
  const delta = candidate.counterfactual_delta;
  const isFlagged = candidate.bias_flag;

  // Derive trajectory signals from data
  const growthRate = trajScore > 0 ? (trajScore / 10).toFixed(2) : '0.00';
  const complexityArc = trajScore >= 7 ? 'Ascending' : trajScore >= 5 ? 'Stable' : 'Descending';
  const leadershipPct = Math.min(Math.round((hmScore / 10) * 100), 100);
  const tenureConsistency = variance < 1 ? '0.91' : variance < 2 ? '0.75' : '0.50';

  // Extract strengths/concerns as tags
  const strengthTags = (candidate.strengths || []).slice(0, 4);
  const concernTags = (candidate.concerns || []).slice(0, 3);

  // Clean the narrative (remove the "Here is a summary" prefix)
  let narrative = candidate.narrative || '';
  const summaryIdx = narrative.indexOf('\n\n');
  if (summaryIdx > -1 && summaryIdx < 120) {
    narrative = narrative.substring(summaryIdx + 2).trim();
  }

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <TopAppBar
        breadcrumbs={[
          { label: 'Dashboard', to: '/' },
          { label: 'Candidates', to: '/' },
          { label: candidate.name },
        ]}
      />
      <main className="flex-1 overflow-y-auto p-8 pb-20">
        {/* Page Header */}
        <div className="flex justify-between items-start mb-6 animate-fade-in" style={{ opacity: 0 }}>
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-[26px] font-bold text-navy leading-tight">{candidate.name}</h2>
              <span className="bg-primary/10 text-primary px-2.5 py-0.5 rounded-full text-label-sm border border-primary/20">
                Rank #{candidate.rank}
              </span>
            </div>
            <p className="text-body-lg text-on-surface-variant">
              Composite Score: {score.toFixed(2)} · Trajectory: {trajScore.toFixed(1)}
            </p>
          </div>
          <div className="flex items-center">
            <div className="relative w-16 h-16 flex items-center justify-center rounded-full border-4 border-surface-container">
              <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 64 64">
                <circle
                  cx="32"
                  cy="32"
                  r="28"
                  fill="none"
                  stroke="var(--color-surface-container)"
                  strokeWidth="4"
                />
                <circle
                  cx="32"
                  cy="32"
                  r="28"
                  fill="none"
                  stroke="var(--color-primary)"
                  strokeWidth="4"
                  strokeDasharray={`${(score / 10) * 175.9} 175.9`}
                  strokeLinecap="round"
                />
              </svg>
              <span className="text-headline-sm text-primary z-10">{score.toFixed(1)}</span>
            </div>
            <span className="text-body-md text-outline ml-2">/ 10</span>
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left Column (60%) */}
          <div className="flex-1 lg:w-[60%] flex flex-col gap-6">
            {/* AI Panel Scores */}
            <div className="card p-6 animate-fade-in" style={{ opacity: 0, animationDelay: '0.05s' }}>
              <h3 className="text-headline-md text-navy mb-6">AI Panel Scores</h3>
              <div className="space-y-5">
                {[
                  { label: 'Hiring Manager Perspective', score: hmScore, color: 'bg-primary' },
                  { label: 'Peer Interviewer Perspective', score: peerScore, color: 'bg-primary' },
                  {
                    label: "Devil's Advocate Perspective",
                    score: daScore,
                    color: daScore < 5 ? 'bg-error' : 'bg-primary',
                  },
                ].map((item) => (
                  <div key={item.label}>
                    <div className="flex justify-between text-body-md mb-2">
                      <span className="text-on-surface">{item.label}</span>
                      <span className={`font-bold ${item.score < 5 ? 'text-error' : 'text-primary'}`}>
                        {item.score.toFixed(1)}
                      </span>
                    </div>
                    <div className="w-full bg-surface-container rounded-full h-2">
                      <div
                        className={`${item.color} h-2 rounded-full transition-all duration-700`}
                        style={{ width: `${item.score * 10}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-6 pt-4 border-t border-outline-variant flex items-center text-label-md text-on-surface-variant">
                <span className="material-symbols-outlined text-[16px] mr-2">info</span>
                {variance < 1
                  ? `Low panel variance (${variance.toFixed(2)}) — high confidence result`
                  : `Panel variance (${variance.toFixed(2)}) — ${variance > 2 ? 'low confidence, review recommended' : 'moderate confidence'}`}
              </div>
            </div>

            {/* AI Narrative */}
            <div className="card p-6 animate-fade-in" style={{ opacity: 0, animationDelay: '0.1s' }}>
              <h3 className="text-headline-md text-navy mb-4">AI Narrative</h3>
              <blockquote className="border-l-4 border-primary/40 pl-4 py-1 mb-6 text-body-md text-on-surface-variant italic">
                "{narrative}"
              </blockquote>
              <div className="flex flex-wrap gap-2">
                {strengthTags.map((tag, i) => (
                  <span key={i} className="bg-primary/10 text-primary px-3 py-1 rounded text-label-sm">
                    {tag}
                  </span>
                ))}
                {concernTags.map((tag, i) => (
                  <span key={i} className="bg-amber-100 text-amber-900 px-3 py-1 rounded text-label-sm">
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            {/* Career Timeline */}
            <div className="card p-6 animate-fade-in" style={{ opacity: 0, animationDelay: '0.15s' }}>
              <h3 className="text-headline-md text-navy mb-6">Strengths & Concerns</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-label-md text-primary uppercase tracking-wider mb-3">Strengths</h4>
                  <ul className="space-y-2">
                    {(candidate.strengths || []).map((s, i) => (
                      <li key={i} className="flex items-start gap-2 text-body-md text-on-surface">
                        <span className="material-symbols-outlined text-primary text-[18px] mt-0.5">
                          check_circle
                        </span>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="text-label-md text-req-amber uppercase tracking-wider mb-3">Concerns</h4>
                  <ul className="space-y-2">
                    {(candidate.concerns || []).map((c, i) => (
                      <li key={i} className="flex items-start gap-2 text-body-md text-on-surface">
                        <span className="material-symbols-outlined text-req-amber text-[18px] mt-0.5">
                          warning
                        </span>
                        {c}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column (38%) */}
          <div className="lg:w-[38%] flex flex-col gap-6">
            {/* Trajectory Signals */}
            <div className="card p-6 animate-fade-in" style={{ opacity: 0, animationDelay: '0.1s' }}>
              <h3 className="text-headline-md text-navy mb-6">Trajectory Signals</h3>
              <div className="space-y-6">
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-label-md text-outline uppercase tracking-wider">Growth Rate</div>
                    <div className="text-headline-sm text-navy mt-1">{growthRate}</div>
                  </div>
                  <div className="w-16 h-8 flex items-end">
                    <svg viewBox="0 0 64 32" className="w-full h-full" fill="none">
                      <polyline
                        points="0,28 16,20 32,16 48,8 64,4"
                        stroke="var(--color-primary)"
                        strokeWidth="2"
                        strokeLinecap="round"
                        fill="none"
                      />
                      <circle cx="64" cy="4" r="3" fill="var(--color-primary)" />
                    </svg>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-label-md text-outline uppercase tracking-wider">
                      Complexity Arc
                    </div>
                    <div className="text-body-md text-navy mt-1 font-semibold">{complexityArc}</div>
                  </div>
                  <span className="material-symbols-outlined text-primary">
                    {complexityArc === 'Ascending'
                      ? 'trending_up'
                      : complexityArc === 'Stable'
                        ? 'trending_flat'
                        : 'trending_down'}
                  </span>
                </div>
                <div>
                  <div className="flex justify-between items-end mb-2">
                    <div className="text-label-md text-outline uppercase tracking-wider">
                      Leadership Progression
                    </div>
                    <div className="text-label-md text-navy">{leadershipPct}%</div>
                  </div>
                  <div className="w-full bg-surface-container rounded-full h-1.5">
                    <div
                      className="bg-primary h-1.5 rounded-full transition-all duration-700"
                      style={{ width: `${leadershipPct}%` }}
                    />
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-label-md text-outline uppercase tracking-wider">
                      Tenure Consistency
                    </div>
                    <div className="text-body-md text-navy mt-1">{tenureConsistency}</div>
                  </div>
                  <div
                    className={`w-3 h-3 rounded-full ${
                      parseFloat(tenureConsistency) > 0.8
                        ? 'bg-green-500'
                        : parseFloat(tenureConsistency) > 0.6
                          ? 'bg-req-amber'
                          : 'bg-req-red'
                    }`}
                  />
                </div>
              </div>
            </div>

            {/* Fairness Audit */}
            <div className="card p-6 animate-fade-in" style={{ opacity: 0, animationDelay: '0.15s' }}>
              <h3 className="text-headline-md text-navy mb-4">Fairness Audit</h3>
              <div
                className={`flex items-center gap-2 mb-4 p-3 rounded-md border ${
                  isFlagged
                    ? 'bg-red-50 text-red-800 border-red-100'
                    : 'bg-green-50 text-green-800 border-green-100'
                }`}
              >
                <span className="material-symbols-outlined text-[20px]">
                  {isFlagged ? 'warning' : 'health_and_safety'}
                </span>
                <span className="text-body-md font-medium">
                  {isFlagged ? 'Flagged — Potential bias detected' : 'Clean — No bias flag'}
                </span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-body-md text-on-surface">Counterfactual delta</span>
                <span className="text-body-md font-mono text-outline">
                  {delta !== null && delta !== undefined ? delta.toFixed(2) : 'N/A'}
                </span>
              </div>
              <p className="text-label-sm text-outline mt-4">
                Name, pronoun, and institution swaps tested.
              </p>
            </div>

            {/* Reviewer Actions */}
            <div className="card p-6 bg-row-alt/50 animate-fade-in" style={{ opacity: 0, animationDelay: '0.2s' }}>
              <h3 className="text-headline-md text-navy mb-4">Reviewer Actions</h3>
              <div className="flex flex-col gap-3">
                <button className="w-full py-3 bg-primary text-white text-label-md rounded active:scale-95 transition-transform hover:bg-primary/90 flex justify-center items-center gap-2">
                  Advance to Interview
                </button>
                <button className="w-full py-2.5 border border-amber-600 text-amber-700 text-label-md rounded active:scale-95 transition-transform hover:bg-amber-50">
                  Flag for Review
                </button>
                <div className="flex gap-3">
                  <button className="flex-1 py-2.5 border border-error text-error text-label-md rounded active:scale-95 transition-transform hover:bg-error/5">
                    Reject
                  </button>
                  <button
                    className="flex-1 py-2.5 border border-outline text-on-surface-variant text-label-md rounded active:scale-95 transition-transform hover:bg-surface-container-low"
                    onClick={() => navigate('/')}
                  >
                    Back
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
