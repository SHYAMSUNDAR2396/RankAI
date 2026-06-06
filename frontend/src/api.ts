/* ── API Client ── */

import type {
  CandidatesResponse,
  Candidate,
  AuditReport,
  JobDescription,
  PipelineStatus,
} from './types';

const BASE = '/api';

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  getCandidates: () => fetchJSON<CandidatesResponse>(`${BASE}/candidates`),

  getCandidate: (id: string) => fetchJSON<Candidate>(`${BASE}/candidates/${id}`),

  getAudit: () => fetchJSON<AuditReport>(`${BASE}/audit`),

  getJobDescription: () => fetchJSON<JobDescription>(`${BASE}/job-description`),

  getPipelineStatus: () => fetchJSON<PipelineStatus>(`${BASE}/pipeline/status`),

  runPipeline: () =>
    fetchJSON<{ message: string }>(`${BASE}/run`, { method: 'POST' }),

  uploadResumes: async (files: File[]) => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    return fetchJSON<{ uploaded: string[] }>(`${BASE}/upload/resumes`, {
      method: 'POST',
      body: formData,
    });
  },

  uploadJobDescription: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJSON<{ uploaded: string }>(`${BASE}/upload/job-description`, {
      method: 'POST',
      body: formData,
    });
  },

  exportCSV: () => {
    window.open(`${BASE}/export/csv`, '_blank');
  },
};
