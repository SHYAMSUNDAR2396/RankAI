/* ── RankAI Data Types ── */

export interface Candidate {
  rank: number | null;
  candidate_id: string;
  name: string;
  composite_score: number | null;
  trajectory_score: number | null;
  hiring_manager_score: number | null;
  peer_interviewer_score: number | null;
  devils_advocate_score: number | null;
  panel_variance: number | null;
  requires_human_review: boolean;
  verdict_consensus: string;
  strengths: string[];
  concerns: string[];
  narrative: string;
  bias_flag: boolean;
  counterfactual_delta: number | null;
}

export interface CandidatesResponse {
  candidates: Candidate[];
  total: number;
}

export interface FlaggedCandidate {
  candidate_id: string;
  name: string;
  delta: number | null;
  original_score: number | null;
  cf_score: number | null;
}

export interface AuditFailure {
  candidate_id: string;
  name: string;
  original_score: number | null;
  cf_score: number | null;
  delta: number | null;
  bias_flag: boolean;
  audit_failure: boolean;
  error: string;
}

export interface AuditReport {
  total_candidates_audited: number;
  flagged_count: number;
  flag_rate: number;
  bias_flag_threshold: number;
  flagged_candidates: FlaggedCandidate[];
  clean_candidates_count: number;
  methodology_note: string;
  audit_failures: AuditFailure[];
}

export interface JobDescription {
  job_id: string;
  title: string;
  company: string;
  raw_text: string;
  requirements: {
    text: string;
    bucket: string;
    dimension: string;
  }[];
}

export interface PipelineStatus {
  running: boolean;
  last_result: {
    returncode: number;
    stdout_tail?: string;
    stderr_tail?: string;
    error?: string;
  } | null;
}
