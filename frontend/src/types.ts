export type NoticeTone = "info" | "success" | "warning" | "error";

export interface JobFormState {
  title: string;
  company: string;
  salary: string;
  job_type: string;
  skills: string;
  requirements: string;
  description: string;
  HRId: string;
  questions: string[];
}

export interface CandidateFormState {
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  gender: string;
  degree: string;
}

export interface HrFormState {
  name?: string;
  email: string;
  password: string;
}

export interface JobQuestionResponse {
  id: number;
  job_id: number;
  question: string;
}

export interface JobListingResponse {
  id: number;
  title: string;
  company: string;
  salary: number;
  job_type: string;
  description: string;
}

export interface JobListResponse {
  jobs: JobListingResponse[];
}

export interface JobDetailResponse extends JobListingResponse {
  hrId: number;
  skills: string;
  requirements: string;
  questions: JobQuestionResponse[];
}

export interface CandidateSummary {
  userId: number;
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  gender: string;
  degree: string;
  status: string;
  CV_FilePath: string;
  processing_status?: string | null;
  result_quality?: string | null;
  queued_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  total_score?: number | null;
}

export interface HrAuthResponse {
  response?: boolean;
  id?: number;
}

export interface ScoreQuestion {
  question: string;
  video?: string;
  summary?: string;
  relevance?: number;
  emotion?: string;
  degraded?: boolean;
  warnings?: string[];
}

export interface ScoreReport {
  user_id: number;
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  cv: string;
  questions: ScoreQuestion[];
  total_score: number;
  total_english_score: number;
  trait1: string;
  trait2: string;
  trait3: string;
  trait4: string;
  trait5: string;
  result_quality: string;
  degraded: boolean;
  warnings: string[];
}

export interface TaskStatusResponse {
  status: string;
  result_quality?: string | null;
  queued_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}
