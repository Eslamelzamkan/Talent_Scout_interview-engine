import type {
  CandidateSummary,
  HrAuthResponse,
  JobDetailResponse,
  JobListResponse,
  ScoreReport,
  TaskStatusResponse,
} from "./types";

export const DEFAULT_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

function trimBaseUrl(baseUrl: string) {
  return baseUrl.replace(/\/+$/, "");
}

function buildUrl(
  baseUrl: string,
  path: string,
  params?: Record<string, string | number | undefined>,
) {
  const url = new URL(`${trimBaseUrl(baseUrl)}${path}`);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== "") {
        url.searchParams.set(key, String(value));
      }
    });
  }
  return url.toString();
}

async function parseResponse(response: Response) {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function request<T>(
  baseUrl: string,
  path: string,
  init?: RequestInit,
  params?: Record<string, string | number | undefined>,
): Promise<T> {
  const response = await fetch(buildUrl(baseUrl, path, params), init);
  const payload = await parseResponse(response);

  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload !== null && "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : response.statusText;
    throw new ApiError(detail || "Request failed", response.status, payload);
  }

  return payload as T;
}

export function assetUrl(baseUrl: string, filePath?: string) {
  if (!filePath) {
    return "";
  }

  const normalized = filePath.replace(/\\/g, "/").replace(/^\/+/, "");
  return `${trimBaseUrl(baseUrl)}/${normalized}`;
}

export const api = {
  health(baseUrl: string) {
    return request<{ status: string }>(baseUrl, "/health");
  },

  getJobs(baseUrl: string) {
    return request<JobListResponse>(baseUrl, "/job/get_jobs");
  },

  getJobInfo(baseUrl: string, jobId: number) {
    return request<JobDetailResponse>(
      baseUrl,
      "/job/get_job_info",
      undefined,
      { job_id: jobId },
    );
  },

  createJob(
    baseUrl: string,
    payload: {
      title: string;
      description: string;
      salary: number;
      company: string;
      job_type: string;
      skills: string;
      requirements: string;
      HRId?: number;
      questions: Array<{ question: string }>;
    },
  ) {
    return request<JobDetailResponse>(baseUrl, "/job/create_job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  createHr(
    baseUrl: string,
    payload: { name: string; email: string; password: string },
  ) {
    return request<HrAuthResponse>(baseUrl, "/hr/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  loginHr(
    baseUrl: string,
    payload: { email: string; password: string },
  ) {
    return request<HrAuthResponse>(baseUrl, "/hr/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  createUser(
    baseUrl: string,
    payload: {
      first_name: string;
      last_name: string;
      jobId: number;
      email: string;
      phone: string;
      gender: string;
      degree: string;
    },
  ) {
    return request<{ id: number }>(baseUrl, "/user/create-user", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  uploadCv(baseUrl: string, uid: number, jobId: number, file: File) {
    const formData = new FormData();
    formData.append("file", file);

    return request<{ file_path: string }>(
      baseUrl,
      "/user/upload-CV",
      {
        method: "PUT",
        body: formData,
      },
      { uid, jobId },
    );
  },

  uploadVideo(
    baseUrl: string,
    userId: number,
    jobId: number,
    questionId: number,
    file: File,
  ) {
    const formData = new FormData();
    formData.append("file", file);

    return request<{ message: string; file_path: string }>(
      baseUrl,
      "/user/upload-video",
      {
        method: "POST",
        body: formData,
      },
      { userId, jobId, questionId },
    );
  },

  listUsers(baseUrl: string, jobId: number, status?: string) {
    return request<CandidateSummary[]>(
      baseUrl,
      "/user/",
      undefined,
      { job_id: jobId, status },
    );
  },

  updateUserStatus(
    baseUrl: string,
    userId: number,
    jobId: number,
    newStatus: string,
  ) {
    return request<{ id: number }>(
      baseUrl,
      `/user/${userId}/status`,
      { method: "PUT" },
      { job_id: jobId, new_status: newStatus },
    );
  },

  computeScores(
    baseUrl: string,
    payload: { user_id: number; job_id: number; hr_id?: number },
  ) {
    return request<{
      status: string;
      message: string;
      queued_at?: string | null;
      started_at?: string | null;
      completed_at?: string | null;
    }>(
      baseUrl,
      "/hr/compute_scores",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    );
  },

  getTaskStatus(baseUrl: string, userId: number, jobId: number) {
    return request<TaskStatusResponse>(
      baseUrl,
      `/hr/task_status/${userId}/${jobId}`,
    );
  },

  getUserScores(
    baseUrl: string,
    payload: { user_id: number; job_id: number },
  ) {
    return request<ScoreReport>(baseUrl, "/hr/get_user_scores", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
};
