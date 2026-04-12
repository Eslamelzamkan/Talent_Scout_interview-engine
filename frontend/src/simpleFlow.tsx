import { type FormEvent, useEffect, useState } from "react";

import { ApiError, DEFAULT_API_BASE_URL, api, assetUrl } from "./api";
import type {
  CandidateFormState,
  CandidateSummary,
  JobDetailResponse,
  JobFormState,
  JobListingResponse,
  NoticeTone,
  ScoreReport,
  TaskStatusResponse,
} from "./types";
import {
  ChoiceCards,
  EmptyState,
  InfoBlock,
  InputField,
  Metric,
  Panel,
  StatusPill,
  TextAreaField,
  formatCurrency,
} from "./ui";

interface NoticeState {
  tone: NoticeTone;
  text: string;
}

const starterQuestions = [
  "Tell us about your background and why it fits this role.",
  "Walk through a project that shows your strongest work.",
  "Why do you want this job, and what would make you effective here?",
];

const jobTypeOptions = [
  {
    value: "Full-time",
    label: "Full-time",
    description: "Permanent role with standard weekly hours.",
  },
  {
    value: "Part-time",
    label: "Part-time",
    description: "Reduced weekly hours with regular scheduling.",
  },
  {
    value: "Internship",
    label: "Internship",
    description: "Training-focused role for students or early-career applicants.",
  },
];

const genderOptions = [
  {
    value: "Male",
    label: "Male",
  },
  {
    value: "Female",
    label: "Female",
  },
];

const defaultJobForm: JobFormState = {
  title: "",
  company: "",
  salary: "",
  job_type: "",
  skills: "",
  requirements: "",
  description: "",
  HRId: "",
  questions: [...starterQuestions],
};

const defaultCandidateForm: CandidateFormState = {
  first_name: "",
  last_name: "",
  email: "",
  phone: "",
  gender: "",
  degree: "",
};

function getErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Unexpected error";
}

function emptyTaskInfo(status = "idle"): TaskStatusResponse {
  return {
    status,
    result_quality: null,
    queued_at: null,
    started_at: null,
    completed_at: null,
  };
}

function getProcessingTone(status?: string | null): NoticeTone {
  if (status === "done") {
    return "success";
  }

  if (status === "failed") {
    return "error";
  }

  return "info";
}

function formatQualityLabel(quality?: string | null) {
  switch (quality) {
    case "complete":
      return "Complete";
    case "partial":
      return "Partial";
    case "failed":
      return "Failed";
    default:
      return "Unknown";
  }
}

function getQualityTone(quality?: string | null): NoticeTone {
  switch (quality) {
    case "complete":
      return "success";
    case "partial":
      return "warning";
    case "failed":
      return "error";
    default:
      return "info";
  }
}

function formatProcessingLabel(status?: string | null) {
  switch (status) {
    case "queued":
      return "Queued";
    case "processing":
      return "Processing";
    case "done":
      return "Done";
    case "failed":
      return "Failed";
    case "not_found":
      return "Not started";
    default:
      return "Submitted";
  }
}

function formatDateTime(value?: string | null) {
  if (!value) {
    return "Not yet";
  }

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function taskInfoFromApplicant(candidate?: CandidateSummary | null): TaskStatusResponse {
  if (!candidate?.processing_status) {
    return emptyTaskInfo("not_started");
  }

  return {
    status: candidate.processing_status,
    result_quality: candidate.result_quality ?? null,
    queued_at: candidate.queued_at ?? null,
    started_at: candidate.started_at ?? null,
    completed_at: candidate.completed_at ?? null,
  };
}

export default function SimpleFlow() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [apiDraft, setApiDraft] = useState(DEFAULT_API_BASE_URL);
  const [healthState, setHealthState] = useState("checking");
  const [notice, setNotice] = useState<NoticeState | null>(null);
  const [busy, setBusy] = useState<Record<string, boolean>>({});

  const [jobs, setJobs] = useState<JobListingResponse[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
  const [jobDetail, setJobDetail] = useState<JobDetailResponse | null>(null);
  const [jobForm, setJobForm] = useState<JobFormState>(defaultJobForm);

  const [candidateForm, setCandidateForm] =
    useState<CandidateFormState>(defaultCandidateForm);
  const [cvFile, setCvFile] = useState<File | null>(null);
  const [videoFiles, setVideoFiles] = useState<Record<number, File | null>>({});
  const [applicants, setApplicants] = useState<CandidateSummary[]>([]);
  const [activeUserId, setActiveUserId] = useState<number | null>(null);
  const [taskInfo, setTaskInfo] = useState<TaskStatusResponse>(emptyTaskInfo());
  const [scoreReport, setScoreReport] = useState<ScoreReport | null>(null);

  const activeApplicant =
    applicants.find((candidate) => candidate.userId === activeUserId) ?? null;

  function showNotice(tone: NoticeTone, text: string) {
    setNotice({ tone, text });
  }

  function resetApplicationState() {
    setCandidateForm(defaultCandidateForm);
    setCvFile(null);
    setVideoFiles({});
  }

  function resetReviewState() {
    setApplicants([]);
    setActiveUserId(null);
    setTaskInfo(emptyTaskInfo());
    setScoreReport(null);
  }

  function selectJob(jobId: number | null) {
    setSelectedJobId(jobId);
    setJobDetail(null);
    resetApplicationState();
    resetReviewState();
  }

  function updatePrompt(index: number, value: string) {
    setJobForm((current) => ({
      ...current,
      questions: current.questions.map((question, itemIndex) =>
        itemIndex === index ? value : question,
      ),
    }));
  }

  function pickActiveApplicant(
    rows: CandidateSummary[],
    preferredUserId?: number,
    currentUserId?: number | null,
  ) {
    return (
      rows.find((row) => row.userId === preferredUserId) ??
      rows.find((row) => row.userId === currentUserId) ??
      rows[0] ??
      null
    );
  }

  async function withBusy<T>(key: string, action: () => Promise<T>) {
    setBusy((current) => ({ ...current, [key]: true }));

    try {
      return await action();
    } finally {
      setBusy((current) => ({ ...current, [key]: false }));
    }
  }

  async function loadHealth(silent = false) {
    try {
      await api.health(apiBaseUrl);
      setHealthState("online");
      if (!silent) {
        showNotice("success", "API connection is healthy.");
      }
    } catch (error) {
      setHealthState("offline");
      if (!silent) {
        showNotice("error", `API connection failed: ${getErrorMessage(error)}`);
      }
    }
  }

  async function loadJobs(silent = false) {
    return withBusy("jobs", async () => {
      try {
        const response = await api.getJobs(apiBaseUrl);
        setJobs(response.jobs);

        if (response.jobs.length === 0) {
          selectJob(null);
          if (!silent) {
            showNotice("info", "No jobs exist yet. Publish the first one below.");
          }
          return;
        }

        const stillSelected = response.jobs.some((job) => job.id === selectedJobId);
        if (!stillSelected) {
          selectJob(response.jobs[0].id);
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          setJobs([]);
          selectJob(null);
          if (!silent) {
            showNotice("info", "No jobs exist yet. Publish the first one below.");
          }
          return;
        }

        throw error;
      }
    }).catch((error: unknown) => {
      showNotice("error", `Unable to load jobs: ${getErrorMessage(error)}`);
    });
  }

  async function loadJobDetail(jobId: number, silent = false) {
    return withBusy("jobDetail", async () => {
      try {
        const detail = await api.getJobInfo(apiBaseUrl, jobId);
        setJobDetail(detail);
      } catch (error) {
        if (!silent) {
          showNotice("error", `Unable to load job details: ${getErrorMessage(error)}`);
        }
      }
    });
  }

  async function loadApplicants(
    jobId: number,
    silent = false,
    preferredUserId?: number,
  ) {
    return withBusy("applicants", async () => {
      try {
        const rows = await api.listUsers(apiBaseUrl, jobId);
        setApplicants(rows);

        const nextApplicant = pickActiveApplicant(rows, preferredUserId, activeUserId);
        setActiveUserId(nextApplicant?.userId ?? null);
        setTaskInfo(taskInfoFromApplicant(nextApplicant));

        if (!nextApplicant) {
          setScoreReport(null);
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          setApplicants([]);
          setActiveUserId(null);
          setTaskInfo(emptyTaskInfo());
          setScoreReport(null);
          return;
        }

        if (!silent) {
          showNotice("error", `Unable to load applicants: ${getErrorMessage(error)}`);
        }
      }
    });
  }

  async function refreshTaskStatus(
    userId = activeUserId ?? undefined,
    jobId = selectedJobId ?? undefined,
    silent = false,
  ) {
    if (!userId || !jobId) {
      return;
    }

    return withBusy("task", async () => {
      try {
        const response = await api.getTaskStatus(apiBaseUrl, userId, jobId);
        setTaskInfo(response);
        setApplicants((current) =>
          current.map((candidate) =>
            candidate.userId === userId
              ? {
                  ...candidate,
                  processing_status: response.status,
                  result_quality: response.result_quality ?? candidate.result_quality ?? null,
                  queued_at: response.queued_at ?? candidate.queued_at ?? null,
                  started_at: response.started_at ?? candidate.started_at ?? null,
                  completed_at: response.completed_at ?? candidate.completed_at ?? null,
                }
              : candidate,
          ),
        );

        if (response.status === "done") {
          await loadScoreReport(userId, jobId, true);
        }

        if (!silent && response.status !== "not_found") {
          const qualityNote =
            response.result_quality && response.result_quality !== "complete"
              ? ` (${formatQualityLabel(response.result_quality).toLowerCase()} result)`
              : "";
          showNotice("info", `Assessment status: ${response.status}${qualityNote}`);
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          setTaskInfo(emptyTaskInfo("not_found"));
          if (!silent) {
            showNotice("info", "This applicant has not been scored yet.");
          }
          return;
        }

        if (!silent) {
          showNotice("error", `Unable to check status: ${getErrorMessage(error)}`);
        }
      }
    });
  }

  async function loadScoreReport(
    userId = activeUserId ?? undefined,
    jobId = selectedJobId ?? undefined,
    silent = false,
  ) {
    if (!userId || !jobId) {
      return;
    }

    return withBusy("scoreReport", async () => {
      try {
        const report = await api.getUserScores(apiBaseUrl, {
          user_id: userId,
          job_id: jobId,
        });
        setScoreReport(report);
        setTaskInfo((current) => ({
          ...current,
          result_quality: report.result_quality,
        }));
        setApplicants((current) =>
          current.map((candidate) =>
            candidate.userId === userId
              ? {
                  ...candidate,
                  result_quality: report.result_quality,
                }
              : candidate,
          ),
        );
        if (!silent) {
          showNotice(
            report.degraded ? "warning" : "success",
            report.degraded
              ? "Assessment report loaded with fallback warnings."
              : "Assessment report loaded.",
          );
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          setScoreReport(null);
          if (!silent) {
            showNotice("info", "No report exists yet for this applicant.");
          }
          return;
        }

        if (!silent) {
          showNotice("error", `Unable to load report: ${getErrorMessage(error)}`);
        }
      }
    });
  }

  async function refreshOverview() {
    await Promise.all([loadHealth(true), loadJobs(true)]);
    showNotice("info", "Workspace refreshed.");
  }

  async function handleCreateJob(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const salary = Number(jobForm.salary);
    const prompts = jobForm.questions.map((question) => question.trim());

    if (!jobForm.title.trim() || !jobForm.company.trim() || !jobForm.description.trim()) {
      showNotice("error", "Title, company, and description are required.");
      return;
    }

    if (!Number.isFinite(salary) || salary <= 0) {
      showNotice("error", "Salary must be a valid positive number.");
      return;
    }

    if (prompts.some((question) => !question)) {
      showNotice("error", "Keep all three interview prompts filled in.");
      return;
    }

    await withBusy("createJob", async () => {
      const created = await api.createJob(apiBaseUrl, {
        title: jobForm.title.trim(),
        description: jobForm.description.trim(),
        salary,
        company: jobForm.company.trim(),
        job_type: jobForm.job_type.trim(),
        skills: jobForm.skills.trim(),
        requirements: jobForm.requirements.trim(),
        questions: prompts.map((question) => ({ question })),
      });

      setJobForm({
        ...defaultJobForm,
        questions: [...starterQuestions],
      });
      setJobs((current) => {
        const next = current.filter((job) => job.id !== created.id);
        return [
          {
            id: created.id,
            title: created.title,
            company: created.company,
            salary: created.salary,
            job_type: created.job_type,
            description: created.description,
          },
          ...next,
        ];
      });
      setSelectedJobId(created.id);
      setJobDetail(created);
      resetApplicationState();
      resetReviewState();
      showNotice("success", `Job #${created.id} is live.`);
    }).catch((error: unknown) => {
      showNotice("error", `Unable to create job: ${getErrorMessage(error)}`);
    });
  }

  async function handleSubmitApplication(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedJobId || !jobDetail) {
      showNotice("error", "Select a job before submitting an application.");
      return;
    }

    const requiredFields = [
      candidateForm.first_name,
      candidateForm.last_name,
      candidateForm.email,
      candidateForm.phone,
      candidateForm.gender,
      candidateForm.degree,
    ];

    if (requiredFields.some((field) => !field.trim())) {
      showNotice("error", "Fill in all applicant details before submitting.");
      return;
    }

    if (!cvFile) {
      showNotice("error", "Attach a CV PDF before submitting.");
      return;
    }

    const missingVideo = jobDetail.questions.find((question) => !videoFiles[question.id]);
    if (missingVideo) {
      showNotice("error", "Upload one video answer for each interview prompt.");
      return;
    }

    await withBusy("submitApplication", async () => {
      const createdUser = await api.createUser(apiBaseUrl, {
        first_name: candidateForm.first_name.trim(),
        last_name: candidateForm.last_name.trim(),
        email: candidateForm.email.trim(),
        phone: candidateForm.phone.trim(),
        gender: candidateForm.gender.trim(),
        degree: candidateForm.degree.trim(),
        jobId: selectedJobId,
      });

      await api.uploadCv(apiBaseUrl, createdUser.id, selectedJobId, cvFile);

      for (const question of jobDetail.questions) {
        await api.uploadVideo(
          apiBaseUrl,
          createdUser.id,
          selectedJobId,
          question.id,
          videoFiles[question.id] as File,
        );
      }

      const computeResponse = await api.computeScores(apiBaseUrl, {
        user_id: createdUser.id,
        job_id: selectedJobId,
      });

      setActiveUserId(createdUser.id);
      setTaskInfo(computeResponse);
      setScoreReport(null);
      await loadApplicants(selectedJobId, true, createdUser.id);
      resetApplicationState();
      showNotice("info", "Application submitted and scoring started.");
    }).catch((error: unknown) => {
      showNotice("error", `Unable to submit application: ${getErrorMessage(error)}`);
    });
  }

  function selectApplicant(candidate: CandidateSummary) {
    setActiveUserId(candidate.userId);
    setTaskInfo(taskInfoFromApplicant(candidate));
    setScoreReport(null);
  }

  useEffect(() => {
    setNotice(null);
    setJobs([]);
    setSelectedJobId(null);
    setJobDetail(null);
    resetApplicationState();
    resetReviewState();
    void loadHealth(true);
    void loadJobs(true);
  }, [apiBaseUrl]);

  useEffect(() => {
    if (!selectedJobId) {
      return;
    }

    void loadJobDetail(selectedJobId, true);
    void loadApplicants(selectedJobId, true);
  }, [apiBaseUrl, selectedJobId]);

  useEffect(() => {
    if (!selectedJobId || !activeUserId) {
      setTaskInfo(emptyTaskInfo());
      setScoreReport(null);
      return;
    }

    void refreshTaskStatus(activeUserId, selectedJobId, true);
  }, [apiBaseUrl, selectedJobId, activeUserId]);

  useEffect(() => {
    if (!selectedJobId || !activeUserId) {
      return;
    }

    if (taskInfo.status !== "queued" && taskInfo.status !== "processing") {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshTaskStatus(activeUserId, selectedJobId, true);
    }, 4000);

    return () => window.clearInterval(intervalId);
  }, [apiBaseUrl, selectedJobId, activeUserId, taskInfo.status]);

  return (
    <div className="app-shell">
      <header className="hero-banner">
        <div className="hero-copy">
          <span className="eyebrow">Interview Engine</span>
          <h1>Create the role, collect the videos, get the score.</h1>
          <p>
            This version removes the control-room ceremony. You publish a job,
            the applicant submits the material, and the system moves straight
            into scoring and review.
          </p>
        </div>

        <div className="hero-console">
          <label className="field compact">
            <span>FastAPI base URL</span>
            <input
              type="text"
              value={apiDraft}
              onChange={(event) => setApiDraft(event.target.value)}
              placeholder="http://127.0.0.1:8000"
            />
          </label>

          <div className="hero-console-actions">
            <button
              className="primary-button"
              type="button"
              onClick={() => setApiBaseUrl(apiDraft.trim() || DEFAULT_API_BASE_URL)}
            >
              Connect
            </button>
            <button className="ghost-button" type="button" onClick={() => void refreshOverview()}>
              Refresh
            </button>
          </div>

          <div className="hero-status">
            <StatusPill
              label={
                healthState === "online"
                  ? "API online"
                  : healthState === "offline"
                    ? "API offline"
                    : "Checking API"
              }
              tone={
                healthState === "online"
                  ? "success"
                  : healthState === "offline"
                    ? "error"
                    : "info"
              }
            />
            <small>Connected target: {apiBaseUrl}</small>
          </div>
        </div>
      </header>

      {notice ? <div className={`notice notice-${notice.tone}`}>{notice.text}</div> : null}

      <div className="simple-grid">
        <aside className="sidebar-stack">
          <Panel eyebrow="Step 1" title="Publish a role">
            <form className="stack-form" onSubmit={handleCreateJob}>
              <div className="two-column">
                <InputField
                  label="Job title"
                  value={jobForm.title}
                  onChange={(value) =>
                    setJobForm((current) => ({ ...current, title: value }))
                  }
                  placeholder="Machine Learning Engineer"
                />
                <InputField
                  label="Company"
                  value={jobForm.company}
                  onChange={(value) =>
                    setJobForm((current) => ({ ...current, company: value }))
                  }
                  placeholder="Northwind Labs"
                />
                <InputField
                  label="Salary"
                  type="number"
                  value={jobForm.salary}
                  onChange={(value) =>
                    setJobForm((current) => ({ ...current, salary: value }))
                  }
                  placeholder="18000"
                />
              </div>

              <ChoiceCards
                label="Job type"
                value={jobForm.job_type}
                onChange={(value) =>
                  setJobForm((current) => ({ ...current, job_type: value }))
                }
                options={jobTypeOptions}
              />

              <TextAreaField
                label="Job description"
                value={jobForm.description}
                onChange={(value) =>
                  setJobForm((current) => ({ ...current, description: value }))
                }
                placeholder="Describe the role, what success looks like, and what kind of candidate you want."
              />

              <TextAreaField
                label="Skills"
                value={jobForm.skills}
                onChange={(value) =>
                  setJobForm((current) => ({ ...current, skills: value }))
                }
                placeholder="Python, FastAPI, NLP, Computer Vision"
              />

              <TextAreaField
                label="Requirements"
                value={jobForm.requirements}
                onChange={(value) =>
                  setJobForm((current) => ({ ...current, requirements: value }))
                }
                placeholder="3+ years experience, strong communication, production ML mindset"
              />

              <div className="question-editor">
                <span className="field-label">Interview prompts</span>
                <p className="helper-copy">
                  Keep these three prompts aligned with the current scoring
                  pipeline and ask for one video answer per prompt.
                </p>
                {jobForm.questions.map((question, index) => (
                  <label className="field" key={`prompt-${index}`}>
                    <span>Prompt {index + 1}</span>
                    <textarea
                      rows={3}
                      value={question}
                      onChange={(event) => updatePrompt(index, event.target.value)}
                    />
                  </label>
                ))}
              </div>

              <button
                className="primary-button"
                type="submit"
                disabled={Boolean(busy.createJob)}
              >
                {busy.createJob ? "Publishing..." : "Publish role"}
              </button>
            </form>
          </Panel>

          <Panel eyebrow="Active roles" title="Choose a job">
            {jobs.length === 0 ? (
              <EmptyState message="No roles have been created yet." />
            ) : (
              <div className="job-rail">
                {jobs.map((job) => (
                  <button
                    key={job.id}
                    type="button"
                    className={`job-rail-card ${
                      job.id === selectedJobId ? "job-rail-card-active" : ""
                    }`}
                    onClick={() => selectJob(job.id)}
                  >
                    <div>
                      <strong>{job.title}</strong>
                      <span>{job.company}</span>
                    </div>
                    <div className="job-rail-meta">
                      <StatusPill label={job.job_type} tone="info" />
                      <small>{formatCurrency(job.salary)}</small>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </Panel>
        </aside>

        <main className="simple-stage">
          <Panel eyebrow="Selected role" title={jobDetail ? jobDetail.title : "No role selected"}>
            {jobDetail ? (
              <div className="detail-stack">
                <div className="hero-card">
                  <div>
                    <span className="eyebrow">Live job</span>
                    <h2>{jobDetail.title}</h2>
                    <p>{jobDetail.company}</p>
                  </div>
                  <StatusPill label={jobDetail.job_type} tone="info" />
                </div>

                <div className="metric-strip">
                  <Metric label="Salary" value={formatCurrency(jobDetail.salary)} />
                  <Metric label="Prompts" value={String(jobDetail.questions.length)} />
                  <Metric label="Skills" value={jobDetail.skills || "Not specified"} />
                </div>

                <InfoBlock title="Description" value={jobDetail.description} />
                <InfoBlock title="Requirements" value={jobDetail.requirements} />

                <div className="question-cards">
                  {jobDetail.questions.map((question, index) => (
                    <article className="mini-card" key={question.id}>
                      <span className="card-index">Prompt {index + 1}</span>
                      <p>{question.question}</p>
                    </article>
                  ))}
                </div>
              </div>
            ) : (
              <EmptyState message="Select a role to open the applicant flow." />
            )}
          </Panel>

          <Panel eyebrow="Step 2" title="Applicant submission">
            {jobDetail ? (
              <form className="stack-form" onSubmit={handleSubmitApplication}>
                <div className="two-column">
                  <InputField
                    label="First name"
                    value={candidateForm.first_name}
                    onChange={(value) =>
                      setCandidateForm((current) => ({ ...current, first_name: value }))
                    }
                    placeholder="Nadine"
                  />
                  <InputField
                    label="Last name"
                    value={candidateForm.last_name}
                    onChange={(value) =>
                      setCandidateForm((current) => ({ ...current, last_name: value }))
                    }
                    placeholder="Elkady"
                  />
                  <InputField
                    label="Email"
                    value={candidateForm.email}
                    onChange={(value) =>
                      setCandidateForm((current) => ({ ...current, email: value }))
                    }
                    placeholder="candidate@example.com"
                  />
                  <InputField
                    label="Phone"
                    value={candidateForm.phone}
                    onChange={(value) =>
                      setCandidateForm((current) => ({ ...current, phone: value }))
                    }
                    placeholder="+20 100 000 0000"
                  />
                  <InputField
                    label="Degree"
                    value={candidateForm.degree}
                    onChange={(value) =>
                      setCandidateForm((current) => ({ ...current, degree: value }))
                    }
                    placeholder="BSc Computer Science"
                  />
                </div>

                <ChoiceCards
                  label="Gender"
                  value={candidateForm.gender}
                  onChange={(value) =>
                    setCandidateForm((current) => ({ ...current, gender: value }))
                  }
                  options={genderOptions}
                />

                <div className="upload-stack">
                  <div className="upload-card">
                    <div>
                      <span className="field-label">CV upload</span>
                      <p>Attach a PDF resume for this applicant.</p>
                    </div>
                    <input
                      type="file"
                      accept=".pdf,application/pdf"
                      onChange={(event) => setCvFile(event.target.files?.[0] ?? null)}
                    />
                  </div>

                  <div className="upload-card">
                    <div>
                      <span className="field-label">Video answers</span>
                      <p>Upload one answer clip for each prompt below.</p>
                    </div>
                    <div className="question-upload-grid">
                      {jobDetail.questions.map((question, index) => (
                        <article className="mini-card" key={question.id}>
                          <span className="card-index">Answer {index + 1}</span>
                          <p>{question.question}</p>
                          <input
                            type="file"
                            accept="video/mp4"
                            onChange={(event) =>
                              setVideoFiles((current) => ({
                                ...current,
                                [question.id]: event.target.files?.[0] ?? null,
                              }))
                            }
                          />
                        </article>
                      ))}
                    </div>
                  </div>
                </div>

                <button
                  className="primary-button"
                  type="submit"
                  disabled={Boolean(busy.submitApplication)}
                >
                  {busy.submitApplication
                    ? "Submitting..."
                    : "Submit application and start scoring"}
                </button>
              </form>
            ) : (
              <EmptyState message="Choose a job first, then the applicant can submit here." />
            )}
          </Panel>

          <div className="result-grid">
            <Panel
              eyebrow="Applicants"
              title="Recent submissions"
              actions={
                selectedJobId ? (
                  <button
                    className="ghost-button"
                    type="button"
                    onClick={() => selectedJobId && void loadApplicants(selectedJobId)}
                    disabled={!selectedJobId || Boolean(busy.applicants)}
                  >
                    {busy.applicants ? "Refreshing..." : "Refresh"}
                  </button>
                ) : null
              }
            >
              {applicants.length === 0 ? (
                <EmptyState message="Applicants for the selected role will appear here." />
              ) : (
                <div className="candidate-table">
                  {applicants.map((candidate) => (
                    <button
                      key={candidate.userId}
                      type="button"
                      className={`candidate-row ${
                        candidate.userId === activeUserId ? "candidate-row-active" : ""
                      }`}
                      onClick={() => selectApplicant(candidate)}
                    >
                      <div className="candidate-main">
                        <strong>
                          {candidate.first_name} {candidate.last_name}
                        </strong>
                        <span>{candidate.email}</span>
                      </div>
                      <div className="candidate-meta">
                        <StatusPill
                          label={formatProcessingLabel(candidate.processing_status)}
                          tone={getProcessingTone(candidate.processing_status)}
                        />
                        {candidate.processing_status === "done" && candidate.result_quality ? (
                          <StatusPill
                            label={formatQualityLabel(candidate.result_quality)}
                            tone={getQualityTone(candidate.result_quality)}
                          />
                        ) : null}
                        <small>
                          {candidate.completed_at
                            ? `Finished ${formatDateTime(candidate.completed_at)}`
                            : candidate.started_at
                              ? `Started ${formatDateTime(candidate.started_at)}`
                              : candidate.queued_at
                                ? `Queued ${formatDateTime(candidate.queued_at)}`
                                : `User #${candidate.userId}`}
                        </small>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>

            <Panel eyebrow="Step 3" title="Assessment result">
              <div className="detail-stack">
                <div className="metric-strip">
                  <Metric label="Selected job" value={selectedJobId ? `#${selectedJobId}` : "None"} />
                  <Metric label="Applicant" value={activeApplicant ? `#${activeApplicant.userId}` : "None"} />
                  <Metric label="Evaluation" value={formatProcessingLabel(taskInfo.status)} />
                  <Metric label="Quality" value={formatQualityLabel(taskInfo.result_quality)} />
                  <Metric label="Started" value={formatDateTime(taskInfo.started_at)} />
                  <Metric label="Finished" value={formatDateTime(taskInfo.completed_at)} />
                </div>

                <div className="toolbar-row">
                  <button
                    className="ghost-button"
                    type="button"
                    onClick={() => void refreshTaskStatus()}
                    disabled={!selectedJobId || !activeUserId || Boolean(busy.task)}
                  >
                    {busy.task ? "Checking..." : "Check status"}
                  </button>
                  <button
                    className="ghost-button"
                    type="button"
                    onClick={() => void loadScoreReport()}
                    disabled={!selectedJobId || !activeUserId || Boolean(busy.scoreReport)}
                  >
                    {busy.scoreReport ? "Loading..." : "Load report"}
                  </button>
                </div>

                {scoreReport ? (
                  <div className="detail-stack">
                    <div className="hero-card">
                      <div>
                        <span className="eyebrow">Applicant summary</span>
                        <h2>
                          {scoreReport.first_name} {scoreReport.last_name}
                        </h2>
                        <p>{scoreReport.email}</p>
                      </div>
                      <div className="report-meta">
                        <StatusPill label={`Score ${Math.round(scoreReport.total_score)}`} tone="success" />
                        <StatusPill
                          label={formatQualityLabel(scoreReport.result_quality)}
                          tone={getQualityTone(scoreReport.result_quality)}
                        />
                      </div>
                    </div>

                    {scoreReport.warnings.length > 0 ? (
                      <div className="notice notice-warning">
                        <strong>Partial evaluation</strong>
                        <div className="warning-list">
                          {scoreReport.warnings.map((warning) => (
                            <p key={warning}>{warning}</p>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    <div className="metric-strip">
                      <Metric label="English" value={String(Math.round(scoreReport.total_english_score))} />
                      <Metric label="Phone" value={scoreReport.phone} />
                      <Metric label="Trait count" value="5" />
                    </div>

                    <div className="trait-strip">
                      {[
                        scoreReport.trait1,
                        scoreReport.trait2,
                        scoreReport.trait3,
                        scoreReport.trait4,
                        scoreReport.trait5,
                      ]
                        .filter(Boolean)
                        .map((trait) => (
                          <span className="trait-chip" key={trait}>
                            {trait}
                          </span>
                        ))}
                    </div>

                    <div className="question-cards">
                      {scoreReport.questions.map((question, index) => (
                        <article className="mini-card report-card" key={`${question.question}-${index}`}>
                          <span className="card-index">Answer {index + 1}</span>
                          <h3>{question.question}</h3>
                          <p>{question.summary || "No summary available yet."}</p>
                          <div className="report-meta">
                            <StatusPill label={`Emotion: ${question.emotion ?? "Neutral"}`} tone="info" />
                            <StatusPill label={`Relevance: ${question.relevance ?? 0}`} tone="success" />
                            {question.degraded ? (
                              <StatusPill label="Fallback used" tone="warning" />
                            ) : null}
                          </div>
                          {question.warnings && question.warnings.length > 0 ? (
                            <div className="warning-list">
                              {question.warnings.map((warning) => (
                                <p key={`${question.question}-${warning}`}>{warning}</p>
                              ))}
                            </div>
                          ) : null}
                          {question.video ? (
                            <a href={assetUrl(apiBaseUrl, question.video)} target="_blank" rel="noreferrer">
                              Open submitted video
                            </a>
                          ) : null}
                        </article>
                      ))}
                    </div>
                  </div>
                ) : (
                  <EmptyState message="After submission, the score and summaries will appear here." />
                )}
              </div>
            </Panel>
          </div>
        </main>
      </div>
    </div>
  );
}
