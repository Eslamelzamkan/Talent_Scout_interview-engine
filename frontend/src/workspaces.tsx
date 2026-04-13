import type { Dispatch, FormEvent, SetStateAction } from "react";

import { assetUrl } from "./api";
import type {
  CandidateFormState,
  CandidateSummary,
  HrFormState,
  JobDetailResponse,
  JobFormState,
  ScoreReport,
} from "./types";
import {
  EmptyState,
  InfoBlock,
  InputField,
  Metric,
  Panel,
  StatusPill,
  TextAreaField,
  formatCurrency,
} from "./ui";

const statusOptions = ["pending", "passed", "accepted", "rejected"];

const candidateNamePlaceholderOptions = [
  { first: "Maya", last: "Hassan" },
  { first: "Omar", last: "Adel" },
  { first: "Lina", last: "Fahmy" },
  { first: "Karim", last: "Nabil" },
  { first: "Sara", last: "Maher" },
  { first: "Youssef", last: "Saleh" },
];

const candidateNamePlaceholder =
  candidateNamePlaceholderOptions[
    Math.floor(Math.random() * candidateNamePlaceholderOptions.length)
  ] ?? { first: "Maya", last: "Hassan" };

export function JobsWorkspace({
  busy,
  jobForm,
  setJobForm,
  jobDetail,
  selectedJobId,
  addQuestionField,
  removeQuestionField,
  updateQuestion,
  handleCreateJob,
  loadJobDetail,
}: {
  busy: Record<string, boolean>;
  jobForm: JobFormState;
  setJobForm: Dispatch<SetStateAction<JobFormState>>;
  jobDetail: JobDetailResponse | null;
  selectedJobId: number | null;
  addQuestionField: () => void;
  removeQuestionField: (index: number) => void;
  updateQuestion: (index: number, value: string) => void;
  handleCreateJob: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  loadJobDetail: (jobId: number) => Promise<void>;
}) {
  return (
    <div className="stage-grid">
      <Panel eyebrow="Author" title="Create a new role">
        <form className="stack-form" onSubmit={handleCreateJob}>
          <div className="two-column">
            <InputField label="Title" value={jobForm.title} onChange={(value) => setJobForm((current) => ({ ...current, title: value }))} placeholder="Senior Backend Engineer" />
            <InputField label="Company" value={jobForm.company} onChange={(value) => setJobForm((current) => ({ ...current, company: value }))} placeholder="Northwind Labs" />
            <InputField label="Salary" value={jobForm.salary} onChange={(value) => setJobForm((current) => ({ ...current, salary: value }))} placeholder="15000" />
            <InputField label="Job type" value={jobForm.job_type} onChange={(value) => setJobForm((current) => ({ ...current, job_type: value }))} placeholder="Full-time" />
            <InputField label="HR id" value={jobForm.HRId} onChange={(value) => setJobForm((current) => ({ ...current, HRId: value }))} placeholder="1" />
            <InputField label="Skills" value={jobForm.skills} onChange={(value) => setJobForm((current) => ({ ...current, skills: value }))} placeholder="FastAPI, PostgreSQL, NLP" />
          </div>

          <TextAreaField label="Requirements" value={jobForm.requirements} onChange={(value) => setJobForm((current) => ({ ...current, requirements: value }))} placeholder="5+ years experience, async Python, ML inference ops..." />
          <TextAreaField label="Description" value={jobForm.description} onChange={(value) => setJobForm((current) => ({ ...current, description: value }))} placeholder="Summarize the mission, the team, and the interview expectations." />

          <div className="question-header">
            <div>
              <span className="field-label">Interview questions</span>
              <p>Shape the answer sequence your candidates will record against.</p>
            </div>
            <button className="ghost-button" type="button" onClick={addQuestionField}>
              Add question
            </button>
          </div>

          <div className="question-list">
            {jobForm.questions.map((question, index) => (
              <div className="question-row" key={`question-${index}`}>
                <textarea
                  value={question}
                  onChange={(event) => updateQuestion(index, event.target.value)}
                  placeholder={`Question ${index + 1}`}
                  rows={2}
                />
                <button className="ghost-button" type="button" onClick={() => removeQuestionField(index)}>
                  Remove
                </button>
              </div>
            ))}
          </div>

          <button className="primary-button" type="submit" disabled={Boolean(busy.createJob)}>
            {busy.createJob ? "Publishing..." : "Publish role"}
          </button>
        </form>
      </Panel>

      <Panel
        eyebrow="Inspect"
        title={jobDetail ? "Role breakdown" : "Select a role"}
        actions={
          selectedJobId ? (
            <button
              className="ghost-button"
              type="button"
              onClick={() => selectedJobId && void loadJobDetail(selectedJobId)}
              disabled={Boolean(busy.jobDetail)}
            >
              {busy.jobDetail ? "Refreshing..." : "Reload detail"}
            </button>
          ) : null
        }
      >
        {jobDetail ? (
          <div className="detail-stack">
            <div className="hero-card">
              <div>
                <span className="eyebrow">Selected role</span>
                <h2>{jobDetail.title}</h2>
                <p>{jobDetail.company}</p>
              </div>
              <StatusPill label={jobDetail.job_type} tone="info" />
            </div>

            <div className="metric-strip">
              <Metric label="Salary" value={formatCurrency(jobDetail.salary)} />
              <Metric label="Questions" value={String(jobDetail.questions.length)} />
              <Metric label="Skills" value={jobDetail.skills} />
            </div>

            <InfoBlock title="Description" value={jobDetail.description} />
            <InfoBlock title="Requirements" value={jobDetail.requirements} />

            <div className="question-cards">
              {jobDetail.questions.map((question, index) => (
                <article className="mini-card" key={question.id}>
                  <span className="card-index">Q{index + 1}</span>
                  <p>{question.question}</p>
                </article>
              ))}
            </div>
          </div>
        ) : (
          <EmptyState message="Select a job to review its questions, skills, and setup." />
        )}
      </Panel>
    </div>
  );
}

export function CandidatesWorkspace({
  busy,
  candidateForm,
  setCandidateForm,
  selectedJobId,
  selectedUserId,
  statusFilter,
  setStatusFilter,
  statusDraft,
  setStatusDraft,
  jobDetail,
  cvFile,
  setCvFile,
  videoFiles,
  setVideoFiles,
  candidates,
  handleCreateCandidate,
  handleUploadCv,
  handleUploadVideo,
  handleUpdateStatus,
  refreshCandidates,
  selectCandidate,
}: {
  busy: Record<string, boolean>;
  candidateForm: CandidateFormState;
  setCandidateForm: Dispatch<SetStateAction<CandidateFormState>>;
  selectedJobId: number | null;
  selectedUserId: number | null;
  statusFilter: string;
  setStatusFilter: Dispatch<SetStateAction<string>>;
  statusDraft: string;
  setStatusDraft: Dispatch<SetStateAction<string>>;
  jobDetail: JobDetailResponse | null;
  cvFile: File | null;
  setCvFile: Dispatch<SetStateAction<File | null>>;
  videoFiles: Record<number, File | null>;
  setVideoFiles: Dispatch<SetStateAction<Record<number, File | null>>>;
  candidates: CandidateSummary[];
  handleCreateCandidate: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleUploadCv: () => Promise<void>;
  handleUploadVideo: (questionId: number) => Promise<void>;
  handleUpdateStatus: () => Promise<void>;
  refreshCandidates: (jobId: number, status: string) => Promise<void>;
  selectCandidate: (row: CandidateSummary) => void;
}) {
  return (
    <div className="stage-grid">
      <Panel eyebrow="Intake" title="Register a candidate">
        <form className="stack-form" onSubmit={handleCreateCandidate}>
          <div className="two-column">
            <InputField label="First name" value={candidateForm.first_name} onChange={(value) => setCandidateForm((current) => ({ ...current, first_name: value }))} placeholder={candidateNamePlaceholder.first} />
            <InputField label="Last name" value={candidateForm.last_name} onChange={(value) => setCandidateForm((current) => ({ ...current, last_name: value }))} placeholder={candidateNamePlaceholder.last} />
            <InputField label="Email" value={candidateForm.email} onChange={(value) => setCandidateForm((current) => ({ ...current, email: value }))} placeholder="candidate@example.com" />
            <InputField label="Phone" value={candidateForm.phone} onChange={(value) => setCandidateForm((current) => ({ ...current, phone: value }))} placeholder="+20 100 000 0000" />
            <InputField label="Gender" value={candidateForm.gender} onChange={(value) => setCandidateForm((current) => ({ ...current, gender: value }))} placeholder="Female" />
            <InputField label="Degree" value={candidateForm.degree} onChange={(value) => setCandidateForm((current) => ({ ...current, degree: value }))} placeholder="BSc Computer Science" />
          </div>

          <div className="selection-summary">
            <Metric label="Selected job" value={selectedJobId ? `#${selectedJobId}` : "None"} />
            <Metric label="Current status view" value={statusFilter} />
          </div>

          <button className="primary-button" type="submit" disabled={!selectedJobId || Boolean(busy.createCandidate)}>
            {busy.createCandidate ? "Creating..." : "Create candidate"}
          </button>
        </form>
      </Panel>

      <Panel eyebrow="Assets" title="CV and answer uploads">
        <div className="upload-stack">
          <div className="upload-card">
            <div>
              <span className="field-label">CV upload</span>
              <p>Attach a PDF to the currently selected candidate.</p>
            </div>
            <input
              type="file"
              accept=".pdf,application/pdf"
              onChange={(event) => setCvFile(event.target.files?.[0] ?? null)}
            />
            <button className="primary-button" type="button" onClick={() => void handleUploadCv()} disabled={!selectedUserId || !selectedJobId || !cvFile || Boolean(busy.uploadCv)}>
              {busy.uploadCv ? "Uploading..." : "Upload CV"}
            </button>
          </div>

          <div className="upload-card">
            <div>
              <span className="field-label">Video answers</span>
              <p>Upload one MP4 per question for the selected role.</p>
            </div>

            {jobDetail?.questions.length ? (
              <div className="question-upload-grid">
                {jobDetail.questions.map((question, index) => {
                  const busyKey = `video-${question.id}`;
                  return (
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
                      <button
                        className="ghost-button"
                        type="button"
                        onClick={() => void handleUploadVideo(question.id)}
                        disabled={!selectedUserId || !selectedJobId || !videoFiles[question.id] || Boolean(busy[busyKey])}
                      >
                        {busy[busyKey] ? "Uploading..." : "Upload answer"}
                      </button>
                    </article>
                  );
                })}
              </div>
            ) : (
              <EmptyState message="Select a job with questions before uploading videos." />
            )}
          </div>
        </div>
      </Panel>

      <Panel
        eyebrow="Pipeline"
        title="Candidates by status"
        actions={
          <button
            className="ghost-button"
            type="button"
            onClick={() => selectedJobId && void refreshCandidates(selectedJobId, statusFilter)}
            disabled={!selectedJobId || Boolean(busy.candidates)}
          >
            {busy.candidates ? "Refreshing..." : "Refresh list"}
          </button>
        }
      >
        <div className="toolbar-row">
          <label className="field compact">
            <span>Status filter</span>
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
              {statusOptions.map((status) => (
                <option key={status} value={status}>
                  {status}
                </option>
              ))}
            </select>
          </label>

          <label className="field compact">
            <span>Next status</span>
            <select value={statusDraft} onChange={(event) => setStatusDraft(event.target.value)}>
              {statusOptions.map((status) => (
                <option key={status} value={status}>
                  {status}
                </option>
              ))}
            </select>
          </label>

          <button className="primary-button" type="button" onClick={() => void handleUpdateStatus()} disabled={!selectedUserId || !selectedJobId || Boolean(busy.updateStatus)}>
            {busy.updateStatus ? "Saving..." : "Update status"}
          </button>
        </div>

        {candidates.length === 0 ? (
          <EmptyState message="No candidates in this status view yet." />
        ) : (
          <div className="candidate-table">
            {candidates.map((candidate) => (
              <button
                key={candidate.userId}
                type="button"
                className={`candidate-row ${
                  candidate.userId === selectedUserId ? "candidate-row-active" : ""
                }`}
                onClick={() => selectCandidate(candidate)}
              >
                <div className="candidate-main">
                  <strong>
                    {candidate.first_name} {candidate.last_name}
                  </strong>
                  <span>{candidate.email}</span>
                </div>
                <div className="candidate-meta">
                  <StatusPill label={candidate.status} tone="info" />
                  <small>
                    {candidate.total_score !== undefined && candidate.total_score !== null
                      ? `Score ${Math.round(candidate.total_score)}`
                      : "No score yet"}
                  </small>
                </div>
              </button>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

export function ReviewWorkspace({
  apiBaseUrl,
  busy,
  hrCreateForm,
  setHrCreateForm,
  hrLoginForm,
  setHrLoginForm,
  selectedHrId,
  selectedJobId,
  selectedUserId,
  taskStatus,
  scoreReport,
  handleCreateHr,
  handleLoginHr,
  handleComputeScores,
  refreshTaskStatus,
  loadScoreReport,
}: {
  apiBaseUrl: string;
  busy: Record<string, boolean>;
  hrCreateForm: HrFormState;
  setHrCreateForm: Dispatch<SetStateAction<HrFormState>>;
  hrLoginForm: HrFormState;
  setHrLoginForm: Dispatch<SetStateAction<HrFormState>>;
  selectedHrId: number | null;
  selectedJobId: number | null;
  selectedUserId: number | null;
  taskStatus: string;
  scoreReport: ScoreReport | null;
  handleCreateHr: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleLoginHr: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleComputeScores: () => Promise<void>;
  refreshTaskStatus: (userId?: number, jobId?: number, silent?: boolean) => Promise<void>;
  loadScoreReport: (userId?: number, jobId?: number, silent?: boolean) => Promise<void>;
}) {
  return (
    <div className="stage-grid">
      <Panel eyebrow="Identity" title="HR accounts and session">
        <div className="split-stack">
          <form className="stack-form" onSubmit={handleCreateHr}>
            <h3>Create HR account</h3>
            <InputField label="Name" value={hrCreateForm.name ?? ""} onChange={(value) => setHrCreateForm((current) => ({ ...current, name: value }))} placeholder="Hiring Manager" />
            <InputField label="Email" value={hrCreateForm.email} onChange={(value) => setHrCreateForm((current) => ({ ...current, email: value }))} placeholder="hr@example.com" />
            <InputField label="Password" type="password" value={hrCreateForm.password} onChange={(value) => setHrCreateForm((current) => ({ ...current, password: value }))} placeholder="********" />
            <button className="ghost-button" type="submit" disabled={Boolean(busy.createHr)}>
              {busy.createHr ? "Creating..." : "Create HR"}
            </button>
          </form>

          <form className="stack-form" onSubmit={handleLoginHr}>
            <h3>Log in for scoring</h3>
            <InputField label="Email" value={hrLoginForm.email} onChange={(value) => setHrLoginForm((current) => ({ ...current, email: value }))} placeholder="hr@example.com" />
            <InputField label="Password" type="password" value={hrLoginForm.password} onChange={(value) => setHrLoginForm((current) => ({ ...current, password: value }))} placeholder="********" />
            <button className="primary-button" type="submit" disabled={Boolean(busy.loginHr)}>
              {busy.loginHr ? "Signing in..." : "Log in"}
            </button>
          </form>
        </div>
      </Panel>

      <Panel eyebrow="Scoring" title="Compute and monitor">
        <div className="review-summary">
          <Metric label="HR id" value={selectedHrId ? `#${selectedHrId}` : "None"} />
          <Metric label="Job id" value={selectedJobId ? `#${selectedJobId}` : "None"} />
          <Metric label="Candidate id" value={selectedUserId ? `#${selectedUserId}` : "None"} />
          <Metric label="Task state" value={taskStatus} />
        </div>

        <div className="toolbar-row">
          <button className="primary-button" type="button" onClick={() => void handleComputeScores()} disabled={!selectedHrId || !selectedJobId || !selectedUserId || Boolean(busy.compute)}>
            {busy.compute ? "Starting..." : "Compute scores"}
          </button>
          <button className="ghost-button" type="button" onClick={() => void refreshTaskStatus()} disabled={!selectedJobId || !selectedUserId || Boolean(busy.task)}>
            {busy.task ? "Refreshing..." : "Check task status"}
          </button>
          <button className="ghost-button" type="button" onClick={() => void loadScoreReport()} disabled={!selectedJobId || !selectedUserId || Boolean(busy.scoreReport)}>
            {busy.scoreReport ? "Loading..." : "Load score report"}
          </button>
        </div>

        <p className="supporting-copy">
          The backend now lazy-loads AI models on demand. Missing weight files will
          degrade individual scoring components without taking the API offline.
        </p>
      </Panel>

      <Panel eyebrow="Report" title="Interview assessment">
        {scoreReport ? (
          <div className="detail-stack">
            <div className="hero-card">
              <div>
                <span className="eyebrow">Candidate overview</span>
                <h2>
                  {scoreReport.first_name} {scoreReport.last_name}
                </h2>
                <p>{scoreReport.email}</p>
              </div>
              <StatusPill label={`User #${scoreReport.user_id}`} tone="info" />
            </div>

            <div className="metric-strip">
              <Metric label="Total score" value={String(Math.round(scoreReport.total_score))} />
              <Metric label="English score" value={String(Math.round(scoreReport.total_english_score))} />
              <Metric label="Phone" value={scoreReport.phone} />
            </div>

            <div className="trait-strip">
              {[scoreReport.trait1, scoreReport.trait2, scoreReport.trait3, scoreReport.trait4, scoreReport.trait5].map((trait) => (
                <span className="trait-chip" key={trait}>
                  {trait}
                </span>
              ))}
            </div>

            <div className="hero-links">
              {scoreReport.cv ? (
                <a href={assetUrl(apiBaseUrl, scoreReport.cv)} target="_blank" rel="noreferrer">
                  Open CV
                </a>
              ) : null}
            </div>

            <div className="question-cards">
              {scoreReport.questions.map((question, index) => (
                <article className="mini-card report-card" key={`${question.question}-${index}`}>
                  <span className="card-index">Response {index + 1}</span>
                  <h3>{question.question}</h3>
                  <p>{question.summary || "No summary available yet."}</p>
                  <div className="report-meta">
                    <StatusPill label={`Emotion: ${question.emotion ?? "Neutral"}`} tone="info" />
                    <StatusPill label={`Relevance: ${question.relevance ?? 0}`} tone="success" />
                  </div>
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
          <EmptyState message="Run scoring or load a report to inspect interview output here." />
        )}
      </Panel>
    </div>
  );
}
