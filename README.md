# Interview Engine

Interview Engine is a FastAPI + React application that helps streamline interview screening workflows. An HR user can create a job with prompts, applicants can submit a CV and recorded answers, and the system produces a structured report intended to assist review (not automate hiring).

## Features

- Create and manage jobs with interview prompts
- Accept applicant CV uploads and recorded responses
- Run background processing for scoring and analysis
- Generate a report with (depending on configuration):
  - overall score
  - English/communication indicators
  - answer summaries
  - answer relevance
  - emotion and personality signals

## Tech Stack

- Backend: FastAPI
- Frontend: React + Vite
- Background processing: async tasks
- Storage: local uploads (dev)
- AI providers: configurable via environment variables

> Note: Specific model weights/checkpoints and provider selection are intentionally not documented in detail here. See the code and configuration templates for supported options.

## Important Notes

- This is a decision-support prototype, not an autonomous hiring system.
- Outputs may be incomplete if optional models or external providers are unavailable.
- Always include human review and follow applicable laws/policies when using AI in hiring contexts.

## Project Structure

- `main.py`: FastAPI app setup
- `Hr/`, `Job/`, `User/`: API routes and service logic
- `Ai/`: model wrappers and media processing
- `frontend/`: React + Vite frontend
- `uploads/`: development upload directory (do not commit real applicant data)

## Setup (Local Development)

### Backend

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Configure environment variables:

- Copy `.env.example` to `.env`
- Fill in your own database credentials and API keys

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Backend URLs:

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### Frontend

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Start the frontend:

```powershell
npm run dev
```

Frontend URL:

- UI: `http://127.0.0.1:5173`

## Docker Compose

For a one-command local stack:

```powershell
docker compose up --build
```

Notes:

- Docker reads your local `.env` at runtime.
- The first run may take longer while dependencies/models are downloaded.
- If you deploy, update any frontend API base URL configuration to match your environment.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).

## Disclaimer

This repository is provided for experimentation, demos, and portfolio use. It should not be treated as a final hiring authority. Use responsibly and ensure compliance with applicable employment and privacy regulations.