# Interview Engine

Interview Engine is a FastAPI + React application for interview screening. An HR user creates a job, an applicant uploads a CV and interview videos, and the system generates a scored report with transcript-based summaries, relevance scoring, English scoring, emotion analysis, and personality traits.

## What It Does

- Create and review jobs with three interview prompts
- Accept applicant CVs and one video answer per prompt
- Run asynchronous scoring in the background
- Show a report with:
  - overall score
  - English score
  - answer summaries
  - answer relevance
  - emotion labels
  - personality trait labels

## Current Model Stack

- Video personality: custom X3D checkpoint in `Ai/Video_Model/X3D_Third_CheckPoint.pth`
- Video emotion: DeepFace-based analyzer
- Text personality: local BERT + per-trait models in `Ai/Text_Model/Models`
- Audio English: original checkpoint when available, otherwise Hugging Face pronunciation fallback
- LLM summaries/relevance: Gemini or Groq, selected from `.env`

## Important Limitations

- This is a decision-support prototype, not an autonomous hiring system.
- Reports can be `partial` when fallback paths are used.
- Free-tier LLM providers can rate-limit or exhaust quota mid-report.
- If `Ai/Audio_Model/EnglishModel_weights_best_epoch.pth` is missing, English scoring uses the fallback pronunciation model rather than the original custom checkpoint.
- Hiring decisions should not rely on the model output alone.

## Project Structure

- `main.py`: FastAPI app setup and startup migrations
- `Hr/`, `Job/`, `User/`: API routes and service logic
- `Ai/`: model wrappers, runtime helpers, media processing
- `frontend/`: React + Vite frontend
- `uploads/`: applicant files served by FastAPI

## Backend Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Fill `.env` from `.env.example` with your real database and LLM keys.

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Backend URLs:

- API: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Frontend Setup

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

If you want a one-command local stack, use Docker Compose:

```powershell
docker compose up --build
```

Services:

- Postgres: `localhost:5433`
- Backend API: `http://127.0.0.1:8000`
- Frontend UI: `http://127.0.0.1:5173`

Notes:

- The compose stack reads your local `.env` file at runtime.
- The backend overrides `POSTGRES_HOST=db` and `POSTGRES_PORT=5432` inside the container network.
- `uploads/` and `.cache/` are mounted as volumes so applicant files and model downloads persist.
- The frontend image is built with `VITE_API_BASE_URL=http://localhost:8000`. Change that build arg in `docker-compose.yml` if you want a different public backend URL.
- The first backend run may take longer because model assets are downloaded into `.cache/`.

## Result Quality

The API and frontend now expose result quality:

- `complete`: no known fallback path was used
- `partial`: one or more model or LLM fallbacks were used
- `failed`: scoring did not complete

When a report is partial, the UI shows explicit warnings instead of presenting the output as fully reliable.



## License

This repository is released under the MIT License. See [LICENSE](LICENSE).

Important:

- The MIT license covers the code in this repository.
- Third-party models, checkpoints, datasets, and API providers keep their own licenses and terms.
- Do not assume downloaded model weights or private checkpoints are automatically redistributable under MIT.

## Disclaimer

This repository is an AI-assisted interview analysis project for experimentation, demos, and portfolio use. It should not be treated as a final hiring authority, and it must not be used as the sole basis for employment decisions.
