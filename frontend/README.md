# OptiMLFlow Next.js Frontend

Minimal, sleek Next.js App Router frontend for OptiMLFlow.

## Setup

1. Install dependencies

```bash
npm install
```

2. Configure environment

```bash
cp .env.example .env.local
```

3. Run development server

```bash
npm run dev
```

The app runs at http://localhost:3000.

## Backend Requirement

Start the FastAPI backend from the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

## Implemented

- Upload dataset and initialize backend session.
- Dataset overview charts (dtype and missing values).
- Task detection and training configuration.
- Async training via `/api/v2/jobs/train` with polling.
- Results summary cards, performance chart, and score table.
- Domain workspace route scaffold.

## Notes

- API base URL is controlled by `NEXT_PUBLIC_API_BASE_URL`.
- This frontend targets existing contracts in `backend/main.py`.
