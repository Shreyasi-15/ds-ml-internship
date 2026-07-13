# AcousticSpace frontend

React + TypeScript dashboard for the Week 1 feature-extraction MVP.

## Run

```bash
npm install
copy .env.example .env
npm run dev
```

The default backend is `http://127.0.0.1:8000`. Change
`VITE_API_BASE_URL` in `.env` when the API uses another host or port.

## Verify

```bash
npm run build
npm run lint
```

The interface validates the selected file, uploads it to FastAPI, and renders
the Week 1 acoustic feature report. It intentionally does not claim that an
audio clip is real or fake; the classifier belongs to Week 2.
