# AcousticSpace

Deepfake audio detection via Room Impulse Response (RIR) mismatch analysis.

Most deepfake audio detectors focus on vocal artifacts (robotic tones, unusual
inflections) — but modern generative AI easily fools those checks. AcousticSpace
takes a different approach: instead of just listening to the voice, it
mathematically isolates the background **Room Impulse Response** (how sound
reflects off walls) and compares it against the voice's claimed environment.
If the acoustic reflections don't match the alleged background, the clip gets
flagged as likely synthetic.

## Project Status

**Week 1 (Core Setup & Data)** — Complete
- FastAPI server running and tested
- Librosa pipeline extracting RT60, reverb ratio, breathing-band energy, and mel-spectrograms
- Dataset manifest builder for ASVspoof-style datasets
- React + TypeScript upload dashboard, connected live to the backend

**Week 2 (Baseline Model + Visualization)** — In progress
- Waveform visualization
- Baseline CNN/Transformer classifier

## Architecture

| Module | Tech | Purpose |
|---|---|---|
| Audio Processing Pipeline | Python, Librosa | Extracts RIR/reverb features and spectrograms |
| Transformer Classifier | PyTorch, HuggingFace | Detects vocal-cadence vs. spatial-acoustics mismatch |
| API Gateway | FastAPI | Serves the pipeline/model for real-time inference |
| Analyst Dashboard | React, TypeScript | Upload interface + results visualization |

## Requirements

- **Python 3.12** (tested on 3.12.10)
- **Node.js 24.x** (tested on v24.18.0)
- **npm 11.x** (tested on 11.16.0)

## Project Structure

```
AcousticSpace/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI server + /extract-features endpoint
│   │   ├── audio_pipeline.py  # Librosa RIR/reverb/spectrogram extraction
│   │   ├── dataset.py         # ASVspoof manifest builder
│   │   └── model.py           # Baseline CNN classifier (Week 2)
│   ├── tests/
│   │   └── test_pipeline.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx            # Main dashboard UI
│       └── index.css
├── dataset/                    # ASVspoof data (not committed)
├── docs/
└── notebooks/
```

## Setup & Running Locally

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`. Interactive API docs at
`http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

**Note:** both servers must be running simultaneously, in separate terminals,
for the dashboard to successfully analyze audio.

## Running Tests

```bash
cd backend
venv\Scripts\activate
pytest tests/ -v
```

## Known Limitations

- RT60 (reverberation time) estimation uses a simplified Schroeder
  backward-integration method; on longer or noisier clips it can produce
  unrealistic values. This is a placeholder proxy pending refinement.
- The classifier (`model.py`) is an untrained baseline architecture — real
  accuracy numbers require training on labeled ASVspoof data (Week 2).

## Author

Shreyasi — Data Science & Machine Learning Intern, Infotact Solutions