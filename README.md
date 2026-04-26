# ClipTrace – Intelligent Video Identity and Tracking System

> A segment-based, dual-layer video identity engine combining deterministic fingerprinting,
> tamper-proof hash chains, and AI-driven matching to track and authenticate video content
> across transformations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  Dashboard │ Register │ Match Clip │ Verify │ Monetize       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP /api/*
┌──────────────────────▼──────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Segmenter  │  │ Fingerprinter│  │   ML Embedder     │  │
│  │  (OpenCV)   │  │ pHash+Motion │  │ (MobileNetV3)     │  │
│  │             │  │ +Audio MFCC  │  │                   │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬──────────┘  │
│         └────────────────┴───────────────────┘             │
│                          │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │              Matching Engine                         │   │
│  │  FAISS Vector Index │ Cosine Similarity              │   │
│  │  Temporal Continuity │ Confidence Scoring            │   │
│  │  Mixed-Content Detection                             │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │              SQLite Database                         │   │
│  │  videos │ segments │ monetization_rules              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Hash Chain (Capture-Time Proof)            │   │
│  │  SHA-256 chain per segment │ HMAC signatures         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Innovations

| Feature | Description |
|---|---|
| **Capture-Time Proof** | SHA-256 hash chain built at registration — each segment links to the previous, making any modification detectable |
| **Dual Verification** | Combines tamper-proof chain (integrity) + perceptual fingerprint (identity) |
| **Public Verify API** | `/api/verify/clip` — open endpoint anyone can call to check a clip's origin |
| **Smart Monetization** | Revenue split weighted by confidence × matched segments, not just block/claim |
| **ML Embeddings** | MobileNetV3 visual embeddings merged with deterministic features for robustness under heavy edits |
| **Mixed-Content Detection** | Detects when a single clip contains footage from multiple source videos |

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ |
| ffmpeg | Any recent (must be on PATH) |

Install ffmpeg on Windows:
```
winget install ffmpeg
```

---

## Quick Start

```bat
REM One command — opens backend + frontend in separate windows
start_all.bat
```

Or separately:
```bat
start_backend.bat    REM http://localhost:8000
start_frontend.bat   REM http://localhost:3000
```

Interactive API docs: **http://localhost:8000/docs**

---

## Project Structure

```
cliptrace/
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── requirements.txt
│   └── app/
│       ├── api/
│       │   ├── videos.py          # Register / list / delete videos
│       │   ├── match.py           # Clip matching endpoint
│       │   ├── verify.py          # Public authenticity API + chain check
│       │   └── monetization.py    # Revenue routing + rule management
│       ├── core/
│       │   ├── config.py          # Constants and settings
│       │   └── chain.py           # Hash chain (Capture-Time Proof)
│       ├── db/
│       │   └── database.py        # SQLAlchemy models + async engine
│       ├── processing/
│       │   ├── segmenter.py       # Video → 1-second segments
│       │   └── fingerprint.py     # Visual + motion + audio fingerprint
│       ├── matching/
│       │   ├── engine.py          # Similarity scoring + aggregation
│       │   └── vector_index.py    # FAISS in-memory index
│       └── ml/
│           └── embedder.py        # MobileNetV3 visual embeddings
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # Registered videos + chain status
│   │   │   ├── Register.jsx       # Upload + register original video
│   │   │   ├── Match.jsx          # Upload clip → find source
│   │   │   ├── Verify.jsx         # Public authenticity check
│   │   │   └── Monetization.jsx   # Revenue routing UI
│   │   ├── components/
│   │   │   ├── Nav.jsx
│   │   │   ├── DropZone.jsx
│   │   │   ├── MatchCard.jsx
│   │   │   └── ConfidenceBar.jsx
│   │   └── api/client.js          # Axios API helpers
│   └── package.json
├── start_backend.bat
├── start_frontend.bat
└── start_all.bat
```

---

## API Reference

### Videos
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/videos/register` | Register original video (multipart: file, title, owner) |
| GET  | `/api/videos/` | List all registered videos |
| DELETE | `/api/videos/{id}` | Delete a video and its segments |

### Matching
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/match/` | Upload clip → returns ranked source matches |

### Verification
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/verify/clip` | Public authenticity check — original / edited / mixed |
| GET  | `/api/verify/chain/{video_id}` | Verify hash chain integrity |

### Monetization
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/monetization/route` | Upload clip → revenue allocation plan |
| GET  | `/api/monetization/rules` | List all monetization rules |
| PUT  | `/api/monetization/rules/{video_id}` | Update action (monetize/block/allow) |

---

## Fingerprint Pipeline

Each 1-second segment produces:

```
Visual (112-dim)  = pHash bits (64) + colour histogram (48)
Motion  (32-dim)  = optical-flow magnitude histogram
Audio   (40-dim)  = MFCC mean (20) + chroma mean (20)
                  ──────────────────────────────────
Combined (256-dim) → L2-normalised embedding
                  +
ML embedding (576-dim, MobileNetV3) merged when available
```

Similarity search uses FAISS IndexFlatIP (inner product on L2-normalised vectors = cosine similarity).

---

## Confidence Scoring

```
confidence = base_similarity × coverage_ratio + continuity_bonus

where:
  base_similarity  = mean cosine similarity of matched segments
  coverage_ratio   = matched_segments / total_query_segments
  continuity_bonus = 0.05 × count_of_consecutive_matching_segments
```

Mixed-content is flagged when two sources cover non-overlapping query segments (overlap < 30%).
#   C l i p T r a c e  
 