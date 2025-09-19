# 🎙️ Unified Voice Application

A **Streamlit-based web application** for managing, searching, and matching voice data. This project allows users to upload audio files, maintain a voice library, and compare voices to determine if they belong to the same person.

---

## **Features**

### 1. Voice Library Management
- Upload individual audio files or ZIP archives containing multiple files.
- Sync and store uploaded audio files in a SQLite database.
- Search by name or match uploaded voice samples with your library.
- Manage the library with options to force sync, clear database, or delete all uploaded files.

### 2. Direct Voice Comparison
- Upload two audio files and compare them directly.
- Calculates similarity using:
  - **MFCC embeddings**
  - **Cosine similarity**
  - **Dynamic Time Warping (DTW)**
- Provides a combined score and verdict:
  - **Same Person**
  - **Likely Same Person**
  - **Different Person**

---

## **Supported Audio Formats**
- WAV
- MP3
- OGG
- FLAC
- M4A

---

## **Installation & Setup**

1. **Clone the repository**
```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
