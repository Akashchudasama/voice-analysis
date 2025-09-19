import os
import glob
import sqlite3
import streamlit as st
import librosa
import numpy as np
import zipfile
from pathlib import Path
import time
from scipy.spatial.distance import cosine
import librosa.sequence
import io

# ---------------- CONFIG ----------------
DB_FILE = "voice_data.db"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("uploaded_audios", exist_ok=True)  # for the Voice Matcher feature

ALLOWED_AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

# ---------------- DB HELPERS ----------------
def init_db():
    """Initializes the SQLite database and creates the 'voices' table."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS voices
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  filename TEXT NOT NULL UNIQUE)''')
    conn.commit()
    conn.close()

init_db()

def save_voice(name, file_path):
    """Saves a voice entry (name and file path) to the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO voices (name, filename) VALUES (?, ?)", (name, file_path))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

def get_voices_by_name(name):
    """Fetches voice entries from the database by a given name (partial match)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, filename FROM voices WHERE name LIKE ?", (f"%{name}%",))
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_db_rows():
    """Fetches all voice entries from the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name, filename FROM voices")
    rows = c.fetchall()
    conn.close()
    return rows

def file_registered_in_db(file_path):
    """Checks if a file path is already registered in the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT 1 FROM voices WHERE filename = ?", (file_path,))
    found = c.fetchone() is not None
    conn.close()
    return found

# ---------------- FILE HELPERS ----------------
def unique_path(target_path):
    """Generates a unique file path to prevent overwriting existing files."""
    base = Path(target_path)
    parent = base.parent
    stem = base.stem
    ext = base.suffix
    counter = 1
    p = base
    while p.exists():
        p = parent / f"{stem}_{counter}{ext}"
        counter += 1
    return str(p)

def save_uploaded_file(uploaded_file, dest_dir):
    """Saves an uploaded Streamlit file to a destination directory."""
    safe_name = os.path.basename(uploaded_file.name)
    dest_path = os.path.join(dest_dir, safe_name)
    dest_path = unique_path(dest_path)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path

def extract_audio_from_zip(zip_path, dest_dir=UPLOAD_DIR):
    """Extracts audio files from a ZIP archive."""
    saved = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if member.endswith("/") or member.startswith("__MACOSX/"):
                continue
            name = os.path.basename(member)
            if not name:
                continue
            lower = name.lower()
            if any(lower.endswith(ext) for ext in ALLOWED_AUDIO_EXTS):
                try:
                    data = z.read(member)
                    dest_path = os.path.join(dest_dir, name)
                    dest_path = unique_path(dest_path)
                    with open(dest_path, "wb") as out:
                        out.write(data)
                    saved.append(dest_path)
                except Exception:
                    continue
    return saved

def scan_uploads_for_audio():
    """Scans the uploads directory for all supported audio files."""
    files = []
    for ext in ALLOWED_AUDIO_EXTS:
        files += glob.glob(os.path.join(UPLOAD_DIR, f"**/*{ext}"), recursive=True)
    files = sorted(list({os.path.abspath(f) for f in files}))
    return files

def sync_uploads_to_db():
    """Synchronizes files from the uploads directory to the database."""
    files = scan_uploads_for_audio()
    count = 0
    for f in files:
        if not file_registered_in_db(f):
            name = Path(f).stem
            save_voice(name, f)
            count += 1
    return count

# ---------------- AUDIO / COMPARISON HELPERS ----------------
def load_audio(source, sr=16000, mono=True):
    """Loads an audio file from a path or BytesIO object."""
    if isinstance(source, (str, bytes)):
        y, _ = librosa.load(source, sr=sr, mono=mono)
        name = source if isinstance(source, str) else "BytesIO"
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        y, _ = librosa.load(source, sr=sr, mono=mono)
        name = "BytesIO"
    else:
        raise TypeError("source must be a file path or BytesIO object")
    if y.size == 0:
        raise ValueError(f"Loaded audio is empty: {name}")
    return y, sr

def extract_mfcc(y, sr, n_mfcc=13, hop_length=512):
    """Extracts Mel-frequency cepstral coefficients (MFCCs) from audio."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc

def embedding_from_mfcc(mfcc):
    """Creates a voice embedding from MFCCs using mean and standard deviation."""
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    emb = np.concatenate([mean, std])
    return emb

def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    d = cosine(a, b)
    if np.isnan(d):
        return 0.0
    return 1.0 - d

def normalized_dtw_distance(m1, m2):
    """Calculates a normalized similarity score using Dynamic Time Warping (DTW)."""
    D, wp = librosa.sequence.dtw(X=m1, Y=m2, metric='euclidean')
    cost = D[-1, -1]
    path_length = len(wp)
    if path_length == 0:
        return 1.0
    norm_cost = cost / path_length
    sim = np.exp(-norm_cost / 50.0)
    return float(sim)

def compare_files(path1, path2, in_memory=False):
    """Compares two audio files and returns a similarity score and verdict."""
    try:
        y1, sr1 = load_audio(path1 if not in_memory else path1)
        y2, sr2 = load_audio(path2 if not in_memory else path2)
        mfcc1 = extract_mfcc(y1, sr1)
        mfcc2 = extract_mfcc(y2, sr2)
        emb1 = embedding_from_mfcc(mfcc1)
        emb2 = embedding_from_mfcc(mfcc2)
        cos_sim = cosine_similarity(emb1, emb2)
        dtw_sim = normalized_dtw_distance(mfcc1, mfcc2)
        combined = 0.6 * cos_sim + 0.4 * dtw_sim
        verdict = "SAME PERSON" if combined >= 0.65 else "DIFFERENT PERSON"
        return {
            "cosine_similarity": float(cos_sim),
            "dtw_similarity": float(dtw_sim),
            "combined_score": float(combined),
            "verdict": verdict
        }
    except Exception as e:
        st.error(f"Error during comparison: {e}")
        return None

def load_mfcc_mean(path, n_mfcc=20):
    """Loads an audio file and returns the mean of its MFCCs."""
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        if y.size < 10:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception:
        return None

def compare_voice(query_path, candidate_path):
    """Compares two voices using the Euclidean distance of their MFCC means."""
    v1 = load_mfcc_mean(query_path)
    v2 = load_mfcc_mean(candidate_path)
    if v1 is None or v2 is None:
        return None
    try:
        dist = float(np.linalg.norm(v1 - v2))
        return dist
    except Exception:
        return None
    
def get_verdict_from_distance(distance):
    """
    Translates a numerical distance score into a user-friendly verdict
    using fixed thresholds.
    """
    SAME_THRESHOLD = 50.0
    LIKELY_SAME_THRESHOLD = 80.0
    
    if distance <= SAME_THRESHOLD:
        return "Same Person"
    elif distance <= LIKELY_SAME_THRESHOLD:
        return "Likely Same Person"
    else:
        return "Different Person"

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Unified Voice App", page_icon="🎤", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🎙️ Unified Voice Application</h1>", unsafe_allow_html=True)

# --- UI Styling and Design ---
st.markdown("""
<style>
    .reportview-container {
        background: #f5f7fa;
        padding: 20px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4CAF50, #81C784);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        background: linear-gradient(45deg, #388E3C, #66BB6A);
    }
    .st-expanderHeader {
        background-color: #fafafa;
        color: #333;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 14px;
        font-weight: 600;
        font-size: 16px;
    }
    .stRadio > label {
        background-color: #fafafa;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 6px 0;
        border: 1px solid #e0e0e0;
        transition: all 0.2s ease;
    }
    .stRadio > label:hover {
        background-color: #f0f0f0;
        border-color: #bdbdbd;
    }
    .stRadio [data-baseweb="radio"] {
        padding-right: 12px;
    }
    .stRadio [data-baseweb="radio"] span:first-child {
        border-color: #4CAF50;
    }
    h2, h3 {
        color: #2E7D32;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 8px;
        margin-bottom: 24px;
        font-weight: 600;
    }
    .main-card {
        background-color: white;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e8ecef;
    }
    .stFileUploader label {
        font-weight: 500;
        color: #37474F;
    }
    .stMetric label {
        font-weight: 600;
        color: #263238;
    }
</style>
""", unsafe_allow_html=True)

with st.spinner("🔄 Syncing uploads folder with database..."):
    new_count = sync_uploads_to_db()
    time.sleep(0.3)

# --- Main Menu ---
menu = ["🗂️ Voice Library Management", "🎤 Direct Comparison"]
choice = st.sidebar.selectbox("Select a Feature", menu, help="Choose between managing your voice library or comparing two audio files directly.")

# --- Voice Data App Page ---
if choice == "🗂️ Voice Library Management":
    st.header("🗂️ Voice Data Management & Matching")
    st.markdown("A powerful tool to **manage, search, and match** voice data from your audio library.", unsafe_allow_html=True)

    st.markdown("---")

    app_menu = ["➕ Add Voice Data", "🔍 Search & Match", "🛠️ Manage Library"]
    app_choice = st.sidebar.radio("Data App Menu", app_menu, help="Navigate through the Voice Data App sections: add files, search/match, or manage the library.")

    # Add Data
    if app_choice == "➕ Add Voice Data":
        with st.container(border=True):
            st.subheader("Add Voice Data to Your Library")
            st.info("Upload individual audio files or a ZIP archive containing multiple audio files to add to your voice library.")
            
            uploaded = st.file_uploader("Upload audio files or ZIP", type=list(x.strip(".") for x in ALLOWED_AUDIO_EXTS) + ["zip"], accept_multiple_files=True, help="Supported formats: WAV, MP3, OGG, FLAC, M4A, or ZIP.")

            col_add, col_info = st.columns([1, 2])
            with col_add:
                submitted = st.button("Save Uploaded Files", help="Process and save the uploaded files to the library.")
            with col_info:
                st.write("")  # Spacer
                if uploaded:
                    st.write(f"**{len(uploaded)}** file(s) ready for processing.")

            if submitted:
                if not uploaded:
                    st.error("Please upload at least one file before saving.")
                else:
                    with st.spinner("Processing files... This may take a moment for large uploads."):
                        saved_all = []
                        for up in uploaded:
                            if up.name.lower().endswith(".zip"):
                                tmp_zip = save_uploaded_file(up, dest_dir=UPLOAD_DIR)
                                extracted = extract_audio_from_zip(tmp_zip, dest_dir=UPLOAD_DIR)
                                saved_all.extend(extracted)
                                try:
                                    os.remove(tmp_zip)
                                except Exception:
                                    pass
                            else:
                                saved_path = save_uploaded_file(up, dest_dir=UPLOAD_DIR)
                                saved_all.append(saved_path)

                    registered = 0
                    for path in saved_all:
                        if os.path.isfile(path) and any(path.lower().endswith(ext) for ext in ALLOWED_AUDIO_EXTS):
                            name = Path(path).stem
                            save_voice(name, os.path.abspath(path))
                            registered += 1

                    if registered > 0:
                        st.success(f"Successfully added **{registered}** new file(s) to your voice library!")
                    else:
                        st.warning("No new audio files were registered. Check file types or if they are duplicates.")
                    st.info(f"Total audio files in uploads folder: **{len(scan_uploads_for_audio())}**")
                    time.sleep(1)
                    st.rerun()

    # Find Data
    elif app_choice == "🔍 Search & Match":
        with st.container(border=True):
            st.subheader("Search or Match Voice Data")
            st.info("Search your voice library by name or upload a sample to find matching voices.")

            method = st.radio("Search By", ["Name", "Voice File"], horizontal=True, help="Choose to search by name or match by uploading a voice sample.")

            if method == "Name":
                search_name = st.text_input("Enter a Name (e.g., 'john' or 'sample_1')", help="Enter a partial or full name to search the library.")
                if st.button("Search for Names", help="Find audio files with matching names."):
                    if not search_name.strip():
                        st.warning("Please enter a name to search.")
                    else:
                        rows = get_voices_by_name(search_name.strip())
                        if not rows:
                            st.warning(f"No results found for **'{search_name}'**.")
                        else:
                            st.success(f"Found **{len(rows)}** match(es):")
                            for idx, (id, name, filepath) in enumerate(rows):
                                with st.expander(f"**Match #{idx+1}: {name}**"):
                                    st.code(filepath)
                                    try:
                                        st.audio(filepath, format="audio/wav")
                                    except Exception:
                                        st.error("Could not play this file.")

            else:  # Voice File Matching
                st.write("Upload a voice sample to match against your library.")
                uploaded_voice = st.file_uploader("Upload Query Voice File", type=list(x.strip(".") for x in ALLOWED_AUDIO_EXTS), help="Upload an audio file to compare against the library.")

                col_match_1, col_match_2 = st.columns(2)
                with col_match_1:
                    match_mode = st.radio("Match Against", ["Database", "Uploads Folder", "Both"], index=2, help="Choose which source to search: database, uploads folder, or both.")
                with col_match_2:
                    top_k = st.slider("Show Top K Matches", 1, 10, 3, help="Select how many top matches to display.")

                if st.button("Start Matching", help="Compare the uploaded voice against the selected source."):
                    if not uploaded_voice:
                        st.error("Please upload a query audio file first.")
                    else:
                        with st.spinner("🔍 Analyzing and matching voices..."):
                            query_path = save_uploaded_file(uploaded_voice, dest_dir=UPLOAD_DIR)
                            candidates = []
                            if match_mode in ("Database", "Both"):
                                candidates.extend(get_all_db_rows())
                            if match_mode in ("Uploads Folder", "Both"):
                                files = scan_uploads_for_audio()
                                folder_rows = [(Path(f).stem, f) for f in files]
                                combined = {os.path.abspath(path): name for name, path in (candidates + folder_rows)}
                                candidates = [(n, p) for p, n in combined.items()]

                            if not candidates:
                                st.warning("No candidate files found to match against.")
                            else:
                                results = []
                                for name, path in candidates:
                                    dist = compare_voice(query_path, path)
                                    if dist is not None:
                                        results.append((name, path, dist))

                                results.sort(key=lambda x: x[2])

                                if results:
                                    st.subheader("Match Results")
                                    shown = results[:top_k]
                                    for idx, (name, path, dist) in enumerate(shown, start=1):
                                        verdict = get_verdict_from_distance(dist)
                                        
                                        color = "green" if verdict == "Same Person" else "orange" if verdict == "Likely Same Person" else "red"
                                        st.markdown(f"**Match #{idx}** — Name: **{name}**")
                                        st.markdown(f"**Verdict:** <span style='color:{color};font-weight:bold;'>{verdict}</span>", unsafe_allow_html=True)
                                        
                                        with st.expander("Details"):
                                            st.write(f"Distance Score: `{dist:.2f}`")
                                            st.code(path)
                                            try:
                                                st.audio(path, format="audio/wav")
                                            except Exception:
                                                st.error("Could not play this file.")
                                else:
                                    st.error("No valid comparisons could be made. Check your audio file format.")
                                
                            try:
                                os.remove(query_path)
                            except Exception:
                                pass

    # Manage
    elif app_choice == "🛠️ Manage Library":
        with st.container(border=True):
            st.subheader("Manage Your Voice Library")
            st.info("Perform maintenance tasks on your database and uploaded files.")
            st.markdown("---")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Registered Files in Database", len(get_all_db_rows()))
            with col_info2:
                st.metric("Files in Uploads Folder", len(scan_uploads_for_audio()))
            
            st.markdown("---")
            
            st.warning("⚠️ **Danger Zone:** These actions are permanent and cannot be undone.")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                if st.button("Force Sync Database", help="Add all audio files from the uploads folder to the database.", type="secondary"):
                    added = sync_uploads_to_db()
                    st.success(f"Added **{added}** new file(s) to the database.")
                    st.rerun()
            with col_m2:
                if st.button("Clear Database", help="Delete all entries from the database (keeps audio files).", type="secondary"):
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("DELETE FROM voices")
                    conn.commit()
                    conn.close()
                    st.success("Database has been cleared.")
                    st.rerun()
            with col_m3:
                if st.button("Delete All Files", help="Delete all audio files from the uploads folder.", type="secondary"):
                    files = scan_uploads_for_audio()
                    deleted = 0
                    for f in files:
                        try:
                            os.remove(f)
                            deleted += 1
                        except Exception:
                            pass
                    st.success(f"Deleted **{deleted}** file(s) from the uploads folder.")
                    st.rerun()

# --- Voice Matcher Page ---
elif choice == "🎤 Direct Comparison":
    with st.container(border=True):
        st.header("🎤 Two-File Voice Comparison")
        st.markdown("Compare two audio files directly to determine if they are from the same person.", unsafe_allow_html=True)
        st.markdown("---")

        path1, path2 = None, None
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("First Audio File")
            file1 = st.file_uploader("🎵 Upload first audio file", type=["wav", "mp3", "flac", "ogg", "m4a"], key="file1", help="Supported formats: WAV, MP3, FLAC, OGG, M4A")
            if file1:
                st.audio(file1, format="audio/wav")
                path1 = save_uploaded_file(file1, dest_dir="uploaded_audios")

        with col2:
            st.subheader("Second Audio File")
            file2 = st.file_uploader("🎵 Upload second audio file", type=["wav", "mp3", "flac", "ogg", "m4a"], key="file2", help="Supported formats: WAV, MP3, FLAC, OGG, M4A")
            if file2:
                st.audio(file2, format="audio/wav")
                path2 = save_uploaded_file(file2, dest_dir="uploaded_audios")

        st.markdown("---")

        if st.button("Compare Voices", help="Compare the two uploaded audio files for similarity."):
            if not path1 or not path2:
                st.warning("⚠️ Please upload **two audio files** to start the comparison.")
            else:
                with st.spinner("🔄 Comparing files... This may take a moment."):
                    res = compare_files(path1, path2)
                    if res:
                        st.subheader("📊 Comparison Results")
                        st.info(f"**Verdict:** The voices are likely from the **{res['verdict']}**.")
                        st.write("---")
                        st.markdown(f"""
                            - **Cosine Similarity:** `{res['cosine_similarity']:.3f}`
                            - **DTW Similarity:** `{res['dtw_similarity']:.3f}`
                            - **Combined Score:** `{res['combined_score']:.3f}`
                        """)
                        st.write("**Note:** A combined score of **0.65 or higher** suggests the same person.")

                try:
                    os.remove(path1)
                    os.remove(path2)
                except Exception:
                    pass