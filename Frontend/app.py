from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Logging
logging.basicConfig(level=logging.DEBUG)

# Load Sentence-BERT model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Failed to load Sentence-BERT model: {e}")

# Constants
DB_PATH = "D:\Ollama X SpacY\knowledge_base.db"  # SQLite database path
SIMILARITY_THRESHOLD = 0.7  # Threshold for similarity matching


# --- Utility Functions ---
def connect_db():
    """Create a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def compute_embedding(text):
    """Compute embedding using Sentence-BERT."""
    return model.encode(text).reshape(1, -1)


# --- API Endpoints ---
@app.post("/chat")
async def chat_endpoint(request: Request):
    """Handle user queries."""
    try:
        data = await request.json()
        question = data.get("message", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        # Generate embedding for user query
        user_embedding = compute_embedding(question)

        # Query database for the best match
        answer, confidence = query_validated_qa(user_embedding)

        # If no valid match found, search sections as fallback
        if not answer:
            related_sections = search_sections(question)
            if related_sections:
                return {
                    "answer": "No direct match found. Here are some related sections:",
                    "related_sections": related_sections,
                    "confidence": 0.0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            else:
                return {
                    "answer": "I'm sorry, I couldn't find anything relevant.",
                    "confidence": 0.0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

        return {
            "answer": answer,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/add")
async def add_to_validated_qa(request: Request):
    """Add a new question-answer pair to the database."""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        if not question or not answer:
            raise HTTPException(
                status_code=400, detail="Both question and answer are required."
            )

        embedding = compute_embedding(question).tobytes()
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ValidatedQA (question, answer, embedding) VALUES (?, ?, ?)",
            (question, answer, embedding),
        )
        conn.commit()
        conn.close()

        return {"message": "Question-Answer pair added successfully."}
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")


@app.get("/list-sections")
def list_sections():
    """List all sections available in the database."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM Sections LIMIT 10;")
        sections = cursor.fetchall()
        conn.close()

        return {"sections": [{"id": sec[0], "content": sec[1]} for sec in sections]}
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")


@app.get("/search-sections")
def search_sections(query: str):
    """Search for terms in the sections table."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content FROM Sections WHERE content LIKE ? LIMIT 10;",
            (f"%{query}%",),
        )
        results = cursor.fetchall()
        conn.close()

        return [{"id": row[0], "content": row[1]} for row in results]
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")


# --- Helper Functions ---
def query_validated_qa(user_embedding):
    """Query the ValidatedQA table for the best match."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
        rows = cursor.fetchall()

        max_similarity = 0.0
        best_answer = None

        for db_question, db_answer, db_embedding in rows:
            db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(
                1, -1
            )
            similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = db_answer

        conn.close()

        if max_similarity >= SIMILARITY_THRESHOLD:
            return best_answer, float(max_similarity)
        return None, 0.0
    except sqlite3.Error as e:
        logging.error(f"Database query error: {e}")
        return None, 0.0
