from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

# Initialize FastAPI app
app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Mount static files (Frontend)
STATIC_FILES_DIR = "/root/Ollama_X_SpacY_Frontend/Frontend"
app.mount("/", StaticFiles(directory=STATIC_FILES_DIR, html=True), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load Sentence-BERT model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Sentence-BERT model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load Sentence-BERT model: {e}")

# Constants
DB_PATH = "/root/Ollama_X_SpacY_Frontend/Frontend/knowledge_base.db"
SIMILARITY_THRESHOLD = 0.7  # Similarity threshold for question matching


# --- Utility Functions ---
def connect_db():
    """Connect to the SQLite database."""
    try:
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed.")


def compute_embedding(text):
    """Compute embedding for a given text using Sentence-BERT."""
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

        # Query the database for the best match
        answer, confidence = query_validated_qa(user_embedding)

        # Fallback: Suggest related sections if no valid match is found
        if not answer:
            related_sections = search_sections(question)
            return {
                "answer": "No direct match found. Here are some related sections:" if related_sections else "I'm sorry, I couldn't find anything relevant.",
                "related_sections": related_sections if related_sections else [],
                "confidence": 0.0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        return {
            "answer": answer,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/add")
async def add_to_validated_qa(request: Request):
    """Add a new question-answer pair to the database."""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()

        if not question or not answer:
            raise HTTPException(status_code=400, detail="Both question and answer are required.")

        # Compute embedding and add to the database
        embedding = compute_embedding(question).tobytes()
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO ValidatedQA (question, answer, embedding) VALUES (?, ?, ?)", (question, answer, embedding))
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


def query_validated_qa(user_embedding):
    """Query the ValidatedQA table for the best match."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
        rows = cursor.fetchall()

        max_similarity = 0.0
        best_answer = None

        for _, db_answer, db_embedding in rows:
            db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)
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


def search_sections(query: str):
    """Search for terms in the sections table."""
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM Sections WHERE content LIKE ? LIMIT 10;", (f"%{query}%",))
        results = cursor.fetchall()
        conn.close()

        return [{"id": row[0], "content": row[1]} for row in results]
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []

@app.on_event("startup")
async def list_routes():
    for route in app.routes:
        print(f"Path: {route.path}, Methods: {route.methods}")
