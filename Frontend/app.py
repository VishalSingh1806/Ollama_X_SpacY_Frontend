from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import ollama

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

DB_PATH = r"D:\Ollama X SpacY\knowledge_base.db"
SIMILARITY_THRESHOLD = 0.7
OLLAMA_MODEL = "llama2"

# Load spaCy model for semantic embeddings
nlp = spacy.load("en_core_web_md")

# Predefined intents for greeting and ending
GREETING_INTENTS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "howdy", "hi there"
]
ENDING_INTENTS = [
    "thanks", "bye", "goodbye", "that's all", "thank you", "see you later", "catch you later"
]

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    question = data.get("message", "").strip()

    if not question:
        return {
            "answer": "Please provide a valid question.",
            "source": "none",
            "confidence": 0.0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    processed_answer, source, confidence = handle_query(question)
    return {
        "answer": processed_answer,
        "source": source,
        "confidence": float(confidence),  # Convert numpy.float32 to Python float
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def generate_ollama_answer(context, question):
    """Generate an answer using Ollama."""
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely and realistically."
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return response.get('message', {}).get('content', "Sorry, I couldn't generate a response.")

def query_database(question):
    """Query the knowledge base for a matching question."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        user_embedding = np.array(nlp(question).vector).reshape(1, -1)

        cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
        rows = cursor.fetchall()

        if not rows:
            return None, "none", 0.0

        max_similarity = 0.0
        best_answer = None

        for db_question, db_answer, db_embedding in rows:
            db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)
            similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = db_answer

        if max_similarity >= SIMILARITY_THRESHOLD:
            return best_answer, "database", max_similarity
        else:
            return None, "none", max_similarity
    except sqlite3.Error as e:
        return f"Database error: {str(e)}", "error", 0.0
    finally:
        conn.close()

def handle_query(question):
    """Handle a question by querying the database and using Ollama as fallback."""
    # Handle greetings
    if question.lower() in GREETING_INTENTS:
        return "Hello! How can I assist you today?", "greeting", 1.0

    # Handle ending intents
    if question.lower() in ENDING_INTENTS:
        return "You're welcome! Have a great day!", "ending", 1.0

    # Query the database
    db_answer, source, confidence = query_database(question)

    if db_answer and confidence >= SIMILARITY_THRESHOLD:
        return db_answer, source, confidence

    if db_answer:
        # Refine low-confidence answers using Ollama
        refined_answer = generate_ollama_answer(db_answer, question)
        return refined_answer, "refined (ollama)", confidence

    # Generate a completely new answer if no match is found
    new_answer = generate_ollama_answer("This is a general knowledge assistant.", question)
    return new_answer, "ollama", 0.5  # Default confidence for generated answers
