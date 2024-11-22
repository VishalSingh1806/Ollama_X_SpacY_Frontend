import sqlite3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

DB_PATH = r"D:\EPR Data\knowledge_base.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load SentenceTransformer model
sbert_model = SentenceTransformer(MODEL_NAME)


def regenerate_embeddings():
    """
    Regenerate all embeddings in the database using SBERT.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch all questions
    cursor.execute("SELECT id, question FROM ValidatedQA")
    rows = cursor.fetchall()

    # Regenerate embeddings
    for row_id, question in rows:
        # Generate new SBERT embedding
        embedding = sbert_model.encode(question, convert_to_tensor=False)
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

        # Update database with new embedding
        cursor.execute(
            "UPDATE ValidatedQA SET embedding = ? WHERE id = ?",
            (embedding_blob, row_id),
        )

    conn.commit()
    conn.close()
    print("All embeddings updated with SBERT.")


def query_database(question):
    """
    Query the database using SBERT embeddings and cosine similarity.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Generate user embedding using SBERT
        user_embedding = sbert_model.encode(question, convert_to_tensor=True)

        # Fetch all database embeddings
        cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
        rows = cursor.fetchall()

        max_similarity = 0.0
        best_answer = None

        for db_question, db_answer, db_embedding_blob in rows:
            db_embedding = np.frombuffer(db_embedding_blob, dtype=np.float32)
            db_embedding_tensor = torch.tensor(db_embedding).unsqueeze(0)

            # Compute cosine similarity
            similarity = util.cos_sim(user_embedding, db_embedding_tensor)[0][0].item()

            if similarity > max_similarity:
                max_similarity = similarity
                best_answer = db_answer

        if max_similarity >= 0.7:  # Confidence threshold
            return best_answer, "database", max_similarity
        else:
            return "No relevant answer found.", "none", max_similarity
    finally:
        conn.close()


def test_system():
    """
    Test the system with sample questions.
    """
    print("Training system with existing database questions...")
    regenerate_embeddings()  # Ensure all embeddings are updated
    print("System training completed.")

    print("Testing system with sample questions:")

    # Test questions
    test_questions = [
        "What is EPR?",
        "Explain EPR to me",
        "Tell me about Extended Producer Responsibility",
        "What is the purpose of EPR?",
    ]

    for question in test_questions:
        start_time = datetime.now()

        # Query the database
        answer, source, confidence = query_database(question)

        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        # Print results
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Source: {source}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Time Taken: {elapsed_time:.2f}s\n")


if __name__ == "__main__":
    test_system()
