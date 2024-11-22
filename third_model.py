import spacy
import ollama
import pdfplumber
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import deque
import sqlite3
import logging
from time import time  # Import time for tracking execution

# Enable logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize conversation history
conversation_history = deque(maxlen=5)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("knowledge_base.db", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ValidatedQA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE,
            answer TEXT,
            source TEXT
        )
    """)
    conn.commit()
    return conn

# Extract and tokenize PDF content
def extract_and_tokenize_pdf(pdf_path):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                sentences = sent_tokenize(text)
                sections.extend(sentences)
    return sections

# Summarize sections for TF-IDF vectorization
def summarize_sections(sections, max_length=150):
    return [section if len(section) <= max_length else section[:max_length].rsplit(' ', 1)[0] + "..." for section in sections]

# Setup TF-IDF vectorizer for PDF
def setup_vectorizer(pdf_path):
    sections = extract_and_tokenize_pdf(pdf_path)
    summaries = summarize_sections(sections)
    vectorizer = TfidfVectorizer()
    summary_vectors = vectorizer.fit_transform(summaries)
    return summaries, vectorizer, summary_vectors

# Load and preprocess PDF data
pdf_path = r"D:\EPR Data\Test_EPR_Data.pdf"
pdf_summaries, vectorizer, summary_vectors = setup_vectorizer(pdf_path)

# Retrieve the most relevant summary
def find_relevant_summary(query):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, summary_vectors).flatten()
    best_index = np.argmax(scores)
    return pdf_summaries[best_index], scores[best_index]

# Process text with spaCy
def process_text_with_spacy(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Save validated question and answer to the database
def save_to_db(conn, question, answer, source):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO ValidatedQA (question, answer, source) VALUES (?, ?, ?)", (question, answer, source))
    conn.commit()

# Fetch the best matching answer from the database
def get_answer_from_db(conn, query, threshold=0.75):
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer, source FROM ValidatedQA")
    rows = cursor.fetchall()
    if not rows:
        return None, 0, None

    questions = [row[0] for row in rows]
    question_vectors = vectorizer.transform(questions)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, question_vectors).flatten()
    best_index = np.argmax(scores)
    confidence = scores[best_index]

    if confidence >= threshold:
        return rows[best_index][1], confidence, rows[best_index][2]
    return None, confidence, None

# Generate a conversational response using Ollama
def generate_conversational_response(summary, query):
    prompt = f"Context: {summary}\n\nQuestion: {query}\n\nProvide a concise and accurate answer."
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    return response.get('message', {}).get('content', "I'm here to help with your questions.")

# Evaluate and select the best answer
def evaluate_answers(db_answer, db_confidence, pdf_answer, pdf_confidence):
    if pdf_confidence > db_confidence + 0.05:
        return pdf_answer, "pdf"
    elif db_answer:
        return db_answer, "database"
    return pdf_answer, "pdf"

# Process a question and validate answers
def process_question(conn, question):
    start_time = time()  # Start tracking time
    processed_query = process_text_with_spacy(question)

    db_answer, db_confidence, db_source = get_answer_from_db(conn, processed_query)
    pdf_summary, pdf_confidence = find_relevant_summary(processed_query)
    pdf_answer = generate_conversational_response(pdf_summary, question)

    if db_answer and pdf_answer:
        final_answer = f"{db_answer} Additionally, {pdf_answer}"
        source = "merged"
    elif db_answer:
        final_answer, source = db_answer, "database"
    elif pdf_answer:
        final_answer, source = pdf_answer, "pdf"
    else:
        final_answer, source = "No suitable answer found.", "none"

    save_to_db(conn, question, final_answer, source)

    # Log execution time
    elapsed_time = time() - start_time
    logging.info(f"Processed question: '{question}' | Source: {source} | Time taken: {elapsed_time:.2f}s")
    return final_answer, source

# Main function for automated validation
def main():
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT question FROM ValidatedQA")
    rows = cursor.fetchall()
    for question, in rows:
        process_question(conn, question)

if __name__ == "__main__":
    main()
