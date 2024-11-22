import spacy
import ollama
import pdfplumber
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
import logging
from time import time

# Enable logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.5
MERGE_CONFIDENCE_GAP = 0.1

# Initialize databases
def init_dbs():
    # Original knowledge base
    conn_original = sqlite3.connect("knowledge_base.db", check_same_thread=False)
    conn_original.execute("PRAGMA journal_mode=WAL;")
    conn_original.cursor().execute("""
        CREATE TABLE IF NOT EXISTS ValidatedQA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE,
            answer TEXT
        )
    """)
    conn_original.commit()
    
    # Processed knowledge base
    conn_processed = sqlite3.connect("processed_knowledge_base.db", check_same_thread=False)
    conn_processed.execute("PRAGMA journal_mode=WAL;")
    conn_processed.cursor().execute("""
        CREATE TABLE IF NOT EXISTS ProcessedQA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE,
            answer TEXT,
            source TEXT,
            confidence REAL
        )
    """)
    conn_processed.commit()
    return conn_original, conn_processed

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

# Load and preprocess PDF data
pdf_path = r"D:\EPR Data\Test_EPR_Data.pdf"
pdf_sections = extract_and_tokenize_pdf(pdf_path)

# Summarize sections for TF-IDF vectorization
def summarize_sections(sections, max_length=150):
    return [section if len(section) <= max_length else section[:max_length].rsplit(' ', 1)[0] + "..." for section in sections]

# Summarize PDF sections
pdf_summaries = summarize_sections(pdf_sections)

# Vectorize summaries for similarity search
vectorizer = TfidfVectorizer()
summary_vectors = vectorizer.fit_transform(pdf_summaries)

# Retrieve the most relevant summary based on query
def find_relevant_summary(query):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, summary_vectors).flatten()
    best_index = np.argmax(scores)
    return pdf_summaries[best_index], scores[best_index]

# Process and lemmatize query with spaCy
def process_text_with_spacy(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Save processed question and answer to the new database
def save_to_processed_db(conn_processed, question, answer, source, confidence):
    cursor = conn_processed.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO ProcessedQA (question, answer, source, confidence)
        VALUES (?, ?, ?, ?)
    """, (question, answer, source, confidence))
    conn_processed.commit()

# Check the original database for a similar question
def get_answer_from_original_db(conn_original, query, threshold=0.7):
    cursor = conn_original.cursor()
    cursor.execute("SELECT question, answer FROM ValidatedQA")
    rows = cursor.fetchall()
    questions = [row[0] for row in rows]
    
    if not questions:
        return None, 0

    question_vectors = vectorizer.transform(questions)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, question_vectors).flatten()
    best_index = np.argmax(scores)
    confidence = scores[best_index]
    
    if confidence >= threshold:
        return rows[best_index][1], confidence
    return None, confidence

# Generate a conversational response using Ollama
def generate_conversational_response(summary, query):
    prompt = f"Context: {summary}\n\nQuestion: {query}\n\nProvide a concise and accurate answer."
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    return response.get('message', {}).get('content', "I'm here to help with your questions.")

# Process question and determine the best answer
def process_question(conn_original, conn_processed, question, question_num):
    start_time = time()
    processed_query = process_text_with_spacy(question)
    db_answer, db_confidence = get_answer_from_original_db(conn_original, processed_query)
    pdf_summary, pdf_confidence = find_relevant_summary(processed_query)
    pdf_answer = generate_conversational_response(pdf_summary, question)
    
    # Determine the best answer
    if db_answer and pdf_answer:
        if abs(db_confidence - pdf_confidence) <= MERGE_CONFIDENCE_GAP:
            final_answer = (
                f"Combined response:\nDatabase: {db_answer}\nPDF: {pdf_answer}"
            )
            source, confidence = "merged", (db_confidence + pdf_confidence) / 2
        elif db_confidence > pdf_confidence:
            final_answer, source, confidence = db_answer, "database", db_confidence
        else:
            final_answer, source, confidence = pdf_answer, "pdf", pdf_confidence
    elif db_answer:
        final_answer, source, confidence = db_answer, "database", db_confidence
    else:
        final_answer, source, confidence = pdf_answer, "pdf", pdf_confidence

    # Save to the new processed database
    save_to_processed_db(conn_processed, question, final_answer, source, confidence)
    end_time = time()
    logging.info(
        f"Q{question_num}: {question} | Source: {source} | Confidence: {confidence:.2f} | Time: {end_time - start_time:.2f}s"
    )
    return final_answer, source

# Main function for automated processing
def main():
    conn_original, conn_processed = init_dbs()
    cursor = conn_original.cursor()
    cursor.execute("SELECT question FROM ValidatedQA")
    rows = cursor.fetchall()
    
    for idx, (question,) in enumerate(rows, start=1):
        process_question(conn_original, conn_processed, question, idx)

if __name__ == "__main__":
    main()
