import spacy
import ollama
import pdfplumber
import sqlite3
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# Load spaCy model and embedding model
nlp = spacy.load("en_core_web_md")
embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Connect to SQLite database (creates if doesn't exist)
conn = sqlite3.connect("knowledge_base.db")
cursor = conn.cursor()

# Create tables for verified responses and PDF content if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS pdf_content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT,
    page_number INTEGER,
    section_text TEXT
)
''')
cursor.execute('''
CREATE TABLE IF NOT EXISTS verified_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    answer TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    source TEXT
)
''')
conn.commit()

# Function to extract and save PDF content
def extract_and_tokenize_pdf(pdf_path, document_name):
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                sections = sent_tokenize(text)
                for section in sections:
                    cursor.execute('''
                    INSERT INTO pdf_content (document_name, page_number, section_text)
                    VALUES (?, ?, ?)
                    ''', (document_name, page_number, section))
    conn.commit()

# Function to check if a query has a verified answer
def find_verified_answer(query):
    cursor.execute("SELECT answer FROM verified_answers WHERE query = ?", (query,))
    result = cursor.fetchone()
    return result[0] if result else None

# Function to generate and save a response if verified as correct
def ask_model(query):
    # Step 1: Check for a verified answer
    verified_answer = find_verified_answer(query)
    if verified_answer:
        return verified_answer

    # Step 2: Generate a response if no verified answer is found
    response, summary = find_relevant_summary(query)  # Use previous function for retrieving summary
    answer = generate_conversational_response(summary, query)

    # Step 3: Ask for feedback
    print(f"Bot: {answer}")
    feedback = input("Is this answer correct? (yes/no): ").strip().lower()
    if feedback == 'yes':
        cursor.execute('''
        INSERT INTO verified_answers (query, answer, source)
        VALUES (?, ?, ?)
        ''', (query, answer, "user_verified"))
        conn.commit()
        print("Answer saved to the knowledge base.")

    return answer

# Dummy functions to simulate response generation
def find_relevant_summary(query):
    # Retrieve summary from PDF or indexed data
    return "Sample summary from PDF", "summary"

def generate_conversational_response(summary, query):
    return f"Generated response based on {summary}"

# Main function to start the conversation
def start_conversation():
    print("Start your conversation with the model. Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting the conversation.")
            break
        response = ask_model(user_input)
        print(f"Bot: {response}")

# Start the conversational loop
start_conversation()
