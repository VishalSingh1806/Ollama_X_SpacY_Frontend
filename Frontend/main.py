import yaml
import sqlite3
import spacy
import numpy as np
import pdfplumber
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import ollama

DB_PATH = r"D:\Ollama X SpacY\knowledge_base.db"

# Load SpaCy model
nlp = spacy.load("en_core_web_md")

# Step 1: Parse YAML files
def parse_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def extract_nlu_data(nlu_file_path):
    nlu_data = parse_yaml(nlu_file_path)
    if isinstance(nlu_data, list):
        questions = []
        for item in nlu_data:
            intent = item.get("intent", "")
            examples = item.get("examples", "").split("\n")
            questions.extend([ex.strip("- ").strip() for ex in examples if ex.strip()])
        return questions
    questions = []
    for intent in nlu_data.get("nlu", []):
        examples = intent.get("examples", "").split("\n")
        questions.extend([ex.strip("- ").strip() for ex in examples if ex.strip()])
    return questions

def extract_domain_responses(domain_file_path):
    domain_data = parse_yaml(domain_file_path)
    responses = {}
    for intent_name, response_data in domain_data.get("responses", {}).items():
        if isinstance(response_data, list) and "text" in response_data[0]:
            responses[intent_name] = response_data[0]["text"]
    return responses

# Step 2: Process PDF
def extract_pdf_content(pdf_path):
    content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                content.extend(sent_tokenize(text))
    return content

def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

# Step 3: Query from all sources
def query_pdf(query, pdf_texts, vectorizer, vectors):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, vectors).flatten()
    best_idx = np.argmax(scores)
    return pdf_texts[best_idx], scores[best_idx]

def query_database(query, conn):
    cursor = conn.cursor()
    user_embedding = np.array(nlp(query).vector).reshape(1, -1)
    cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
    rows = cursor.fetchall()
    best_answer, max_similarity = None, 0.0
    for db_question, db_answer, db_embedding in rows:
        db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)
        similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]
        if similarity > max_similarity:
            max_similarity, best_answer = similarity, db_answer
    return best_answer, max_similarity

def refine_with_ollama(context, query):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer concisely and realistically."
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    return response.get("message", {}).get("content", "No response generated.")

def handle_query(question, domain_responses, pdf_texts, vectorizer, vectors, conn):
    # Check for greeting intent
    greeting_intents = ["hello", "hi", "hey", "good morning", "good afternoon", "howdy"]
    farewell_intents = ["bye", "goodbye", "thanks", "see you", "cheers", "thank you"]

    lower_question = question.lower()

    if any(greet in lower_question for greet in greeting_intents):
        return "Hello! How can I assist you today?", "greeting", 1.0

    if any(farewell in lower_question for farewell in farewell_intents):
        return "Goodbye! Have a great day!", "farewell", 1.0

    # Existing logic for querying domain, database, and PDF
    domain_answer = domain_responses.get(question, "No domain response found.")
    db_answer, db_confidence = query_database(question, conn)
    pdf_answer, pdf_confidence = query_pdf(question, pdf_texts, vectorizer, vectors)
    
    context = f"Domain: {domain_answer}\nDatabase: {db_answer}\nPDF: {pdf_answer}"
    final_answer = refine_with_ollama(context, question)

    return final_answer, "combined", max(db_confidence, pdf_confidence, 0.5)

# Step 4: Save to database
def process_question(question, pdf_texts, vectorizer, vectors, domain_responses, conn, question_number):
    start_time = time.time()

    # Handle the query
    final_answer, source, confidence = handle_query(question, domain_responses, pdf_texts, vectorizer, vectors, conn)

    # Save to database
    cursor = conn.cursor()
    embedding = np.array(nlp(question).vector, dtype=np.float32).tobytes()
    cursor.execute("""
        INSERT OR REPLACE INTO ValidatedQA (question, answer, confidence, source, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (question, final_answer, confidence, source, embedding))
    conn.commit()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Log the result
    print(f"Q{question_number}: {question}")
    print(f"Selected Answer: {final_answer}")
    print(f"Source: {source}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Time Taken: {elapsed_time:.2f}s\n")

# Step 5: Main workflow
def main():
    nlu_file = r"D:\ChatBot\data\nlu.yml"
    domain_file = r"D:\ChatBot\domain.yml"
    pdf_file = r"D:\EPR Data\Test_EPR_Data.pdf"

    # Extract data
    questions = extract_nlu_data(nlu_file)
    domain_responses = extract_domain_responses(domain_file)
    pdf_texts = extract_pdf_content(pdf_file)
    vectorizer, vectors = vectorize_texts(pdf_texts)

    # Process each question
    conn = sqlite3.connect(DB_PATH)
    for question_number, question in enumerate(questions, start=1):
        process_question(question, pdf_texts, vectorizer, vectors, domain_responses, conn, question_number)
    conn.close()
    print("All questions processed and saved.")

if __name__ == "__main__":
    main()
