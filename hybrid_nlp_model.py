import spacy
import ollama
import pdfplumber
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import deque
import sqlite3

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize conversation history with a maximum length of 5 to retain context
conversation_history = deque(maxlen=5)

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
pdf_path = r"D:\EPR Data\Section wise EPR data\test data 9.pdf"
pdf_sections = extract_and_tokenize_pdf(pdf_path)

# Summarize each section
def summarize_sections(sections, max_length=150):
    summaries = []
    for section in sections:
        summary = section if len(section) <= max_length else section[:max_length].rsplit(' ', 1)[0] + "..."
        summaries.append(summary)
    return summaries

# Summarize PDF sections
pdf_summaries = summarize_sections(pdf_sections)

# Vectorize summaries for efficient similarity search
vectorizer = TfidfVectorizer()
summary_vectors = vectorizer.fit_transform(pdf_summaries)

# Retrieve the most relevant summary based on query
def find_relevant_summary(query):
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, summary_vectors).flatten()
    best_index = np.argmax(similarity_scores)
    return pdf_summaries[best_index]

# Process and lemmatize query with spaCy
def process_text_with_spacy(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Initialize SQLite database to store validated answers
def init_db():
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ValidatedQA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT
        )
    """)
    conn.commit()
    conn.close()

# Save validated question and answer to the database
def save_to_db(question, answer):
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO ValidatedQA (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

# Check the database for a similar question and return the answer if it meets the threshold
def get_answer_from_db(query, threshold=0.7):
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM ValidatedQA")
    rows = cursor.fetchall()
    conn.close()
    
    # Vectorize the query and existing questions for similarity matching
    questions = [row[0] for row in rows]
    question_vectors = vectorizer.transform(questions)
    query_vector = vectorizer.transform([query])
    
    # Calculate similarities and find the best match
    similarity_scores = cosine_similarity(query_vector, question_vectors).flatten()
    best_index = np.argmax(similarity_scores)
    if similarity_scores[best_index] >= threshold:
        return rows[best_index][1]
    return None

# Generate a conversational response using Ollama with context
def generate_conversational_response(summary, query):
    # Compile recent conversation history as context
    context = "\n".join([f"User: {q}\nBot: {a}" for q, a in conversation_history])
    prompt = f"Answer the following question based on the provided information:\n\n{context}\n\nContext: {summary}\n\nQuestion: {query}\n\nAnswer concisely."
    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    if 'message' in response and 'content' in response['message']:
        return response['message']['content']
    return "I'm here to help with your questions."

# Main function to ask questions with context and enhanced summarization
def ask_model(query):
    processed_query = process_text_with_spacy(query)
    
    # Check database for a relevant answer before generating a new one
    db_answer = get_answer_from_db(processed_query)
    if db_answer:
        return db_answer  # Return stored answer if a close match is found

    relevant_summary = find_relevant_summary(processed_query)
    answer = generate_conversational_response(relevant_summary, query)
    
    # Store the question and answer in conversation history for context
    conversation_history.append((query, answer))
    return answer

def start_conversation():
    """
    Starts a conversation loop with the model, retaining context across questions.
    Adds a feedback mechanism to validate the model's answers.
    """
    init_db()
    print("Start your conversation with the model. Type 'exit' to end.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting the conversation.")
            break
        
        # Get the model's response
        response = ask_model(user_input)
        print(f"Bot: {response}")
        
        # Ask for feedback
        feedback = input("Was this answer correct? (yes/no): ").strip().lower()
        if feedback == 'y':
            save_to_db(user_input, response)
            print("The response has been saved to the knowledge base.")
        elif feedback == 'n':
            print("The response was not saved.")

# Start the conversational loop
start_conversation()
