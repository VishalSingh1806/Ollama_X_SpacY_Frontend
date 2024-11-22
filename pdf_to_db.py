import sqlite3
import pdfplumber
from nltk.tokenize import sent_tokenize

# Path to the PDF file
pdf_path = "D:\EPR Data\Test_EPR_Data.pdf"

# Function to extract and tokenize PDF content
def extract_and_tokenize_pdf(pdf_path):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    sections.append({
                        "page_number": page_number,
                        "content": sentence
                    })
    return sections

# Extract content from the PDF
pdf_sections = extract_and_tokenize_pdf(pdf_path)

# Connect to SQLite database (or create it)
conn = sqlite3.connect("knowledge_base.db")
cursor = conn.cursor()

# Create the Sections table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Sections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        page_number INTEGER,
        content TEXT
    )
""")

# Insert PDF content into the Sections table
for section in pdf_sections:
    cursor.execute("INSERT INTO Sections (page_number, content) VALUES (?, ?)", 
                   (section["page_number"], section["content"]))

# Commit and close the connection
conn.commit()
conn.close()

print("PDF content has been successfully stored in the database.")


# Reconnect to the database to verify data storage
conn = sqlite3.connect("knowledge_base.db")
cursor = conn.cursor()

# Fetch and print a sample of rows from the Sections table
cursor.execute("SELECT * FROM Sections LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)  # Each row should include id, page_number, and content

conn.close()
