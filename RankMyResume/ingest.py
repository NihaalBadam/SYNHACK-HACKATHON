import os
import pypdf
import database
from sentence_transformers import SentenceTransformer

RESUME_FOLDER = 'resumes' # Make sure you create this folder
# Load the "agent" model. This will download it the first time.
print("Loading semantic model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

def parse_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def ingest_all_resumes():
    """
    Goes through the 'resumes' folder, parses each PDF,
    generates its embedding, and adds it to the database.
    """
    # First, ensure the database table exists
    database.create_table()
    
    print(f"\nStarting ingestion from '{RESUME_FOLDER}' folder...")
    
    for filename in os.listdir(RESUME_FOLDER):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(RESUME_FOLDER, filename)
            
            # 1. Parse PDF to get text
            print(f"--- Processing: {filename} ---")
            resume_text = parse_pdf(file_path)
            
            if resume_text:
                # 2. Generate "agent" embedding
                print("Generating semantic embedding...")
                embedding = model.encode(resume_text)
                
                # 3. Add to database
                database.add_resume_to_db(filename, resume_text, embedding)
            else:
                print(f"Could not extract text from {filename}. Skipping.")
    
    print("\n--- Ingestion Complete ---")
    print("All resumes have been processed and stored in the database.")

if __name__ == '__main__':
    ingest_all_resumes()