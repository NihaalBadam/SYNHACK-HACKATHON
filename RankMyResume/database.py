import sqlite3
import pickle

DATABASE_FILE = 'resume_database.db'

def create_table():
    """Creates the main 'resumes' table in the database if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL UNIQUE,
        full_text TEXT,
        embedding BLOB
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database and 'resumes' table are ready.")

def add_resume_to_db(file_name, full_text, embedding):
    """Adds a single resume's data to the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Serialize the numpy embedding array into a binary format
    embedding_blob = sqlite3.Binary(pickle.dumps(embedding))
    
    try:
        cursor.execute('''
        INSERT INTO resumes (file_name, full_text, embedding)
        VALUES (?, ?, ?)
        ''', (file_name, full_text, embedding_blob))
        conn.commit()
        print(f"Successfully added: {file_name}")
    except sqlite3.IntegrityError:
        print(f"Skipped (already exists): {file_name}")
    except Exception as e:
        print(f"Error adding {file_name}: {e}")
    
    conn.close()

def get_all_resumes():
    """Fetches all resumes and their data from the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT file_name, full_text, embedding FROM resumes")
    rows = cursor.fetchall()
    conn.close()
    
    # Deserialize the embeddings back into numpy arrays
    resumes = []
    for row in rows:
        file_name, full_text, embedding_blob = row
        embedding = pickle.loads(embedding_blob)
        resumes.append({
            "file_name": file_name,
            "full_text": full_text,
            "embedding": embedding
        })
        
    return resumes