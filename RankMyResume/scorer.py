from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the same model used during ingestion
print("Loading scoring model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Scoring model loaded.")

def calculate_keyword_score(resume_text, requirements_list):
    """
    Calculates a score based on how many explicit requirements are met.
    """
    if not requirements_list:
        return 0
        
    score = 0
    resume_text_lower = resume_text.lower()
    
    for req in requirements_list:
        if req.strip().lower() in resume_text_lower:
            score += 1
            
    # Normalize the score (0 to 1)
    return score / len(requirements_list)

def calculate_semantic_score(jd_embedding, resume_embedding):
    """
    Calculates the "GPT agent" score using cosine similarity.
    This measures the contextual similarity.
    """
    # Reshape for sklearn
    jd_embedding = jd_embedding.reshape(1, -1)
    resume_embedding = resume_embedding.reshape(1, -1)
    
    # Calculate similarity
    sim_score = cosine_similarity(jd_embedding, resume_embedding)[0][0]
    
    # Ensure score is 0-1 (it should be, but good practice)
    return max(0, min(1, sim_score))

def rank_resumes(job_description, requirements_str, all_resumes, kw_weight, sem_weight):
    """
    The main ranking function.
    Combines keyword and semantic scores for all resumes.
    """
    
    # 1. Prepare JD and Requirements
    jd_embedding = model.encode(job_description)
    requirements_list = [req.strip() for req in requirements_str.split(',') if req.strip()]
    
    ranked_list = []
    
    # 2. Score every resume in the database
    for resume in all_resumes:
        # Score 1: Keyword Score
        keyword_score = calculate_keyword_score(resume['full_text'], requirements_list)
        
        # Score 2: "Agent" Semantic Score
        semantic_score = calculate_semantic_score(jd_embedding, resume['embedding'])
        
        # 3. Calculate Final Weighted Score
        # Normalize weights in case they don't add up to 1
        total_weight = kw_weight + sem_weight
        if total_weight == 0: total_weight = 1 # Avoid division by zero
        
        final_score = ((keyword_score * kw_weight) + (semantic_score * sem_weight)) / total_weight
        
        ranked_list.append({
            "file_name": resume['file_name'],
            "final_score": round(final_score * 100, 2), # As a percentage
            "keyword_score": round(keyword_score * 100, 2),
            "semantic_score": round(semantic_score * 100, 2)
        })
        
    # 4. Sort by highest score first
    ranked_list.sort(key=lambda x: x['final_score'], reverse=True)
    
    return ranked_list