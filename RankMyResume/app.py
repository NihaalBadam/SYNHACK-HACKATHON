from flask import Flask, render_template, request, send_from_directory
import database
import scorer
import os

app = Flask(__name__)

# Cache the resumes in memory so we don't query the DB on every request
print("Loading all resumes from database into memory...")
ALL_RESUMES = database.get_all_resumes()
print(f"Successfully loaded {len(ALL_RESUMES)} resumes.")

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    
    if request.method == 'POST':
        # 1. Get all inputs from the recruiter
        job_description = request.form.get('job_description', '')
        requirements_str = request.form.get('requirements', '')
        
        # Get the weights from the sliders
        kw_weight = request.form.get('keyword_weight', '50')
        sem_weight = request.form.get('semantic_weight', '50')
        
        # 2. Call the "brain" to rank the resumes
        results = scorer.rank_resumes(
            job_description=job_description,
            requirements_str=requirements_str,
            all_resumes=ALL_RESUMES,
            kw_weight=int(kw_weight) / 100,  # Convert from 0-100 to 0-1
            sem_weight=int(sem_weight) / 100
        )
        
        # Pass the original inputs back to the template
        return render_template(
            'index.html',
            results=results,
            job_description=job_description,
            requirements=requirements_str,
            kw_weight=kw_weight,
            sem_weight=sem_weight
        )
    
    # For a GET request, just show the page
    return render_template(
        'index.html',
        results=[],
        kw_weight=100, 
        sem_weight=30   
    )
# Define the path to your resumes folder
RESUME_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resumes')

@app.route('/resumes/<path:filename>')
def get_resume(filename):
    """Safely serves a file from the 'resumes' directory."""
    return send_from_directory(RESUME_FOLDER, filename)

if __name__ == '__main__':
    # 'debug=True' makes the server auto-reload when you save changes
    app.run(port=5001, debug=True)