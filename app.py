"""
Pet Vet AI - MVP Flask Application with Groq AI Integration
AI-powered pet health diagnostic app
"""

import os
import json
import csv
import uuid
import base64
import requests
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pet-vet-ai-secret-key-2024')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'settings.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Settings management
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def get_groq_api_key():
    """Get API key from settings or environment variable"""
    settings = load_settings()
    if settings.get('groq_api_key'):
        return settings['groq_api_key']
    return os.environ.get('GROQ_API_KEY', '')

# Groq API Configuration
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_MODEL = "llama-3.2-90b-vision-preview"  # Groq's vision model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== GROQ AI ANALYSIS ====================

def analyze_pet_image(image_path, animal_type="dog"):
    """
    Use Groq's vision model to analyze a pet health image
    Returns AI's diagnostic assessment
    """
    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        return {"error": "Groq API key not configured. Go to Settings to add your key.", "success": False}
    
    try:
        # Convert image to base64
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Prepare the prompt for the AI
        animal_names = {
            "dog": "dog", "cat": "cat", "horse": "horse", "goat": "goat",
            "sheep": "sheep", "cow": "cow", "reptile": "reptile/snake/lizard",
            "small_pet": "small pet (hamster/gerbil/rabbit/guinea pig)"
        }
        animal = animal_names.get(animal_type, "pet")
        
        prompt = f"""You are a veterinary expert AI. Analyze this photo of a {animal} and identify any visible health conditions or symptoms.

Look for:
- Skin conditions (rashes, lesions, hot spots, wounds, parasites)
- Eye health (discharge, redness, cloudiness)
- Ear issues (discharge, swelling, redness)
- Mouth/dental problems
- Paw/limb injuries
- Swelling or abnormalities
- Coat/fur condition
- Signs of pain or discomfort

Respond in JSON format:
{{
    "diagnosis": "brief diagnosis or 'No clear issues visible'",
    "confidence": "high/medium/low",
    "severity": "critical/urgent/monitor/none",
    "symptoms_observed": ["list of visible symptoms"],
    "recommendation": "brief recommendation (home care, vet visit, emergency)",
    "description": "detailed explanation of what you see"
}}

If the image is unclear or not of a pet, return {{"error": "Image unclear or not a pet"}}"""

        # Call Groq API
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse JSON from response
            try:
                # Try to extract JSON from the response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                diagnosis = json.loads(content)
                diagnosis["success"] = True
                return diagnosis
            except json.JSONDecodeError:
                return {"diagnosis": content[:200], "success": True, "confidence": "low"}
        else:
            return {"error": f"API error: {response.status_code}", "success": False}
            
    except Exception as e:
        return {"error": str(e), "success": False}

# ==================== CONDITION DATABASE ====================

CONDITIONS = {
    "dog": [
        {"id": "d1", "name": "Hot Spot", "symptoms": ["red_patch", "hair_loss", "moist_skin", "itching"], "severity": "moderate", "description": "Localized area of inflamed, irritated skin"},
        {"id": "d2", "name": "Ear Infection", "symptoms": ["head_shaking", "odor", "discharge", "scratching_ears"], "severity": "moderate", "description": "Bacterial or yeast infection in the ear canal"},
        {"id": "d3", "name": "Conjunctivitis", "symptoms": ["red_eyes", "discharge", "squinting", "pawing_at_face"], "severity": "moderate", "description": "Inflammation of the eye membrane"},
        {"id": "d4", "name": "Ringworm", "symptoms": ["circular_lesions", "hair_loss", "scaling", "broken_hair"], "severity": "moderate", "description": "Fungal infection causing circular patches"},
        {"id": "d5", "name": "Flea Allergy Dermatitis", "symptoms": ["intense_itching", "hair_loss", "red_bumps", "flea_dirt"], "severity": "moderate", "description": "Allergic reaction to flea bites"},
        {"id": "d6", "name": "Lipoma", "symptoms": ["soft_lump", "movable_under_skin", "no_pain"], "severity": "low", "description": "Benign fatty tumor"},
        {"id": "d7", "name": "Dental Disease", "symptoms": ["bad_breath", "red_gums", "drooling", "difficulty_eating"], "severity": "high", "description": "Gum disease and tooth decay"},
        {"id": "d8", "name": "Paw Injury", "symptoms": ["limping", "swollen_paw", "cuts_on_paw", "licking_paw"], "severity": "moderate", "description": "Injury to the paw pad or between toes"},
        {"id": "d9", "name": "Skin Tag", "symptoms": ["skin_growth", "hanging_tag", "soft", "same_color"], "severity": "low", "description": "Benign skin growth"},
        {"id": "d10", "name": "Hot Spot (Acute)", "symptoms": ["oozing_skin", "red_ inflamed", "painful", "quick_onset"], "severity": "high", "description": "Rapidly developing irritated skin area"},
    ],
    "cat": [
        {"id": "c1", "name": "Ear Mites", "symptoms": ["dark_discharge", "scratching_ears", "head_shaking", "odor"], "severity": "moderate", "description": "Parasitic ear infection"},
        {"id": "c2", "name": "Ringworm", "symptoms": ["circular_lesions", "hair_loss", "scaling", "broken_hair"], "severity": "moderate", "description": "Fungal infection causing circular patches"},
        {"id": "c3", "name": "Feline Acne", "symptoms": ["blackheads", "chin_bumps", "redness", "swelling"], "severity": "low", "description": "Oil gland clogging on chin"},
        {"id": "c4", "name": "Conjunctivitis", "symptoms": ["red_eyes", "discharge", "squinting", "blinking"], "severity": "moderate", "description": "Inflammation of the eye membrane"},
        {"id": "c5", "name": "Matted Fur", "symptoms": ["tangled_fur", "skin_irritation", "odor", "neglect"], "severity": "moderate", "description": "Tangled coat causing skin issues"},
        {"id": "c6", "name": "Wound/Abscess", "symptoms": ["swelling", "pus", "pain", "fever"], "severity": "high", "description": "Infected wound or bite abscess"},
        {"id": "c7", "name": "Dental Resorption", "symptoms": ["drooling", "mouth_pain", "weight_loss", "bad_breath"], "severity": "high", "description": "Tooth dissolving disease"},
        {"id": "c8", "name": "Fleas", "symptoms": ["itching", "flea_dirt", "hair_loss", "restlessness"], "severity": "moderate", "description": "External parasites"},
    ],
    "horse": [
        {"id": "h1", "name": "Rain Rot", "symptoms": ["crusty_scabs", "hair_loss", "dandruff", "back_lesions"], "severity": "moderate", "description": "Bacterial skin infection from moisture"},
        {"id": "h2", "name": "Scratches (Mud Fever)", "symptoms": ["swollen_legs", "crusty_skin", "pain", "lameness"], "severity": "moderate", "description": "Bacterial infection on lower legs"},
        {"id": "h3", "name": "Sweet Itch", "symptoms": ["intense_itching", "mane_tail_rubbing", "hair_loss", "sores"], "severity": "moderate", "description": "Allergic reaction to insect bites"},
        {"id": "h4", "name": "Hoof Abscess", "symptoms": ["acute_lameness", "heat_in_hoof", "pulse", "reluctance_to_stand"], "severity": "high", "description": "Pus pocket in the hoof"},
        {"id": "h5", "name": "Rain Scald", "symptoms": ["matted_hair", "skin_lesions", "drainage", "back_legs"], "severity": "moderate", "description": "Fungal infection from wet conditions"},
        {"id": "h6", "name": "Eye Injury", "symptoms": ["cloudy_eye", "tearing", "squinting", "swelling"], "severity": "high", "description": "Trauma to the eye"},
        {"id": "h7", "name": "Cuts and Lacerations", "symptoms": ["open_wound", "bleeding", "swelling", "pain"], "severity": "varies", "description": "External injuries"},
        {"id": "h8", "name": "Mud Fever", "symptoms": ["swollen_legs", "crusty_pastern", "cracked_skin", "lameness"], "severity": "moderate", "description": "Infection from wet muddy conditions"},
    ],
    "goat": [
        {"id": "g1", "name": "Foot Rot", "symptoms": ["lameness", "foul_odor", "hoof_decay", "swollen_legs"], "severity": "high", "description": "Bacterial hoof infection"},
        {"id": "g2", "name": "Mastitis", "symptoms": ["swollen_udder", "abnormal_milk", "fever", "loss_of_appetite"], "severity": "high", "description": "Udder infection"},
        {"id": "g3", "name": "Coughing", "symptoms": ["persistent_cough", "nasal_discharge", "labored_breathing", "weight_loss"], "severity": "moderate", "description": "Respiratory issue - needs vet attention"},
        {"id": "g4", "name": "Sore Mouth", "symptoms": ["blisters", "scabs_on_mouth", "drooling", "refusing_food"], "severity": "moderate", "description": "Viral infection causing mouth sores"},
    ],
    "sheep": [
        {"id": "s1", "name": "Foot Rot", "symptoms": ["lameness", "foul_odor", "hoof_decay", "swollen_legs"], "severity": "high", "description": "Bacterial hoof infection"},
        {"id": "s2", "name": "Wool Rot", "symptoms": ["matted_wool", "skin_discoloration", "odor", "itching"], "severity": "moderate", "description": "Fungal skin infection"},
        {"id": "s3", "name": "Pinkeye", "symptoms": ["red_eyes", "cloudy_cornea", "tearing", "squinting"], "severity": "moderate", "description": "Bacterial eye infection"},
    ],
    "cow": [
        {"id": "b1", "name": "Mastitis", "symptoms": ["swollen_udder", "abnormal_milk", "fever", "loss_of_appetite"], "severity": "high", "description": "Udder infection - common dairy issue"},
        {"id": "b2", "name": "Lameness", "symptoms": ["limping", "hoof_pain", "swollen_legs", "reluctance_to_move"], "severity": "high", "description": "Various hoof issues"},
        {"id": "b3", "name": "Bloat", "symptoms": ["distended_abdomen", "restlessness", "no_rumination", "labored_breathing"], "severity": "critical", "description": "Emergency - gas buildup in stomach"},
        {"id": "b4", "name": "Ringworm", "symptoms": ["circular_lesions", "hair_loss", "scaling", "crusting"], "severity": "low", "description": "Fungal skin infection"},
    ],
    "reptile": [
        {"id": "r1", "name": "Scale Rot", "symptoms": ["discolored_scales", "ulcers", "swelling", "loss_of_appetite"], "severity": "high", "description": "Bacterial infection from poor conditions"},
        {"id": "r2", "name": "Mites", "symptoms": ["visible_parasites", "restlessness", "soaking", "black_specks"], "severity": "moderate", "description": "External parasites"},
        {"id": "r3", "name": "Stomatitis (Mouth Rot)", "symptoms": ["swollen_jaws", "white_patches", "drooling", "refusing_food"], "severity": "high", "description": "Mouth infection"},
        {"id": "r4", "name": "Shedding Problems", "symptoms": ["retained_skin", "constricted_toes", "incomplete_shed", "irritation"], "severity": "moderate", "description": "Improper humidity"},
        {"id": "r5", "name": "Respiratory Infection", "symptoms": ["mouth_open", "mucus", "wheezing", "lethargy"], "severity": "high", "description": "Needs vet attention"},
    ],
    "small_pet": [
        {"id": "sp1", "name": "Wet Tail", "symptoms": ["matted_tail", "diarrhea", "lethargy", "hunched_posture"], "severity": "critical", "description": "Emergency in hamsters - needs immediate vet"},
        {"id": "sp2", "name": "Fur Mites", "symptoms": ["hair_loss", "scratching", "dandruff", "skin_irritation"], "severity": "moderate", "description": "Parasitic infestation"},
        {"id": "sp3", "name": "Bumblefoot", "symptoms": ["swollen_feet", "redness", "sores", "limping"], "severity": "moderate", "description": "Pressure sore on feet"},
        {"id": "sp4", "name": "Dental Problems", "symptoms": ["drooling", "weight_loss", "not_eating", "long_teeth"], "severity": "high", "description": "Overgrown teeth - common in rabbits/guinea pigs"},
    ]
}

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/diagnose')
def diagnose_page():
    """Main diagnosis page"""
    return render_template('diagnose.html', conditions=CONDITIONS)

@app.route('/upload', methods=['POST'])
def upload_photo():
    """Handle photo upload and analysis"""
    if 'photo' not in request.files:
        flash('No photo uploaded', 'error')
        return redirect(url_for('diagnose_page'))
    
    file = request.files['photo']
    animal_type = request.form.get('animal_type', 'dog')
    use_ai = request.form.get('use_ai', 'true') == 'true'
    symptoms = request.form.getlist('symptoms')
    
    if file.filename == '':
        flash('No photo selected', 'error')
        return redirect(url_for('diagnose_page'))
    
    if file and allowed_file(file.filename):
        # Save the file
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get AI analysis if enabled
        ai_diagnosis = None
        if use_ai and GROQ_API_KEY:
            ai_diagnosis = analyze_pet_image(filepath, animal_type)
        
        # Also run symptom matching (fallback / comparison)
        symptom_diagnosis = analyze_symptoms(animal_type, symptoms)
        
        # Combine results
        diagnosis = combine_diagnoses(ai_diagnosis, symptom_diagnosis, use_ai)
        
        # Save diagnosis record
        save_diagnosis(animal_type, filename, symptoms, diagnosis, ai_diagnosis)
        
        return render_template('result.html', 
                             diagnosis=diagnosis,
                             animal_type=animal_type,
                             image=filename,
                             ai_used=use_ai and ai_diagnosis and ai_diagnosis.get('success'))
    
    flash('Invalid file type', 'error')
    return redirect(url_for('diagnose_page'))

def combine_diagnoses(ai_diagnosis, symptom_diagnosis, use_ai):
    """Combine AI analysis with symptom matching"""
    if use_ai and ai_diagnosis and ai_diagnosis.get('success'):
        return {
            'ai_diagnosis': ai_diagnosis,
            'symptom_diagnosis': symptom_diagnosis,
            'message': 'AI has analyzed your image. View results below.',
            'confidence': ai_diagnosis.get('confidence', 'medium'),
            'combined': True
        }
    else:
        return {
            'ai_diagnosis': None,
            'symptom_diagnosis': symptom_diagnosis,
            'message': symptom_diagnosis.get('message', 'Based on symptoms:'),
            'confidence': symptom_diagnosis.get('confidence', 'low'),
            'combined': False
        }

def analyze_symptoms(animal_type, selected_symptoms):
    """Simple symptom matching algorithm"""
    animal_conditions = CONDITIONS.get(animal_type, [])
    
    if not selected_symptoms:
        return {
            'conditions': animal_conditions[:3],
            'message': 'Please select symptoms for more accurate diagnosis',
            'confidence': 'low'
        }
    
    matches = []
    for condition in animal_conditions:
        condition_symptoms = condition.get('symptoms', [])
        matching = len(set(selected_symptoms) & set(condition_symptoms))
        if matching > 0:
            match_percent = (matching / len(condition_symptoms)) * 100
            matches.append({
                'condition': condition,
                'match_count': matching,
                'match_percent': round(match_percent, 1)
            })
    
    matches.sort(key=lambda x: x['match_percent'], reverse=True)
    
    if matches:
        top = matches[0]
        confidence = 'high' if top['match_percent'] > 50 else 'medium' if top['match_percent'] > 25 else 'low'
        return {
            'conditions': matches[:3],
            'message': f"Based on {len(selected_symptoms)} symptoms:",
            'confidence': confidence,
            'top_match': top
        }
    
    return {
        'conditions': [],
        'message': 'No matching conditions found. Please consult a vet.',
        'confidence': 'none'
    }

def save_diagnosis(animal_type, image_file, symptoms, diagnosis, ai_diagnosis=None):
    """Save diagnosis for learning"""
    record = {
        'timestamp': datetime.now().isoformat(),
        'animal_type': animal_type,
        'image': image_file,
        'symptoms': symptoms,
        'ai_diagnosis': ai_diagnosis.get('diagnosis') if ai_diagnosis else None,
        'confidence': diagnosis.get('confidence', 'unknown')
    }
    
    file_path = os.path.join(DATA_DIR, 'diagnoses.json')
    if os.path.exists(file_path):
        with open(file_path) as f:
            data = json.load(f)
    else:
        data = []
    
    data.append(record)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

# API endpoint for mobile app
@app.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    """API endpoint for diagnosis"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    animal_type = request.form.get('animal_type', 'dog')
    use_ai = request.form.get('use_ai', 'true') == 'true'
    symptoms = request.form.getlist('symptoms')
    
    image = request.files['image']
    filename = f"{uuid.uuid4()}_{secure_filename(image.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    
    # Get AI analysis
    ai_diagnosis = None
    if use_ai and GROQ_API_KEY:
        ai_diagnosis = analyze_pet_image(filepath, animal_type)
    
    # Get symptom diagnosis
    symptom_diagnosis = analyze_symptoms(animal_type, symptoms)
    
    # Combine
    diagnosis = combine_diagnoses(ai_diagnosis, symptom_diagnosis, use_ai)
    
    return jsonify({
        'success': True,
        'image_id': filename,
        'diagnosis': diagnosis
    })

# Feedback endpoint for learning
@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """Receive user feedback to improve AI"""
    data = request.get_json()
    feedback_file = os.path.join(DATA_DIR, 'feedback.json')
    
    if os.path.exists(feedback_file):
        with open(feedback_file) as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []
    
    feedback_data.append({
        'timestamp': datetime.now().isoformat(),
        'diagnosis_id': data.get('diagnosis_id'),
        'correct': data.get('correct'),
        'correct_diagnosis': data.get('correct_diagnosis'),
        'notes': data.get('notes')
    })
    
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    return jsonify({'success': True, 'message': 'Feedback saved - AI is learning!'})

@app.route('/pets')
def pets_page():
    """Pet profiles page"""
    return render_template('pets.html')

@app.route('/settings', methods=['GET', 'POST'])
def settings_page():
    """Settings page for API keys and preferences"""
    settings = load_settings()
    
    if request.method == 'POST':
        # Update settings
        new_settings = {
            'groq_api_key': request.form.get('groq_api_key', ''),
            'default_animal': request.form.get('default_animal', 'dog')
        }
        
        # Mask the API key for display (show only first 8 chars)
        if new_settings['groq_api_key']:
            settings.update(new_settings)
            save_settings(settings)
            flash('Settings saved successfully!', 'success')
        else:
            flash('Please enter a valid API key', 'error')
        
        settings = load_settings()
    
    # Mask API key for display
    display_settings = settings.copy()
    if display_settings.get('groq_api_key'):
        key = display_settings['groq_api_key']
        display_settings['groq_api_key'] = key[:8] + '...' if len(key) > 8 else '***'
    
    return render_template('settings.html', settings=display_settings)

@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    """API to check if settings are configured"""
    settings = load_settings()
    return jsonify({
        'configured': bool(settings.get('groq_api_key')),
        'ai_enabled': settings.get('ai_enabled', True),
        'default_animal': settings.get('default_animal', 'dog')
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)