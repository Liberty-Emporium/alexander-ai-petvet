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
import hashlib
import bcrypt
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session, g

import time as _rl_time
from collections import defaultdict as _defaultdict
_rate_store = _defaultdict(list)
_RATE_WINDOW = 60
_RATE_MAX = 10

def _check_login_rate(ip):
    now = _rl_time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < _RATE_WINDOW]
    if len(_rate_store[ip]) >= _RATE_MAX:
        return False
    _rate_store[ip].append(now)
    return True

from werkzeug.utils import secure_filename
from functools import wraps

# ============================================================
# RATE LIMITER — No external dependencies required
# ============================================================
import time as _rl_time

def _is_rate_limited(db, key, max_calls=5, window_seconds=60):
    """Returns True if this key has exceeded the rate limit."""
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS rate_limits (
            key TEXT NOT NULL, window_start INTEGER NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (key, window_start))""")
        db.execute("DELETE FROM rate_limits WHERE window_start < ?",
                   (int(_rl_time.time()) - window_seconds * 2,))
        now = int(_rl_time.time())
        ws = now - (now % window_seconds)
        row = db.execute(
            "SELECT count FROM rate_limits WHERE key=? AND window_start=?",
            (key, ws)).fetchone()
        if row is None:
            db.execute("INSERT OR IGNORE INTO rate_limits VALUES (?,?,1)", (key, ws))
            db.commit()
            return False
        if row[0] >= max_calls:
            return True
        db.execute("UPDATE rate_limits SET count=count+1 WHERE key=? AND window_start=?",
                   (key, ws))
        db.commit()
        return False
    except Exception:
        return False


app = Flask(__name__)

# Session security hardening
app.config['SESSION_COOKIE_SECURE'] = False  # Set True when HTTPS confirmed
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.secret_key = os.environ.get('SECRET_KEY', 'pet-vet-ai-secret-key-2024')

import secrets as _secrets_module

def _get_csrf_token():
    """Generate or retrieve CSRF token from session."""
    if 'csrf_token' not in session:
        session['csrf_token'] = _secrets_module.token_hex(32)
    return session['csrf_token']

def _validate_csrf():
    """Validate CSRF token on POST requests. Returns True if valid."""
    if request.method != 'POST':
        return True
    # Skip API routes
    if request.path.startswith('/api/'):
        return True
    token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
    return token and token == session.get('csrf_token')

app.jinja_env.globals['csrf_token'] = _get_csrf_token


# Configuration
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')  # ✅ uses /data volume
DATA_DIR = os.environ.get('DATA_DIR', os.path.join('/data'))
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'settings.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ==================== AUTH HELPERS ====================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ==================== DATABASE ====================
DB_FILE = os.path.join(DATA_DIR, 'petvet.db')

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_FILE)
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA synchronous=NORMAL")
        g.db.execute("PRAGMA foreign_keys=ON")
        g.db.execute("PRAGMA busy_timeout=5000")
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DB_FILE)
    db.execute('''CREATE TABLE IF NOT EXISTS user_api_keys (
        user_id TEXT PRIMARY KEY,
        groq_key TEXT DEFAULT '',
        qwen_key TEXT DEFAULT '',
        active_provider TEXT DEFAULT 'groq',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    db.commit()
    db.close()

init_db()

# Settings management
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def get_groq_api_key(user_id=None):
    """Get API key - user's own key first, then system admin key"""
    # Try user-specific key first
    if user_id:
        db = get_db()
        c = db.cursor()
        c.execute('SELECT groq_key FROM user_api_keys WHERE user_id = ?', (user_id,))
        row = c.fetchone()
        if row and row[0]:
            return row[0]
    # Fall back to system admin key from settings
    settings = load_settings()
    return settings.get('groq_api_key', '')

# Groq API Configuration (no env var fallback)
GROQ_API_KEY = ''
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

# ── OpenRouter AI (single provider) ──────────────────────────────────────────
def get_openrouter_key(user_id=None):
    """Get OpenRouter API key from config or env."""
    import os
    return get_config('openrouter_key', os.environ.get('OPENROUTER_API_KEY', ''))

def get_openrouter_model(user_id=None):
    """Get selected OpenRouter model from config."""
    return get_config('openrouter_model', 'google/gemini-flash-1.5')

def call_openrouter(messages, user_id=None, max_tokens=1000):
    """Call OpenRouter API with any model. Returns text string."""
    import urllib.request as _ur, json as _json
    key = get_openrouter_key(user_id)
    if not key:
        return "AI unavailable — add your OpenRouter API key in Settings ⚙️"
    model = get_openrouter_model(user_id)
    try:
        payload = _json.dumps({
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens
        }).encode()
        req = _ur.Request(
            'https://openrouter.ai/api/v1/chat/completions',
            data=payload,
            headers={
                'Authorization': f'Bearer {key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://libertyemporium.com',
                'X-Title': 'Liberty App'
            }
        )
        with _ur.urlopen(req, timeout=30) as resp:
            return _json.loads(resp.read())['choices'][0]['message']['content']
    except Exception as e:
        return f"AI error: {e}"
# ─────────────────────────────────────────────────────────────────────────────

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
    # Check if user is logged in and has available diagnoses
    if 'user_id' in session:
        subs = load_subscriptions()
        user_sub = subs.get(session['user_id'], {'plan': 'free', 'diagnoses_used': 0})
        
        if user_sub.get('plan') == 'free' and user_sub.get('diagnoses_used', 0) >= 5:
            flash('Free plan limit reached (5/month). Upgrade for unlimited diagnoses!', 'error')
            return redirect(url_for('upgrade'))
    
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
        api_key = get_groq_api_key(session.get('user_id'))
        if use_ai and api_key:
            ai_diagnosis = analyze_pet_image(filepath, animal_type)
        
        # Also run symptom matching (fallback / comparison)
        symptom_diagnosis = analyze_symptoms(animal_type, symptoms)
        
        # Combine results
        diagnosis = combine_diagnoses(ai_diagnosis, symptom_diagnosis, use_ai)
        
        # Save diagnosis record
        save_diagnosis(animal_type, filename, symptoms, diagnosis, ai_diagnosis)
        
        # Increment diagnosis count for logged-in users
        if 'user_id' in session:
            increment_diagnosis_count()
        
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

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact page"""
    if request.method == 'POST':
        # Basic form handling — in production wire up to email/SMTP
        flash('Thanks for your message! We\'ll get back to you soon.', 'success')
        return redirect(url_for('contact'))
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
    api_key = get_groq_api_key(session.get('user_id'))
    if use_ai and api_key:
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
    """Admin system-wide settings for API keys"""
    settings = load_settings()
    
    if request.method == 'POST':
        new_settings = {
            'groq_api_key': request.form.get('groq_api_key', ''),
            'default_animal': request.form.get('default_animal', 'dog')
        }
        if new_settings['groq_api_key']:
            settings.update(new_settings)
            save_settings(settings)
            flash('Settings saved successfully!', 'success')
        else:
            flash('Please enter a valid API key', 'error')
        settings = load_settings()
    
    display_settings = settings.copy()
    if display_settings.get('groq_api_key'):
        key = display_settings['groq_api_key']
        display_settings['groq_api_key'] = key[:8] + '...' if len(key) > 8 else '***'
    
    return render_template('settings.html', settings=display_settings,
        key_set=bool(get_openrouter_key()),
        current_key=get_openrouter_key(),
        current_model=get_openrouter_model())

@app.route('/my-settings', methods=['GET', 'POST'])
@login_required
def my_settings():
    """Per-user AI API key settings"""
    user_id = session['user_id']
    db = get_db()
    
    if request.method == 'POST':
        groq_key = request.form.get('groq_key', '').strip()
        qwen_key = request.form.get('qwen_key', '').strip()
        active_provider = request.form.get('active_provider', 'groq')
        db.execute('''INSERT OR REPLACE INTO user_api_keys 
            (user_id, groq_key, qwen_key, active_provider, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)''',
            (user_id, groq_key, qwen_key, active_provider))
        db.commit()
        flash('Your AI settings saved!', 'success')
        return redirect(url_for('my_settings'))
    
    row = db.execute('SELECT * FROM user_api_keys WHERE user_id = ?', (user_id,)).fetchone()
    user_keys = dict(row) if row else {'groq_key': '', 'qwen_key': '', 'active_provider': 'groq'}
    
    # Mask keys for display
    for k in ['groq_key', 'qwen_key']:
        if user_keys.get(k):
            user_keys[k] = user_keys[k][:8] + '...'
    
    return render_template('my_settings.html', user_keys=user_keys)

@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    """API to check if settings are configured"""
    settings = load_settings()
    return jsonify({
        'configured': bool(settings.get('groq_api_key')),
        'ai_enabled': settings.get('ai_enabled', True),
        'default_animal': settings.get('default_animal', 'dog')
    })

# ==================== USER ACCOUNTS & SUBSCRIPTIONS ====================

USERS_FILE = os.path.join(DATA_DIR, 'users.json')
SUBSCRIPTIONS_FILE = os.path.join(DATA_DIR, 'subscriptions.json')
PETS_FILE = os.path.join(DATA_DIR, 'pets.json')

def load_pets():
    if os.path.exists(PETS_FILE):
        with open(PETS_FILE) as f:
            return json.load(f)
    return {}

def save_pets(pets):
    with open(PETS_FILE, 'w') as f:
        json.dump(pets, f, indent=2)

def get_user_pets(user_id):
    pets = load_pets()
    return pets.get(user_id, [])

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_subscriptions():
    if os.path.exists(SUBSCRIPTIONS_FILE):
        with open(SUBSCRIPTIONS_FILE) as f:
            return json.load(f)
    return {}

def save_subscriptions(subs):
    with open(SUBSCRIPTIONS_FILE, 'w') as f:
        json.dump(subs, f, indent=2)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def _is_sha256(h):
    return len(h) == 64 and all(c in '0123456789abcdef' for c in h)

def verify_password(password, hashed):
    if _is_sha256(hashed):
        return hashlib.sha256(password.encode()).hexdigest() == hashed
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')
        name = request.form.get('name', '')
        
        if not email or not password or not name:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('register'))
        
        users = load_users()
        if email in users:
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        users[email] = {
            'name': name,
            'password': hash_password(password),
            'created': datetime.now().isoformat()
        }
        save_users(users)
        
        # Create free subscription
        subs = load_subscriptions()
        subs[email] = {
            'plan': 'free',
            'diagnoses_used': 0,
            'created': datetime.now().isoformat()
        }
        save_subscriptions(subs)
        
        session['user_id'] = email
        session['user_name'] = name
        flash('Account created! Welcome to Pet Vet AI!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Rate limiting — 10 login attempts per minute per IP
    _ip = request.remote_addr or 'unknown'
    if _is_rate_limited(get_db(), f'login:{_ip}', max_calls=10, window_seconds=60):
        return jsonify({'error': 'Too many login attempts. Please wait 1 minute.'}), 429

    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')
        
        users = load_users()
        if email in users and verify_password(password, users[email]['password']):
            session['user_id'] = email
            session['user_name'] = users[email]['name']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password', 'error')
        return render_template('login.html'), 200
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

# ==================== SMTP ====================

def send_email(to, subject, body):
    import smtplib
    from email.mime.text import MIMEText
    cfg = {
        'host':     os.environ.get('SMTP_HOST', ''),
        'port':     int(os.environ.get('SMTP_PORT', 587)),
        'user':     os.environ.get('SMTP_USER', ''),
        'password': os.environ.get('SMTP_PASSWORD', ''),
        'from':     os.environ.get('SMTP_FROM', os.environ.get('SMTP_USER', '')),
    }
    if not cfg['host'] or not cfg['user'] or not cfg['password']:
        print(f'[EMAIL] SMTP not configured, skipping email to {to}', flush=True)
        return False
    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = cfg['from']
        msg['To'] = to
        with smtplib.SMTP(cfg['host'], cfg['port'], timeout=15) as s:
            s.ehlo(); s.starttls()
            s.login(cfg['user'], cfg['password'])
            s.sendmail(cfg['from'], [to], msg.as_string())
        return True
    except Exception as e:
        print(f'[EMAIL] Failed: {e}', flush=True)
        return False

# ==================== FORGOT PASSWORD ====================

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    sent = False
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user:
            token = hashlib.sha256(os.urandom(32)).hexdigest()
            expires = (datetime.now() + __import__('datetime').timedelta(hours=1)).isoformat()
            db.execute('''CREATE TABLE IF NOT EXISTS password_resets (
                token TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL)''')
            db.execute('DELETE FROM password_resets WHERE user_id = ?', (user['id'],))
            db.execute('INSERT INTO password_resets (token, user_id, expires_at) VALUES (?,?,?)',
                       (token, user['id'], expires))
            db.commit()
            reset_url = request.host_url.rstrip('/') + f'/reset-password/{token}'
            send_email(email, 'Pet Vet AI — Reset Your Password',
                f'Hi {user["name"]},\n\nReset your password here (valid 1 hour):\n{reset_url}\n\n— Pet Vet AI')
        sent = True
    return render_template('forgot_password.html', sent=sent)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    db = get_db()
    try:
        db.execute('CREATE TABLE IF NOT EXISTS password_resets (token TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL)')
        record = db.execute('SELECT * FROM password_resets WHERE token = ?', (token,)).fetchone()
    except Exception:
        record = None
    if not record:
        flash('Invalid or expired reset link.', 'error')
        return redirect(url_for('forgot_password'))
    from datetime import datetime as dt
    if dt.utcnow() > dt.fromisoformat(record['expires_at']):
        db.execute('DELETE FROM password_resets WHERE token = ?', (token,))
        db.commit()
        flash('Reset link expired. Please request a new one.', 'error')
        return redirect(url_for('forgot_password'))
    error = None
    if request.method == 'POST':
        new_pass = request.form.get('new_password', '')
        confirm = request.form.get('confirm_password', '')
        if len(new_pass) < 6:
            error = 'Password must be at least 6 characters.'
        elif new_pass != confirm:
            error = 'Passwords do not match.'
        else:
            db.execute('UPDATE users SET password_hash = ? WHERE id = ?',
                       (hashlib.sha256(new_pass.encode()).hexdigest(), record['user_id']))
            db.execute('DELETE FROM password_resets WHERE token = ?', (token,))
            db.commit()
            flash('Password reset! You can now log in.', 'success')
            return redirect(url_for('login'))
    return render_template('reset_password.html', token=token, error=error)

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    subs = load_subscriptions()
    user_sub = subs.get(user_id, {'plan': 'free', 'diagnoses_used': 0})
    
    # Get diagnosis history
    diagnoses_file = os.path.join(DATA_DIR, 'diagnoses.json')
    diagnoses = []
    if os.path.exists(diagnoses_file):
        with open(diagnoses_file) as f:
            all_diagnoses = json.load(f)
            # Filter for this user (we'll need to track user in diagnoses)
            diagnoses = all_diagnoses[-10:]  # Last 10
    
    return render_template('dashboard.html', 
                         user_name=session['user_name'],
                         subscription=user_sub)

@app.route('/upgrade', methods=['GET', 'POST'])
@login_required
def upgrade():
    user_id = session['user_id']
    subs = load_subscriptions()
    user_sub = subs.get(user_id, {'plan': 'free'})
    
    if request.method == 'POST':
        plan = request.form.get('plan', 'free')
        
        # In production, this would integrate with Stripe
        # For now, we simulate the upgrade
        subs[user_id] = {
            'plan': plan,
            'diagnoses_used': user_sub.get('diagnoses_used', 0),
            'upgraded': datetime.now().isoformat()
        }
        save_subscriptions(subs)
        
        flash(f'Upgraded to {plan.title()} plan!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('upgrade.html', current_plan=user_sub.get('plan', 'free'))

def check_diagnosis_limit():
    """Check if user has reached their diagnosis limit"""
    if 'user_id' not in session:
        return True  # Not logged in, allow limited use
    
    user_id = session['user_id']
    subs = load_subscriptions()
    user_sub = subs.get(user_id, {'plan': 'free', 'diagnoses_used': 0})
    
    if user_sub.get('plan') == 'free':
        return user_sub.get('diagnoses_used', 0) < 5
    return True  # Premium unlimited

def increment_diagnosis_count():
    """Increment user's diagnosis count"""
    if 'user_id' not in session:
        return
    
    user_id = session['user_id']
    subs = load_subscriptions()
    if user_id in subs:
        current = subs[user_id].get('diagnoses_used', 0)
        subs[user_id]['diagnoses_used'] = current + 1
        save_subscriptions(subs)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_id = session['user_id']
    users = load_users()
    user = users.get(user_id, {})
    
    if request.method == 'POST':
        # Update name
        if request.form.get('name'):
            users[user_id]['name'] = request.form.get('name')
            session['user_name'] = request.form.get('name')
        
        # Update password
        if request.form.get('new_password'):
            users[user_id]['password'] = hash_password(request.form.get('new_password'))
        
        save_users(users)
        flash('Profile updated!', 'success')
    
    return render_template('profile.html', user=user)

# Override upload to check limits
original_upload = None


@app.route('/health')
def health_check():
    """Health check endpoint for Railway and monitoring."""
    try:
        db = get_db()
        db.execute("SELECT 1").fetchone()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
    status = "ok" if db_status == "ok" else "degraded"
    return __import__('json').dumps({"status": status, "db": db_status}), \
           200 if status == "ok" else 503, \
           {"Content-Type": "application/json"}


# ============================================================

# ============================================================
# STRUCTURED LOGGING + METRICS
# ============================================================
import logging as _log, time as _t

_log_handler = _log.StreamHandler()
_log_handler.setFormatter(_log.Formatter('%(asctime)s %(levelname)s %(message)s'))
app.logger.addHandler(_log_handler)
app.logger.setLevel(_log.INFO)

def _ensure_metrics():
    try:
        db = get_db()
        db.execute("""CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric TEXT NOT NULL, value REAL DEFAULT 1,
            tenant_slug TEXT,
            created_at TEXT DEFAULT (datetime('now')))""")
        db.commit()
    except Exception:
        pass

def track(metric, value=1, slug=None):
    try:
        _ensure_metrics()
        get_db().execute(
            "INSERT INTO metrics (metric,value,tenant_slug) VALUES (?,?,?)",
            (metric, value, slug))
        get_db().commit()
    except Exception:
        pass

@app.before_request
def _start_timer():
    from flask import g
    g._start = _t.time()


@app.after_request
def _add_security_headers(response):
    """Security headers on every response."""
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    if 'Content-Security-Policy' not in response.headers:
        response.headers['Content-Security-Policy'] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' https: data: blob:;"
    return response

@app.after_request
def _log_req(response):
    from flask import g
    if not request.path.startswith('/static'):
        ms = (_t.time() - getattr(g, '_start', _t.time())) * 1000
        if ms > 800:
            app.logger.warning(f"SLOW {request.method} {request.path} {response.status_code} {ms:.0f}ms")
    return response



# ============================================================
# SEO — Sitemap + Robots.txt
# ============================================================
@app.route('/sitemap.xml')
def sitemap():
    """Auto-generated XML sitemap for SEO."""
    host = request.host_url.rstrip('/')
    urls = [
        {'loc': f"{host}/",          'priority': '1.0', 'changefreq': 'weekly'},
        {'loc': f"{host}/login",     'priority': '0.8', 'changefreq': 'monthly'},
        {'loc': f"{host}/signup",    'priority': '0.9', 'changefreq': 'monthly'},
        {'loc': f"{host}/pricing",   'priority': '0.8', 'changefreq': 'monthly'},
    ]
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for u in urls:
        xml.append(f"  <url>")
        xml.append(f"    <loc>{u['loc']}</loc>")
        xml.append(f"    <changefreq>{u['changefreq']}</changefreq>")
        xml.append(f"    <priority>{u['priority']}</priority>")
        xml.append(f"  </url>")
    xml.append('</urlset>')
    return '\n'.join(xml), 200, {'Content-Type': 'application/xml'}

@app.route('/robots.txt')
def robots():
    """robots.txt for search engine crawling guidance."""
    host = request.host_url.rstrip('/')
    content = f"""User-agent: *
Allow: /
Disallow: /admin
Disallow: /overseer
Disallow: /api/
Sitemap: {host}/sitemap.xml
"""
    return content, 200, {'Content-Type': 'text/plain'}


# GLOBAL ERROR HANDLERS
# ============================================================
@app.errorhandler(404)
def not_found_error(e):
    if request.path.startswith('/api/'):
        return __import__('flask').jsonify({'error': 'Not found'}), 404
    return render_template('404.html') if os.path.exists(
        os.path.join(app.template_folder or 'templates', '404.html')
    ) else ('<h1>404 - Page Not Found</h1>', 404)

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"UNHANDLED_500: {str(e)}", exc_info=True)
    if request.path.startswith('/api/'):
        return __import__('flask').jsonify({'error': 'Internal server error'}), 500
    return '<h1>500 - Something went wrong. We are looking into it.</h1>', 500

@app.errorhandler(429)
def rate_limit_error(e):
    return __import__('flask').jsonify({'error': 'Too many requests. Please slow down.'}), 429


# ── Admin-only API token UI routes ───────────────────────────────────────────
@app.route('/api/token/ui', methods=['POST'])
def api_token_ui_generate():
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin only'}), 403
    import secrets as _s, hashlib as _h, datetime as _dt
    user_id = session.get('user_id') or session.get('super_admin_id') or 1
    label = 'ui-generated'
    raw_token = _s.token_urlsafe(48)
    token_hash = _h.sha256(raw_token.encode()).hexdigest()
    expires_at = (_dt.datetime.utcnow() + _dt.timedelta(days=365)).isoformat()
    conn = get_db()
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS api_tokens (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, token_hash TEXT UNIQUE, label TEXT, expires_at TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
        conn.execute('DELETE FROM api_tokens WHERE user_id=? AND label=?', (user_id, label))
        conn.execute('INSERT INTO api_tokens (user_id,token_hash,label,expires_at) VALUES (?,?,?,?)', (user_id, token_hash, label, expires_at))
        conn.commit()
    finally:
        conn.close()
    return jsonify({'success':True,'api_token':raw_token,'expires_at':expires_at})

@app.route('/api/token/ui', methods=['DELETE'])
def api_token_ui_revoke():
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin only'}), 403
    user_id = session.get('user_id') or session.get('super_admin_id') or 1
    conn = get_db()
    try:
        conn.execute('DELETE FROM api_tokens WHERE user_id=? AND label=?', (user_id, 'ui-generated'))
        conn.commit()
    finally:
        conn.close()
    return jsonify({'success':True})


# ── API Key Infrastructure ────────────────────────────────────────────────────
import secrets as _api_secrets, hashlib as _api_hash, functools as _api_functools

_API_KEYS_FILE = os.path.join(os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/data'), 'api_keys.json')

def _load_api_keys():
    if os.path.exists(_API_KEYS_FILE):
        with open(_API_KEYS_FILE) as f:
            import json as _j; return _j.load(f)
    return {}

def _save_api_keys(keys):
    with open(_API_KEYS_FILE, 'w') as f:
        import json as _j; _j.dump(keys, f, indent=2)

def _require_api_key(f):
    """Decorator: require valid API key via X-API-Key header, Authorization: Bearer, or ?api_key= param."""
    @_api_functools.wraps(f)
    def decorated(*args, **kwargs):
        key = (request.headers.get('X-API-Key') or
               request.args.get('api_key') or
               (request.headers.get('Authorization','')[7:].strip() if request.headers.get('Authorization','').startswith('Bearer ') else None))
        if not key:
            return jsonify({'error': 'API key required. Pass as X-API-Key header or Authorization: Bearer <key>'}), 401
        keys = _load_api_keys()
        if key not in keys:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/admin/api-generator')
@login_required
def _admin_api_generator_page():
    if session.get('username') != os.environ.get('ADMIN_USER','admin') and session.get('role') != 'overseer':
        return jsonify({'error': 'Admin only'}), 403
    keys = _load_api_keys()
    new_key = request.args.get('new_key', '')
    base_url = request.host_url.rstrip('/')
    return render_template('admin_api_generator.html',
        api_keys=keys, new_key=new_key, base_url=base_url,
        endpoints=[('GET', '/api/users/count', 'Total registered users'), ('GET', '/api/diagnoses/count', 'Total diagnoses run'), ('GET', '/api/stats', 'App stats'), ('GET', '/health', 'Health check (no auth)')])

@app.route('/admin/api-generator/generate', methods=['POST'])
@login_required
def _admin_api_generate():
    if session.get('username') != os.environ.get('ADMIN_USER','admin') and session.get('role') != 'overseer':
        return jsonify({'error': 'Admin only'}), 403
    from datetime import datetime as _dt
    label = request.form.get('label','Testing Key').strip() or 'Testing Key'
    raw_key = 'pv_' + _api_secrets.token_urlsafe(32)
    keys = _load_api_keys()
    keys[raw_key] = {'name': label, 'created_by': 'admin', 'created_at': _dt.utcnow().isoformat(), 'active': True}
    _save_api_keys(keys)
    flash(f'API key generated!', 'success')
    return redirect('/admin/api-generator?new_key=' + raw_key)

@app.route('/admin/api-generator/revoke/<path:key>', methods=['POST'])
@login_required
def _admin_api_revoke(key):
    if session.get('username') != os.environ.get('ADMIN_USER','admin') and session.get('role') != 'overseer':
        return jsonify({'error': 'Admin only'}), 403
    keys = _load_api_keys()
    if key in keys:
        del keys[key]
        _save_api_keys(keys)
        flash('Key revoked.', 'success')
    return redirect('/admin/api-generator')

# ── Public API ───────────────────────────────────────────────────────────────
@app.route('/api/users/count', methods=['GET'])
@_require_api_key
def _api_pv_user_count():
    try:
        users = load_users() if callable(globals().get('load_users')) else {}
        return jsonify({'total_users': len(users)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/diagnoses/count', methods=['GET'])
@_require_api_key
def _api_pv_diagnoses():
    try:
        db = get_db()
        count = db.execute('SELECT COUNT(*) FROM metrics WHERE metric="diagnosis"').fetchone()
        return jsonify({'total_diagnoses': count[0] if count else 0})
    except Exception as e:
        return jsonify({'total_diagnoses': 0})

@app.route('/api/stats', methods=['GET'])
@_require_api_key
def _api_pv_stats():
    try:
        users = load_users() if callable(globals().get('load_users')) else {}
        return jsonify({'total_users': len(users), 'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
@app.route('/vets')
def vets_page():
    """Vet network page - find nearby vets"""
    return render_template('vets.html')

@app.route('/add-pet')
def add_pet_page():
    return render_template('add-pet.html')

# ── Pets API ─────────────────────────────────────────────
@app.route('/api/pets', methods=['GET'])
@login_required
def api_get_pets():
    """Return all pets for the logged-in user."""
    return jsonify(get_user_pets(session['user_id']))

@app.route('/api/pets', methods=['POST'])
@login_required
def api_save_pet():
    """Add or update a pet."""
    data = request.get_json()
    if not data or not data.get('name'):
        return jsonify({'error': 'Name is required'}), 400
    user_id = session['user_id']
    pets = load_pets()
    user_pets = pets.get(user_id, [])
    # Edit existing
    pet_id = data.get('id')
    if pet_id:
        for i, p in enumerate(user_pets):
            if p['id'] == pet_id:
                data['updated_at'] = datetime.utcnow().isoformat()
                user_pets[i] = {**p, **data}
                pets[user_id] = user_pets
                save_pets(pets)
                return jsonify({'ok': True, 'pet': user_pets[i]})
        return jsonify({'error': 'Pet not found'}), 404
    # Add new
    new_pet = {
        'id': str(uuid.uuid4()),
        'name':    data.get('name', '').strip(),
        'species': data.get('species', 'dog'),
        'breed':   data.get('breed', '').strip(),
        'age':     data.get('age', ''),
        'weight':  data.get('weight', ''),
        'notes':   data.get('notes', '').strip(),
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
    }
    user_pets.append(new_pet)
    pets[user_id] = user_pets
    save_pets(pets)
    return jsonify({'ok': True, 'pet': new_pet}), 201

@app.route('/api/pets/<pet_id>', methods=['DELETE'])
@login_required
def api_delete_pet(pet_id):
    """Delete a pet by ID."""
    user_id = session['user_id']
    pets = load_pets()
    user_pets = pets.get(user_id, [])
    original_len = len(user_pets)
    user_pets = [p for p in user_pets if p['id'] != pet_id]
    if len(user_pets) == original_len:
        return jsonify({'error': 'Pet not found'}), 404
    pets[user_id] = user_pets
    save_pets(pets)
    return jsonify({'ok': True})

# ==================== ADMIN OVERSEER ====================

ADMIN_USER = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS_HASH = hashlib.sha256(os.environ.get('ADMIN_PASSWORD', 'admin1').encode()).hexdigest()

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('is_admin'):
            flash('Admin access required.', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username == ADMIN_USER and hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASS_HASH:
            session['is_admin'] = True
            session['admin_user'] = username
            return redirect(url_for('overseer'))
        flash('Invalid admin credentials.', 'error')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    session.pop('admin_user', None)
    return redirect(url_for('index'))

@app.route('/overseer')
@admin_required
def overseer():
    users = load_users()
    subs  = load_subscriptions()
    user_list = []
    for email, u in users.items():
        sub = subs.get(email, {})
        user_list.append({
            'email':      email,
            'name':       u.get('name', ''),
            'plan':       sub.get('plan', 'free'),
            'diagnoses':  sub.get('diagnoses_used', 0),
            'created':    u.get('created', '')[:10] if u.get('created') else '',
        })
    user_list.sort(key=lambda x: x['created'], reverse=True)
    paid_count = sum(1 for u in user_list if u['plan'] != 'free')
    mrr = paid_count * 9.99
    return render_template('overseer.html',
        users=user_list, total=len(user_list),
        paid=paid_count, free=len(user_list)-paid_count, mrr=mrr)

@app.route('/overseer/user/<email>/upgrade', methods=['POST'])
@admin_required
def overseer_upgrade(email):
    subs = load_subscriptions()
    if email not in subs: subs[email] = {}
    subs[email]['plan'] = 'premium'
    subs[email]['diagnoses_limit'] = 999999
    save_subscriptions(subs)
    flash(f'{email} upgraded to Premium.', 'success')
    return redirect(url_for('overseer'))

@app.route('/overseer/user/<email>/downgrade', methods=['POST'])
@admin_required
def overseer_downgrade(email):
    subs = load_subscriptions()
    if email not in subs: subs[email] = {}
    subs[email]['plan'] = 'free'
    subs[email]['diagnoses_limit'] = 3
    save_subscriptions(subs)
    flash(f'{email} downgraded to Free.', 'success')
    return redirect(url_for('overseer'))

@app.route('/overseer/user/<path:email>/delete', methods=['POST'])
@admin_required
def overseer_delete_user(email):
    users = load_users()
    subs  = load_subscriptions()
    users.pop(email, None)
    subs.pop(email, None)
    save_users(users)
    save_subscriptions(subs)
    flash(f'{email} deleted.', 'success')
    return redirect(url_for('overseer'))
