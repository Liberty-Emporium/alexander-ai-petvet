"""
Pet Vet AI - Multi-Tenant Flask Application with Groq AI Integration
AI-powered pet health diagnostic app
"""

import os
import json
import uuid
import base64
import requests
import hashlib
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session, g
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pet-vet-ai-secret-key-2024')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
DATA_DIR = os.environ.get('DATA_DIR', '/data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

PLAN_LIMITS = {
    'free':       {'diagnoses_per_month': 5,   'pets': 2},
    'pro':        {'diagnoses_per_month': 999,  'pets': 999},
    'enterprise': {'diagnoses_per_month': 9999, 'pets': 9999},
}

# ==================== AUTH HELPERS ====================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def plan_required(min_plan):
    """Decorator to require a minimum plan level."""
    order = ['free', 'pro', 'enterprise']
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please log in to continue.', 'error')
                return redirect(url_for('login'))
            db = get_db()
            user = db.execute('SELECT plan FROM users WHERE id = ?', (session['user_id'],)).fetchone()
            if not user or order.index(user['plan']) < order.index(min_plan):
                flash(f'This feature requires the {min_plan.title()} plan.', 'error')
                return redirect(url_for('upgrade'))
            return f(*args, **kwargs)
        return decorated
    return decorator

# ==================== DATABASE ====================
DB_FILE = os.path.join(DATA_DIR, 'petvet.db')

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DB_FILE)
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            plan TEXT NOT NULL DEFAULT 'free',
            stripe_customer_id TEXT,
            diagnoses_used_this_month INTEGER NOT NULL DEFAULT 0,
            month_reset TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS pets (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            animal_type TEXT NOT NULL,
            breed TEXT,
            age TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS diagnoses (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            pet_id TEXT,
            animal_type TEXT,
            image_file TEXT,
            symptoms TEXT,
            ai_diagnosis TEXT,
            confidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS user_api_keys (
            user_id TEXT PRIMARY KEY,
            groq_key TEXT DEFAULT '',
            qwen_key TEXT DEFAULT '',
            active_provider TEXT DEFAULT 'groq',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS admin_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    ''')
    # Create default admin account
    admin_id = str(uuid.uuid4())
    existing = db.execute("SELECT id FROM users WHERE email = 'admin'").fetchone()
    if not existing:
        db.execute(
            "INSERT INTO users (id, email, name, password_hash, plan) VALUES (?, ?, ?, ?, ?)",
            (admin_id, 'admin', 'Admin', hash_password('admin1'), 'enterprise')
        )
    db.commit()
    db.close()

init_db()

# ==================== HELPERS ====================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_user():
    if 'user_id' not in session:
        return None
    db = get_db()
    return db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()

def get_admin_setting(key, default=''):
    db = get_db()
    row = db.execute('SELECT value FROM admin_settings WHERE key = ?', (key,)).fetchone()
    return row['value'] if row else default

def set_admin_setting(key, value):
    db = get_db()
    db.execute('INSERT OR REPLACE INTO admin_settings (key, value) VALUES (?, ?)', (key, value))
    db.commit()

def get_groq_api_key(user_id=None):
    """Get API key — user's own key first, then system admin key."""
    if user_id:
        db = get_db()
        row = db.execute('SELECT groq_key FROM user_api_keys WHERE user_id = ?', (user_id,)).fetchone()
        if row and row['groq_key']:
            return row['groq_key']
    return get_admin_setting('groq_api_key', '')

def check_and_reset_monthly_quota(user):
    """Reset monthly diagnosis count if month has changed."""
    db = get_db()
    current_month = datetime.now().strftime('%Y-%m')
    if user['month_reset'] != current_month:
        db.execute(
            'UPDATE users SET diagnoses_used_this_month = 0, month_reset = ? WHERE id = ?',
            (current_month, user['id'])
        )
        db.commit()

def user_can_diagnose(user):
    """Returns True if user is within their plan's monthly limit."""
    check_and_reset_monthly_quota(user)
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user['id'],)).fetchone()
    limit = PLAN_LIMITS.get(user['plan'], PLAN_LIMITS['free'])['diagnoses_per_month']
    return user['diagnoses_used_this_month'] < limit

def increment_diagnosis_count(user_id):
    db = get_db()
    db.execute(
        'UPDATE users SET diagnoses_used_this_month = diagnoses_used_this_month + 1 WHERE id = ?',
        (user_id,)
    )
    db.commit()

# ==================== GROQ AI ANALYSIS ====================

GROQ_MODEL = "llama-3.2-90b-vision-preview"

def analyze_pet_image(image_path, animal_type="dog", user_id=None):
    groq_api_key = get_groq_api_key(user_id)
    if not groq_api_key:
        return {"error": "Groq API key not configured. Go to Settings to add your key.", "success": False}

    try:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

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

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
        data = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]}],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            try:
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
        {"id": "d10", "name": "Hot Spot (Acute)", "symptoms": ["oozing_skin", "red_inflamed", "painful", "quick_onset"], "severity": "high", "description": "Rapidly developing irritated skin area"},
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
    return render_template('index.html')

@app.route('/diagnose')
def diagnose_page():
    return render_template('diagnose.html', conditions=CONDITIONS)

@app.route('/upload', methods=['POST'])
def upload_photo():
    user = get_current_user()
    if user:
        if not user_can_diagnose(user):
            flash('Monthly diagnosis limit reached. Upgrade for more!', 'error')
            return redirect(url_for('upgrade'))

    if 'photo' not in request.files:
        flash('No photo uploaded', 'error')
        return redirect(url_for('diagnose_page'))

    file = request.files['photo']
    animal_type = request.form.get('animal_type', 'dog')
    use_ai = request.form.get('use_ai', 'true') == 'true'
    symptoms = request.form.getlist('symptoms')
    pet_id = request.form.get('pet_id')

    if file.filename == '':
        flash('No photo selected', 'error')
        return redirect(url_for('diagnose_page'))

    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        ai_diagnosis = None
        user_id = user['id'] if user else None
        if use_ai and get_groq_api_key(user_id):
            ai_diagnosis = analyze_pet_image(filepath, animal_type, user_id)

        symptom_diagnosis = analyze_symptoms(animal_type, symptoms)
        diagnosis = combine_diagnoses(ai_diagnosis, symptom_diagnosis, use_ai)

        # Save diagnosis record scoped to user
        if user:
            db = get_db()
            db.execute(
                'INSERT INTO diagnoses (id, user_id, pet_id, animal_type, image_file, symptoms, ai_diagnosis, confidence) VALUES (?,?,?,?,?,?,?,?)',
                (str(uuid.uuid4()), user['id'], pet_id, animal_type, filename,
                 json.dumps(symptoms),
                 ai_diagnosis.get('diagnosis') if ai_diagnosis else None,
                 diagnosis.get('confidence', 'unknown'))
            )
            db.commit()
            increment_diagnosis_count(user['id'])

        return render_template('result.html',
                               diagnosis=diagnosis,
                               animal_type=animal_type,
                               image=filename,
                               ai_used=use_ai and ai_diagnosis and ai_diagnosis.get('success'))

    flash('Invalid file type', 'error')
    return redirect(url_for('diagnose_page'))

def combine_diagnoses(ai_diagnosis, symptom_diagnosis, use_ai):
    if use_ai and ai_diagnosis and ai_diagnosis.get('success'):
        return {
            'ai_diagnosis': ai_diagnosis,
            'symptom_diagnosis': symptom_diagnosis,
            'message': 'AI has analyzed your image.',
            'confidence': ai_diagnosis.get('confidence', 'medium'),
            'combined': True
        }
    return {
        'ai_diagnosis': None,
        'symptom_diagnosis': symptom_diagnosis,
        'message': symptom_diagnosis.get('message', 'Based on symptoms:'),
        'confidence': symptom_diagnosis.get('confidence', 'low'),
        'combined': False
    }

def analyze_symptoms(animal_type, selected_symptoms):
    animal_conditions = CONDITIONS.get(animal_type, [])
    if not selected_symptoms:
        return {'conditions': animal_conditions[:3], 'message': 'Select symptoms for a more accurate diagnosis', 'confidence': 'low'}

    matches = []
    for condition in animal_conditions:
        matching = len(set(selected_symptoms) & set(condition.get('symptoms', [])))
        if matching > 0:
            match_percent = (matching / len(condition['symptoms'])) * 100
            matches.append({'condition': condition, 'match_count': matching, 'match_percent': round(match_percent, 1)})

    matches.sort(key=lambda x: x['match_percent'], reverse=True)
    if matches:
        top = matches[0]
        confidence = 'high' if top['match_percent'] > 50 else 'medium' if top['match_percent'] > 25 else 'low'
        return {'conditions': matches[:3], 'message': f"Based on {len(selected_symptoms)} symptoms:", 'confidence': confidence, 'top_match': top}

    return {'conditions': [], 'message': 'No matching conditions found. Please consult a vet.', 'confidence': 'none'}

# ==================== AUTH ROUTES ====================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')
        name = request.form.get('name', '')

        if not email or not password or not name:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('register'))

        db = get_db()
        if db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))

        user_id = str(uuid.uuid4())
        db.execute(
            'INSERT INTO users (id, email, name, password_hash, plan, month_reset) VALUES (?,?,?,?,?,?)',
            (user_id, email, name, hash_password(password), 'free', datetime.now().strftime('%Y-%m'))
        )
        db.commit()

        session['user_id'] = user_id
        session['user_name'] = name
        flash('Account created! Welcome to Pet Vet AI!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')

        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user and verify_password(password, user['password_hash']):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid email or password', 'error')
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

# ==================== DASHBOARD ====================

@app.route('/dashboard')
@login_required
def dashboard():
    user = get_current_user()
    check_and_reset_monthly_quota(user)
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user['id'],)).fetchone()

    limit = PLAN_LIMITS[user['plan']]['diagnoses_per_month']
    diagnoses = db.execute(
        'SELECT * FROM diagnoses WHERE user_id = ? ORDER BY created_at DESC LIMIT 10',
        (user['id'],)
    ).fetchall()
    pets = db.execute('SELECT * FROM pets WHERE user_id = ?', (user['id'],)).fetchall()

    return render_template('dashboard.html',
                           user=user,
                           user_name=user['name'],
                           subscription={'plan': user['plan'], 'diagnoses_used': user['diagnoses_used_this_month']},
                           diagnoses_limit=limit,
                           diagnoses=diagnoses,
                           pets=pets)

# ==================== PETS ====================

@app.route('/pets')
@login_required
def pets_page():
    db = get_db()
    pets = db.execute('SELECT * FROM pets WHERE user_id = ? ORDER BY created_at DESC', (session['user_id'],)).fetchall()
    return render_template('pets.html', pets=pets)

@app.route('/add-pet', methods=['GET', 'POST'])
@login_required
def add_pet_page():
    if request.method == 'POST':
        db = get_db()
        db.execute(
            'INSERT INTO pets (id, user_id, name, animal_type, breed, age, notes) VALUES (?,?,?,?,?,?,?)',
            (str(uuid.uuid4()), session['user_id'],
             request.form.get('name'), request.form.get('animal_type'),
             request.form.get('breed'), request.form.get('age'), request.form.get('notes'))
        )
        db.commit()
        flash('Pet added!', 'success')
        return redirect(url_for('pets_page'))
    return render_template('add-pet.html')

# ==================== SETTINGS ====================

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings_page():
    """Admin-only system-wide API key settings."""
    user = get_current_user()
    if user['plan'] != 'enterprise':
        flash('Admin settings require enterprise plan.', 'error')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        key = request.form.get('groq_api_key', '').strip()
        if key:
            set_admin_setting('groq_api_key', key)
            flash('System API key saved!', 'success')
        else:
            flash('Please enter a valid API key.', 'error')

    raw = get_admin_setting('groq_api_key', '')
    display_key = (raw[:8] + '...') if len(raw) > 8 else ('***' if raw else '')
    return render_template('settings.html', settings={'groq_api_key': display_key})

@app.route('/my-settings', methods=['GET', 'POST'])
@login_required
def my_settings():
    user_id = session['user_id']
    db = get_db()

    if request.method == 'POST':
        groq_key = request.form.get('groq_key', '').strip()
        qwen_key = request.form.get('qwen_key', '').strip()
        active_provider = request.form.get('active_provider', 'groq')
        db.execute(
            'INSERT OR REPLACE INTO user_api_keys (user_id, groq_key, qwen_key, active_provider, updated_at) VALUES (?,?,?,?,CURRENT_TIMESTAMP)',
            (user_id, groq_key, qwen_key, active_provider)
        )
        db.commit()
        flash('Your AI settings saved!', 'success')
        return redirect(url_for('my_settings'))

    row = db.execute('SELECT * FROM user_api_keys WHERE user_id = ?', (user_id,)).fetchone()
    user_keys = dict(row) if row else {'groq_key': '', 'qwen_key': '', 'active_provider': 'groq'}
    for k in ['groq_key', 'qwen_key']:
        if user_keys.get(k):
            user_keys[k] = user_keys[k][:8] + '...'

    return render_template('my_settings.html', user_keys=user_keys)

# ==================== UPGRADE / BILLING ====================

@app.route('/upgrade', methods=['GET', 'POST'])
@login_required
def upgrade():
    user = get_current_user()

    if request.method == 'POST':
        plan = request.form.get('plan', 'free')
        db = get_db()
        db.execute('UPDATE users SET plan = ? WHERE id = ?', (plan, user['id']))
        db.commit()
        flash(f'Upgraded to {plan.title()} plan!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('upgrade.html', current_plan=user['plan'])

# ==================== PROFILE ====================

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = get_current_user()
    db = get_db()

    if request.method == 'POST':
        if request.form.get('name'):
            db.execute('UPDATE users SET name = ? WHERE id = ?', (request.form.get('name'), user['id']))
            session['user_name'] = request.form.get('name')
        if request.form.get('new_password'):
            db.execute('UPDATE users SET password_hash = ? WHERE id = ?',
                       (hash_password(request.form.get('new_password')), user['id']))
        db.commit()
        flash('Profile updated!', 'success')
        return redirect(url_for('profile'))

    user = get_current_user()
    return render_template('profile.html', user=user)

# ==================== API ENDPOINTS ====================

@app.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    animal_type = request.form.get('animal_type', 'dog')
    use_ai = request.form.get('use_ai', 'true') == 'true'
    symptoms = request.form.getlist('symptoms')
    user_id = session.get('user_id')

    image = request.files['image']
    filename = f"{uuid.uuid4()}_{secure_filename(image.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    ai_diagnosis = None
    if use_ai and get_groq_api_key(user_id):
        ai_diagnosis = analyze_pet_image(filepath, animal_type, user_id)

    symptom_diagnosis = analyze_symptoms(animal_type, symptoms)
    diagnosis = combine_diagnoses(ai_diagnosis, symptom_diagnosis, use_ai)

    return jsonify({'success': True, 'image_id': filename, 'diagnosis': diagnosis})

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.get_json()
    db = get_db()
    # Store feedback linked to user if logged in
    user_id = session.get('user_id', 'anonymous')
    db.execute(
        'INSERT INTO diagnoses (id, user_id, ai_diagnosis, confidence) VALUES (?,?,?,?)',
        (str(uuid.uuid4()), user_id, data.get('correct_diagnosis'), 'feedback')
    )
    db.commit()
    return jsonify({'success': True, 'message': 'Feedback saved!'})

@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    configured = bool(get_admin_setting('groq_api_key'))
    return jsonify({'configured': configured, 'ai_enabled': True, 'default_animal': 'dog'})

# ==================== MISC ====================

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/vets')
def vets_page():
    return render_template('vets.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
