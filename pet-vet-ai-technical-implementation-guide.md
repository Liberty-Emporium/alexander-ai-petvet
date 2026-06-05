# 🐾 Pet Vet AI — Technical Implementation Guide
### Prepared by Mingo for Django | 2026-06-04
### For: Pet Vet AI Modernization (Jay approved 2026-06-05)

---

## 📋 Overview

This document provides technical specifications for implementing the 5 new features in the Pet Vet AI Django app. Each feature includes: data models, API endpoints, UI components, and integration notes.

**Existing codebase:** Django app on Django's machine
**Design system:** Midnight Executive (navy/ice blue/cyan accent)
**Existing features:** AI photo analysis (Groq vision), multi-language support (8 langs), free tier (5 diagnoses/mo)

---

## Feature 1: AI Vet Chat Assistant 🤖

### Description
Conversational symptom checker that asks follow-up questions like a real vet nurse. Provides triage-level guidance.

### Data Models

```python
# models.py additions

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    pet = models.ForeignKey('Pet', on_delete=models.SET_NULL, null=True, blank=True)
    title = models.CharField(max_length=200, default="New Consultation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_archived = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-updated_at']

class ChatMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=[('user', 'User'), ('assistant', 'Assistant')])
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Triage assessment (filled by AI)
    urgency_level = models.CharField(
        max_length=20,
        choices=[
            ('emergency', 'Emergency — See vet now'),
            ('urgent', 'Urgent — Within 24 hours'),
            ('monitor', 'Monitor — Watch for changes'),
            ('home_care', 'Home care — Manage at home'),
            ('info', 'Informational — General guidance'),
        ],
        null=True, blank=True
    )
    recommended_action = models.TextField(blank=True)
    
    class Meta:
        ordering = ['created_at']

class SymptomTag(models.Model):
    """Tags extracted from chat for analytics and health timeline"""
    name = models.CharField(max_length=100, unique=True)
    
class ChatSymptom(models.Model):
    chat = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='symptoms')
    symptom = models.ForeignKey(SymptomTag, on_delete=models.CASCADE)
    confidence = models.FloatField(default=1.0)  # AI confidence
    mentioned_at = models.DateTimeField(auto_now_add=True)
```

### API Endpoints

```
POST   /api/chat/sessions/              — Create new chat session
GET    /api/chat/sessions/              — List user's sessions
GET    /api/chat/sessions/{id}/         — Get session with messages
POST   /api/chat/sessions/{id}/message/ — Send message, get AI response
DELETE /api/chat/sessions/{id}/         — Archive session
GET    /api/chat/sessions/{id}/summary/ — Get triage summary
```

### AI Prompt Structure

```
SYSTEM_PROMPT = """You are a veterinary triage assistant. Your role:
1. Ask clarifying questions about the pet's symptoms (one at a time)
2. Assess urgency level: emergency/urgent/monitor/home_care/info
3. Provide home care advice when appropriate
4. ALWAYS recommend seeing a vet for serious symptoms
5. NEVER diagnose — only provide triage guidance
6. Be empathetic and clear
7. Support these languages: EN, ES, FR, PT, DE, ZH, JA, AR

Species you can advise on: dogs, cats, horses, goats, cattle, reptiles, birds, rabbits

Response format:
- Follow-up question OR assessment
- Urgency level (one of the 5 levels)
- Recommended action (1-2 sentences)
- Home care tips (if applicable)
- Disclaimer: "This is not a substitute for professional veterinary care"

Current conversation context:
- Pet species: {species}
- Pet age: {age}
- Pet weight: {weight}
- Previous symptoms mentioned: {symptoms}
"""
```

### UI Components

1. **Chat Interface** (`/chat/`)
   - Left sidebar: session list (newest first, searchable)
   - Main area: message bubble interface (user right, AI left)
   - Input: text field + send button + voice input (stretch goal)
   - Urgency badge on each AI response (color-coded: red/orange/yellow/green/blue)
   - "See a vet now" CTA button when urgency is emergency/urgent

2. **Session Summary Card**
   - After 3+ AI messages, show summary card:
   - Symptoms mentioned (tags)
   - Urgency assessment
   - Recommended action
   - "Save to Health Timeline" button

### Free Tier Limits
- Free users: 5 chat sessions/month
- Pet Parent ($12/mo): 20 sessions/month
- Premium ($29/mo): unlimited sessions
- Track via `ChatSession` count per user per month

---

## Feature 2: Telemedicine / Video Consultation Booking 📹

### Description
Pet owners can book and attend video consultations with licensed veterinarians.

### Data Models

```python
class Veterinarian(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    license_number = models.CharField(max_length=100)
    specialties = models.JSONField(default=list)  # ["general", "dermatology", "surgery"]
    bio = models.TextField(blank=True)
    photo = models.ImageField(upload_to='vet_photos/', blank=True)
    is_available = models.BooleanField(default=True)
    rating = models.FloatField(default=0.0)
    total_consultations = models.IntegerField(default=0)
    
    # Availability (simplified — could be expanded)
    available_days = models.JSONField(default=list)  # [0,1,2,3,4] = Mon-Fri
    available_hours_start = models.TimeField(default="09:00")
    available_hours_end = models.TimeField(default="17:00")
    timezone = models.CharField(max_length=50, default="America/New_York")

class ConsultationSlot(models.Model):
    vet = models.ForeignKey(Veterinarian, on_delete=models.CASCADE, related_name='slots')
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    is_booked = models.BooleanField(default=False)
    
    class Meta:
        indexes = [
            models.Index(fields=['vet', 'start_time', 'is_booked']),
        ]

class VideoConsultation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='consultations')
    pet = models.ForeignKey('Pet', on_delete=models.SET_NULL, null=True)
    vet = models.ForeignKey(Veterinarian, on_delete=models.CASCADE, related_name='consultations')
    slot = models.OneToOneField(ConsultationSlot, on_delete=models.CASCADE)
    
    status = models.CharField(
        max_length=20,
        choices=[
            ('scheduled', 'Scheduled'),
            ('in_progress', 'In Progress'),
            ('completed', 'Completed'),
            ('cancelled', 'Cancelled'),
            ('no_show', 'No Show'),
        ],
        default='scheduled'
    )
    
    # Pre-consultation
    symptoms_description = models.TextField(blank=True)
    photos = models.JSONField(default=list)  # List of uploaded photo URLs
    urgency = models.CharField(max_length=20, blank=True)
    
    # During/after consultation
    video_room_id = models.CharField(max_length=200, blank=True)  # Daily.co or Twilio room
    notes = models.TextField(blank=True)  # Vet's notes
    prescription = models.ForeignKey('Prescription', on_delete=models.SET_NULL, null=True, blank=True)
    follow_up_recommended = models.BooleanField(default=False)
    follow_up_notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Auto-save to health timeline
    saved_to_timeline = models.BooleanField(default=False)
```

### API Endpoints

```
GET    /api/vets/                       — List available vets
GET    /api/vets/{id}/                  — Vet profile
GET    /api/vets/{id}/slots/            — Available slots for a vet
POST   /api/consultations/             — Book a consultation
GET    /api/consultations/             — User's consultations
GET    /api/consultations/{id}/        — Consultation details
POST   /api/consultations/{id}/cancel/ — Cancel consultation
POST   /api/consultations/{id}/join/   — Get video room token
POST   /api/consultations/{id}/complete/ — Mark complete (vet only)
POST   /api/consultations/{id}/notes/  — Save consultation notes (vet only)
```

### Video Integration Options

**Recommended: Daily.co** (simplest, generous free tier)
- REST API for room creation
- Built-in React component
- HIPAA compliance available on paid plan
- Alternative: Twilio Video (more complex, HIPAA eligible)

### UI Components

1. **Vet Discovery Page** (`/telemedicine/vets/`)
   - Vet cards: photo, name, specialties, rating, next available slot
   - Filter by specialty, availability, rating
   - "Book Now" button on each card

2. **Booking Flow** (`/telemedicine/book/<vet_id>/`)
   - Step 1: Select date (calendar view, only available dates highlighted)
   - Step 2: Select time slot (30-min intervals)
   - Step 3: Select pet (from user's pets)
   - Step 4: Describe symptoms (text area + photo upload)
   - Step 5: Confirm booking (summary + payment if applicable)

3. **Video Call Page** (`/telemedicine/consultation/<id>/`)
   - Video window (main area)
   - Sidebar: pet info, symptoms, chat (text backup)
   - Controls: mute, camera off, share photos, end call
   - Post-call: rating + notes saved to health timeline

4. **Upcoming Consultations** (`/telemedicine/upcoming/`)
   - List of scheduled consultations
   - Countdown to next appointment
   - "Join Call" button (active 5 min before start)

---

## Feature 3: Pet Health Timeline & Medical History 📋

### Description
Digital timeline of every vet visit, vaccination, procedure, and health event for each pet.

### Data Models

```python
class Pet(models.Model):
    """Extend existing Pet model if it exists"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='pets')
    name = models.CharField(max_length=100)
    species = models.CharField(max_length=50)  # dog, cat, horse, goat, cattle, reptile, etc.
    breed = models.CharField(max_length=100, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)  # kg
    weight_unit = models.CharField(max_length=10, default='kg')
    microchip_id = models.CharField(max_length=50, blank=True)
    photo = models.ImageField(upload_to='pet_photos/', blank=True)
    allergies = models.JSONField(default=list)
    medications = models.JSONField(default=list)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class HealthEvent(models.Model):
    """Core timeline event model"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pet = models.ForeignKey(Pet, on_delete=models.CASCADE, related_name='health_events')
    
    event_type = models.CharField(
        max_length=30,
        choices=[
            ('vet_visit', 'Vet Visit'),
            ('vaccination', 'Vaccination'),
            ('surgery', 'Surgery'),
            ('medication', 'Medication'),
            ('lab_result', 'Lab Result'),
            ('consultation', 'Video Consultation'),
            ('ai_diagnosis', 'AI Diagnosis'),
            ('wearable_alert', 'Wearable Alert'),
            ('grooming', 'Grooming'),
            ('dental', 'Dental Procedure'),
            ('injury', 'Injury'),
            ('illness', 'Illness'),
            ('weight_check', 'Weight Check'),
            ('other', 'Other'),
        ]
    )
    
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    date = models.DateField()
    time = models.TimeField(null=True, blank=True)
    
    # Source
    source = models.CharField(
        max_length=20,
        choices=[
            ('manual', 'Manual Entry'),
            ('consultation', 'Video Consultation'),
            ('ai_chat', 'AI Chat'),
            ('wearable', 'Wearable Device'),
            ('vet_portal', 'Vet Portal'),
        ],
        default='manual'
    )
    source_id = models.CharField(max_length=200, blank=True)  # Link to consultation, chat, etc.
    
    # Details
    vet_name = models.CharField(max_length=200, blank=True)
    clinic_name = models.CharField(max_length=200, blank=True)
    cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Documents
    documents = models.JSONField(default=list)  # [{name, url, type}]
    
    # For vaccinations
    vaccine_name = models.CharField(max_length=200, blank=True)
    next_due_date = models.DateField(null=True, blank=True)
    
    # For medications
    medication_name = models.CharField(max_length=200, blank=True)
    dosage = models.CharField(max_length=100, blank=True)
    frequency = models.CharField(max_length=100, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-date', '-time']
        indexes = [
            models.Index(fields=['pet', '-date']),
            models.Index(fields=['pet', 'event_type']),
        ]

class MedicationReminder(models.Model):
    pet = models.ForeignKey(Pet, on_delete=models.CASCADE, related_name='medication_reminders')
    health_event = models.ForeignKey(HealthEvent, on_delete=models.CASCADE)
    reminder_time = models.TimeField()
    days_of_week = models.JSONField(default=list)  # [0,1,2,3,4,5,6]
    is_active = models.BooleanField(default=True)
    last_sent = models.DateTimeField(null=True, blank=True)
```

### API Endpoints

```
POST   /api/pets/                      — Add a pet
GET    /api/pets/                      — List user's pets
GET    /api/pets/{id}/                 — Pet profile
PUT    /api/pets/{id}/                 — Update pet
DELETE /api/pets/{id}/                 — Remove pet

GET    /api/pets/{id}/timeline/        — Get health timeline (paginated)
POST   /api/pets/{id}/timeline/        — Add health event
GET    /api/pets/{id}/timeline/{event_id}/ — Event details
PUT    /api/pets/{id}/timeline/{event_id}/ — Update event
DELETE /api/pets/{id}/timeline/{event_id}/ — Delete event

POST   /api/pets/{id}/timeline/upload/ — Upload document (PDF, image)
GET    /api/pets/{id}/timeline/export/  — Export timeline as PDF

GET    /api/pets/{id}/reminders/       — Medication reminders
POST   /api/pets/{id}/reminders/       — Create reminder
PUT    /api/pets/{id}/reminders/{id}/  — Update reminder
DELETE /api/pets/{id}/reminders/{id}/  — Delete reminder

GET    /api/pets/{id}/share/           — Generate shareable link (for vets/boarders)
```

### UI Components

1. **Pet Profile Page** (`/pets/<id>/`)
   - Pet photo, name, species, breed, age, weight
   - Quick stats: total events, upcoming reminders, last vet visit
   - Action buttons: Add Event, Upload Document, Share Records

2. **Health Timeline** (`/pets/<id>/timeline/`)
   - Vertical timeline view (newest at top)
   - Each event: icon (by type), title, date, description, documents
   - Filter by event type (chips: All, Vet Visits, Vaccinations, Medications, etc.)
   - Search by keyword
   - "Add Event" FAB (floating action button)
   - Color-coded event types

3. **Add Event Modal**
   - Event type selector (dropdown)
   - Dynamic form fields based on event type
   - Date picker, time picker
   - Document upload (drag & drop)
   - "Save" button

4. **Share Records**
   - Generate shareable link (expires in 7 days)
   - QR code for easy sharing
   - View-only access for recipient

---

## Feature 4: AI-Powered Prescription/Dosage Calculator 💊

### Description
Weight-based medication dosing with drug interaction checking.

### Data Models

```python
class MedicationDatabase(models.Model):
    """Reference database of common veterinary medications"""
    name = models.CharField(max_length=200)
    generic_name = models.CharField(max_length=200, blank=True)
    drug_class = models.CharField(max_length=100, blank=True)
    
    # Dosing
    species = models.JSONField(default=list)  # ["dog", "cat", "horse"]
    dose_per_kg = models.FloatField()  # mg/kg
    dose_unit = models.CharField(max_length=20, default='mg')
    frequency = models.CharField(max_length=100)  # "Twice daily", "Once daily"
    max_daily_dose = models.FloatField(null=True, blank=True)  # mg/kg/day
    max_single_dose = models.FloatField(null=True, blank=True)  # mg
    
    # Safety
    contraindications = models.JSONField(default=list)  # ["pregnant", "kidney_disease"]
    side_effects = models.JSONField(default=list)
    drug_interactions = models.JSONField(default=list)  # [medication_db_id, ...]
    warnings = models.TextField(blank=True)
    
    # Species-specific notes
    species_notes = models.JSONField(default=dict)  # {"dog": "...", "cat": "..."}
    
    is_active = models.BooleanField(default=True)

class DosageCalculation(models.Model):
    """User's dosage calculations (saved history)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    pet = models.ForeignKey(Pet, on_delete=models.CASCADE)
    medication = models.ForeignKey(MedicationDatabase, on_delete=models.CASCADE)
    
    pet_weight = models.FloatField()  # kg at time of calculation
    calculated_dose = models.FloatField()  # mg
    volume_to_administer = models.FloatField(null=True, blank=True)  # ml (if liquid)
    concentration = models.CharField(max_length=50, blank=True)  # e.g., "50mg/5ml"
    
    # Interaction check results
    current_medications = models.JSONField(default=list)
    interactions_found = models.JSONField(default=list)  # [{med, severity, description}]
    
    calculated_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)
```

### API Endpoints

```
GET    /api/medications/               — Search medications (by name, species)
GET    /api/medications/{id}/          — Medication details
POST   /api/medications/calculate/     — Calculate dosage
GET    /api/medications/history/       — User's calculation history
POST   /api/medications/interactions/ — Check drug interactions
```

### Calculation Logic

```python
def calculate_dose(medication, pet_weight_kg, concentration=None):
    """
    Calculate medication dose based on weight.
    
    Args:
        medication: MedicationDatabase instance
        pet_weight_kg: float, pet weight in kg
        concentration: str, e.g., "50mg/5ml" for liquid meds
    
    Returns:
        dict with dose_mg, volume_ml (if liquid), warnings
    """
    dose_mg = medication.dose_per_kg * pet_weight_kg
    
    # Check max single dose
    warnings = []
    if medication.max_single_dose:
        max_mg = medication.max_single_dose * pet_weight_kg
        if dose_mg > max_mg:
            dose_mg = max_mg
            warnings.append(f"Dose capped at maximum single dose ({max_mg}mg)")
    
    result = {
        'dose_mg': round(dose_mg, 2),
        'frequency': medication.frequency,
        'warnings': warnings,
    }
    
    # Calculate volume for liquid medications
    if concentration:
        # Parse "50mg/5ml" → 10mg/ml
        match = re.match(r'(\d+)mg/(\d+)ml', concentration)
        if match:
            mg_per_ml = int(match.group(1)) / int(match.group(2))
            result['volume_ml'] = round(dose_mg / mg_per_ml, 2)
            result['concentration'] = concentration
    
    return result

def check_interactions(medication, current_medication_ids):
    """Check for drug interactions"""
    interactions = []
    for med_id in current_medication_ids:
        if med_id in medication.drug_interactions:
            other_med = MedicationDatabase.objects.get(id=med_id)
            interactions.append({
                'medication': other_med.name,
                'severity': 'high',  # Could be expanded
                'description': f"Potential interaction between {medication.name} and {other_med.name}. Consult your veterinarian.",
            })
    return interactions
```

### UI Components

1. **Dosage Calculator** (`/tools/dosage-calculator/`)
   - Step 1: Select pet (auto-fills weight)
   - Step 2: Search medication (autocomplete from database)
   - Step 3: Enter concentration (if liquid)
   - Step 4: Review current medications (for interaction check)
   - Result card: dose in mg, volume in ml, frequency, warnings
   - "Save to Pet's Records" button

2. **Medication Search**
   - Search by name or generic name
   - Filter by species
   - Results: name, drug class, typical use

3. **Calculation History**
   - Table of past calculations
   - Filter by pet, medication
   - "Recalculate" button (reopens with same params)

---

## Feature 5: Wearable Device Integration 📡

### Description
Sync data from pet health wearables (FitBark, Whistle, etc.) for proactive health monitoring.

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Wearable    │────▶│  API Gateway │────▶│  Pet Vet AI     │
│  Device      │     │  (webhooks)  │     │  Django App     │
└─────────────┘     └──────────────┘     └─────────────────┘
                                              │
                                              ▼
                                        ┌──────────────┐
                                        │  Health      │
                                        │  Timeline    │
                                        └──────────────┘
```

### Data Models

```python
class WearableDevice(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pet = models.ForeignKey(Pet, on_delete=models.CASCADE, related_name='wearables')
    
    device_type = models.CharField(
        max_length=50,
        choices=[
            ('fitbark', 'FitBark'),
            ('whistle', 'Whistle'),
            ('fi', 'Fi Collar'),
            ('petpace', 'PetPace'),
            ('other', 'Other'),
        ]
    )
    device_id = models.CharField(max_length=200)  # Device's unique ID
    name = models.CharField(max_length=100, default="My Pet's Collar")
    
    # API credentials (encrypted)
    api_key_encrypted = models.BinaryField()
    api_secret_encrypted = models.BinaryField(blank=True)
    
    is_active = models.BooleanField(default=True)
    last_sync = models.DateTimeField(null=True, blank=True)
    sync_frequency_minutes = models.IntegerField(default=60)
    
    created_at = models.DateTimeField(auto_now_add=True)

class WearableData(models.Model):
    """Time-series data from wearables"""
    device = models.ForeignKey(WearableDevice, on_delete=models.CASCADE, related_name='data')
    timestamp = models.DateTimeField()
    
    # Activity
    steps = models.IntegerField(null=True, blank=True)
    activity_minutes = models.IntegerField(null=True, blank=True)  # Active minutes
    rest_minutes = models.IntegerField(null=True, blank=True)
    calories_burned = models.FloatField(null=True, blank=True)
    
    # Health
    heart_rate = models.IntegerField(null=True, blank=True)  # BPM
    temperature = models.FloatField(null=True, blank=True)  # °C
    sleep_quality = models.FloatField(null=True, blank=True)  # 0-100 score
    sleep_duration_minutes = models.IntegerField(null=True, blank=True)
    
    # Alerts
    alert_type = models.CharField(
        max_length=30,
        choices=[
            ('none', 'None'),
            ('low_activity', 'Low Activity'),
            ('high_temp', 'High Temperature'),
            ('irregular_heart', 'Irregular Heart Rate'),
            ('poor_sleep', 'Poor Sleep'),
            ('anomaly', 'Anomaly Detected'),
        ],
        default='none'
    )
    alert_details = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['device', '-timestamp']),
            models.Index(fields=['device', 'alert_type']),
        ]

class WearableAlert(models.Model):
    """Alerts generated from wearable data analysis"""
    pet = models.ForeignKey(Pet, on_delete=models.CASCADE, related_name='wearable_alerts')
    device = models.ForeignKey(WearableDevice, on_delete=models.CASCADE)
    
    alert_type = models.CharField(max_length=30)
    severity = models.CharField(
        max_length=10,
        choices=[('info', 'Info'), ('warning', 'Warning'), ('critical', 'Critical')]
    )
    title = models.CharField(max_length=200)
    description = models.TextField()
    
    is_read = models.BooleanField(default=False)
    is_acknowledged = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
```

### API Endpoints

```
POST   /api/wearables/                — Register a device
GET    /api/wearables/                — List user's devices
GET    /api/wearables/{id}/           — Device details
PUT    /api/wearables/{id}/           — Update device settings
DELETE /api/wearables/{id}/           — Remove device
POST   /api/wearables/{id}/sync/      — Trigger manual sync
GET    /api/wearables/{id}/data/      — Get device data (paginated, filterable by date)

GET    /api/wearables/alerts/         — List alerts
PUT    /api/wearables/alerts/{id}/    — Mark as read/acknowledged

# Webhook endpoints for device push
POST   /api/wearables/webhook/fitbark/  — FitBark webhook
POST   /api/wearables/webhook/whistle/  — Whistle webhook
```

### Integration Notes

**FitBark API:**
- REST API: `https://app.fitbark.com/api/v2`
- Auth: OAuth2
- Endpoints: `/activity_series`, `/sleep`, `/bark_count`
- Rate limit: 100 requests/hour

**Whistle API:**
- No public API — requires partnership agreement
- Alternative: manual data import (CSV upload)

**Recommended approach for MVP:**
1. Start with manual CSV import (works with any device)
2. Add FitBark OAuth integration (most accessible API)
3. Add webhook support for real-time data

### UI Components

1. **Device Setup** (`/wearables/add/`)
   - Device type selector (FitBark, Whistle, Fi, PetPace, Other)
   - OAuth flow for supported devices
   - Manual entry for unsupported devices
   - Link to pet

2. **Device Dashboard** (`/wearables/<id>/`)
   - Current stats: steps, activity, heart rate, temperature
   - 7-day trend charts (activity, sleep, temperature)
   - Alert history
   - Sync status + "Sync Now" button

3. **Pet Health Dashboard Integration**
   - Add wearable data section to pet profile
   - Daily activity summary
   - Alert badges on pet card

4. **Alerts Center** (`/wearables/alerts/`)
   - List of all alerts across all pets
   - Filter by severity, pet, device
   - "Acknowledge" + "Add to Timeline" actions

---

## 🔧 Shared Components

### Navigation Updates

Add to main navigation:
```
Dashboard | AI Chat 🤖 | Telemedicine 📹 | My Pets 🐾 | Health Timeline 📋 | Dosage Calculator 💊 | Wearables 📡
```

### Notification System

```python
class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    type = models.CharField(max_length=50)  # 'consultation_reminder', 'medication_reminder', 'wearable_alert', 'vaccination_due'
    title = models.CharField(max_length=200)
    message = models.TextField()
    link = models.CharField(max_length=500, blank=True)  # Deep link
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
```

### Settings Page Updates

Add sections:
- **Notifications**: Email, SMS, push preferences
- **Connected Devices**: Manage wearables
- **Data & Privacy**: Export data, delete account
- **Billing**: Subscription management

---

## 📊 Database Migration Order

Run migrations in this order to avoid foreign key issues:

1. `Pet` model (if not existing)
2. `ChatSession`, `ChatMessage`, `SymptomTag`, `ChatSymptom`
3. `Veterinarian`, `ConsultationSlot`, `VideoConsultation`
4. `HealthEvent`, `MedicationReminder`
5. `MedicationDatabase`, `DosageCalculation`
6. `WearableDevice`, `WearableData`, `WearableAlert`
7. `Notification`

---

## 🧪 Testing Checklist

### AI Chat Assistant
- [ ] Create chat session
- [ ] Send message, receive AI response
- [ ] Urgency assessment accuracy
- [ ] Free tier limit enforcement
- [ ] Multi-language support
- [ ] Session persistence
- [ ] Save to health timeline

### Telemedicine
- [ ] Vet listing and filtering
- [ ] Slot availability display
- [ ] Booking flow (all steps)
- [ ] Video room creation and joining
- [ ] Consultation notes saved
- [ ] Auto-save to health timeline
- [ ] Cancellation flow

### Health Timeline
- [ ] Add/view/edit/delete events
- [ ] Filter by type
- [ ] Document upload
- [ ] Share link generation
- [ ] Medication reminders
- [ ] PDF export

### Dosage Calculator
- [ ] Medication search
- [ ] Dose calculation accuracy
- [ ] Liquid medication volume calculation
- [ ] Drug interaction checking
- [ ] Calculation history
- [ ] Save to pet records

### Wearables
- [ ] Device registration
- [ ] Manual data import (CSV)
- [ ] FitBark OAuth flow
- [ ] Data visualization (charts)
- [ ] Alert generation
- [ ] Alert notifications

---

## 📁 File Structure

```
pet_vet_ai/
├── chat/
│   ├── models.py          # ChatSession, ChatMessage, SymptomTag
│   ├── views.py           # Chat API views
│   ├── ai_service.py      # AI prompt and response handling
│   ├── urls.py
│   └── templates/chat/
│       ├── session_list.html
│       └── chat_room.html
├── telemedicine/
│   ├── models.py          # Veterinarian, ConsultationSlot, VideoConsultation
│   ├── views.py           # Booking and video views
│   ├── video_service.py   # Daily.co/Twilio integration
│   ├── urls.py
│   └── templates/telemedicine/
│       ├── vet_list.html
│       ├── booking.html
│       └── video_call.html
├── pets/
│   ├── models.py          # Pet, HealthEvent, MedicationReminder
│   ├── views.py           # Pet profile and timeline views
│   ├── timeline_export.py # PDF export
│   ├── urls.py
│   └── templates/pets/
│       ├── pet_list.html
│       ├── pet_profile.html
│       └── timeline.html
├── medications/
│   ├── models.py          # MedicationDatabase, DosageCalculation
│   ├── views.py           # Calculator views
│   ├── calculator.py      # Dose calculation logic
│   ├── interactions.py    # Drug interaction checking
│   ├── urls.py
│   └── templates/medications/
│       ├── calculator.html
│       └── history.html
├── wearables/
│   ├── models.py          # WearableDevice, WearableData, WearableAlert
│   ├── views.py           # Device management views
│   ├── integrations/
│   │   ├── fitbark.py     # FitBark API client
│   │   ├── csv_import.py  # Manual CSV import
│   │   └── webhooks.py    # Webhook handlers
│   ├── urls.py
│   └── templates/wearables/
│       ├── device_list.html
│       ├── device_dashboard.html
│       └── alerts.html
└── notifications/
    ├── models.py          # Notification
    ├── tasks.py           # Celery tasks for reminders
    └── services.py        # Notification dispatch
```

---

## 🚀 Recommended Implementation Order

1. **Pet model + Health Timeline** (Foundation — other features depend on this)
2. **AI Chat Assistant** (Highest user value, can use existing AI infrastructure)
3. **Dosage Calculator** (Relatively simple, high perceived value)
4. **Telemedicine Booking** (Complex — video integration, scheduling logic)
5. **Wearable Integration** (MVP: manual CSV import → FitBark OAuth → webhooks)

---

*This is a living document. Update as implementation progresses and new requirements emerge.*
