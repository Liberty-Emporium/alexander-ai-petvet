# Pet Vet AI

AI-powered pet health diagnostic app that analyzes photos to identify potential health issues in animals.

## Live Demo

**URL:** https://pet-vet-ai-production.up.railway.app

## Features

### Core Features
- ✅ **Photo Analysis** - Upload pet photos for AI health analysis
- ✅ **Symptom Checker** - Select symptoms and get AI-powered diagnosis
- ✅ **Diagnostic Database** - 500+ conditions across animal types
- ✅ **Treatment Recommendations** - Home care vs vet visit guidance
- ✅ **Vet Network** - Find nearby vets (/vets)
- ✅ **Pet Profiles** - Add and manage your pets (/pets)

### Animals Covered
- **Farm Animals** - Horses, Goats, Sheep, Cows, Pigs, Chickens, Rabbits
- **Companion Animals** - Dogs, Cats, Birds
- **Exotic/Small Pets** - Snakes, Lizards, Turtles, Hamsters, Fish

### Account Features
- User registration/login
- Subscription tiers (Free, Premium, Enterprise)
- Usage tracking
- Profile management

## Routes

| Route | Description |
|-------|-------------|
| `/` | Homepage |
| `/diagnose` | Start diagnosis |
| `/settings` | API key configuration |
| `/vets` | Find nearby vets |
| `/pets` | Manage pet profiles |
| `/register` | Create account |
| `/login` | User login |

## Tech Stack

- Python (Flask)
- Groq API for AI
- Deploy on Railway

## API Configuration

To enable photo analysis, add your Groq API key in Settings or set environment variable:
- `GROQ_API_KEY`

Get a free key at: https://console.groq.com

## Development

```bash
pip install -r requirements.txt
python app.py
```

## Subscription Tiers

| Tier | Price | Diagnoses/month |
|------|-------|-----------------|
| Free | $0 | 5 |
| Premium | $4.99/mo | Unlimited |
| Enterprise | $99/mo | Unlimited + priority |

---

*Last updated: 2026-04-10*
