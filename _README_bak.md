# SafeSpace AI 2.0 — Enhanced

> AI-powered Medical & Mental Health Assistant | Multi-agent · Multimodal · WhatsApp + Web

---

## Architecture

```
Any Channel (WhatsApp / Streamlit / REST)
        ↓
  Channel Adapter  →  ChatRequest
        ↓
    Core Engine
        ├── Pydantic Intent Classifier
        ├── CrewAI Crew
        │     ├── SafetyAgent   (crisis detection, emergency escalation)
        │     ├── DoctorAgent   (symptom guidance, red flags)
        │     └── TherapistAgent (emotional support, therapist finder)
        └── SQLite Memory  ←→  LangSmith Tracing
        ↓
   ChatResponse
        ↓
  Channel Adapter  →  User
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agents | CrewAI (hierarchical crew) |
| LLM | Groq — Llama 3.3 70B |
| Vision | Groq — Llama 4 Maverick |
| Speech | Groq — Whisper large-v3 |
| Web framework | FastAPI |
| Memory | SQLite (aiosqlite) |
| Observability | LangSmith |
| WhatsApp | Twilio |
| Maps | Google Maps API |
| Web UI | Streamlit |
| Deployment | Railway / Render |

## Setup

```bash
# 1. Clone and install
git clone <repo>
cd safespace-ai
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Fill in your API keys in .env

# 3. Run
python app.py
```

## Project Structure

```
safespace2/
├── app.py                        # FastAPI entrypoint
├── pyproject.toml
├── .env.example
├── core/
│   ├── config.py                 # All settings (pydantic-settings)
│   ├── schemas.py                # ChatRequest, ChatResponse, Intent
│   └── engine.py                 # Main handle_request() entry point
├── agents/
│   ├── crew.py                   # CrewAI crew definition
│   ├── intent_classifier.py      # Pydantic structured intent
│   ├── doctor.py                 # DoctorAgent
│   ├── therapist.py              # TherapistAgent
│   └── safety.py                 # SafetyAgent
├── tools/
│   ├── maps_tool.py              # Google Maps therapist finder
│   └── emergency_tool.py        # Twilio emergency call
├── memory/
│   └── store.py                  # SQLite session memory
├── multimodal/
│   ├── speech.py                 # Whisper transcription
│   └── vision.py                 # Llama 4 Maverick image analysis
├── interfaces/
│   ├── whatsapp/
│   │   ├── webhook.py            # Twilio webhook adapter
│   │   └── sender.py             # WhatsApp message sender
│   └── streamlit_ui/
│       └── app.py                # Streamlit web UI
├── observability/
│   └── tracer.py                 # LangSmith tracing
└── tests/
    ├── test_engine.py
    ├── test_agents.py
    └── test_memory.py
```

---
title: SafeSpace AI
emoji: 🌿
colorFrom: green
colorTo: green
sdk: streamlit
sdk_version: "1.40.0"
app_file: hf_app.py
pinned: false
---

# SafeSpace AI 2.0

> AI-powered Medical & Mental Health Assistant  
> Multi-agent · Multimodal · WhatsApp + Web

Built with CrewAI, Groq (Llama 3.3 70B), LangSmith tracing, and OpenStreetMap for therapist discovery.

## Features
- 🩺 Medical symptom guidance (DoctorAgent)
- 🧠 Mental health support (TherapistAgent)  
- 🚨 Crisis detection & emergency escalation (SafetyAgent)
- 📍 Nearby therapist finder (OpenStreetMap — no API key)
- 🗣️ Voice & image input support
- 📊 LangSmith observability tracing
- 🔌 MCP server for therapist directory (optional)

## Tech Stack
CrewAI · Groq · LangSmith · FastAPI · Streamlit · SQLite · Google Maps API