Output recording of Web UI channel - https://drive.google.com/file/d/1GpLaM2ymyDChYf9XFsl2B_R4af7kUAod/view?usp=drive_link
Output recording of WhatsApp channel via Mobile Phone - https://drive.google.com/file/d/1jdH45zFpfvuc1GiRQ0sa4__ra_WKWqrt/view?usp=drive_link
Output recoeding of WhatsApp (HF/backend) - https://drive.google.com/file/d/1sTEQGIYj2ztpMHG_-Fxp1Jts5BTpEnPG/view?usp=drive_link

<img width="214" height="430" alt="image" src="https://github.com/user-attachments/assets/fd5050b6-9815-4692-8ade-fc85f66e67e2" />

<img width="214" height="430" alt="image" src="https://github.com/user-attachments/assets/c32e6c0e-65c0-489c-9fea-f34609224996" />

<img width="214" height="430" alt="image" src="https://github.com/user-attachments/assets/8815e3f6-32ac-4f35-a402-cbaeda8e09ac" />

<img width="214" height="430" alt="image" src="https://github.com/user-attachments/assets/b7a7953a-8cd3-44a1-a639-8a6f76515803" />

<img width="832" height="418" alt="image" src="https://github.com/user-attachments/assets/e157c9c4-a091-4601-aaa5-0e6cbccd8c2d" />

---
# 🌿 SafeSpace AI 2.0

> **AI-powered Medical & Mental Health Assistant**  
> Multi-agent · Multimodal · WhatsApp + Web · Production Deployed

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-green)](https://huggingface.co/spaces/HiteshiAglawe0505/safespace-ai)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-orange)](https://crewai.com)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-red)](https://groq.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-purple)](https://smith.langchain.com)

---

## What is SafeSpace AI?

SafeSpace is a **production-deployed, multi-channel AI health assistant** that provides:
- 🩺 **Medical guidance** — symptom analysis, home care advice, red-flag detection
- 🧠 **Mental health support** — CBT-based therapy conversations, crisis detection
- 🚨 **Emergency escalation** — auto-triggers a real phone call via Twilio when crisis is detected
- 📍 **Therapist finder** — real clinic listings via Google Maps Places API
- 🖼️ **Medical image analysis** — upload a photo of a rash, wound, or scalp condition
- 🎙️ **Voice message support** — speak your symptoms, Whisper transcribes them

Available on **WhatsApp** (via Twilio) and a **Streamlit web app** — both powered by the same backend deployed on Hugging Face Spaces.

---

## 🏗️ Architecture

```
User (WhatsApp / Web Browser)
         │
         ▼
┌─────────────────────────────────┐
│         NGINX (port 7860)       │  ← single public port on HF Spaces
│  /whatsapp/webhook → FastAPI    │
│  /*                → Streamlit  │
└─────────────────────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
FastAPI    Streamlit
(port 8000) (port 8501)
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│              CORE ENGINE                    │
│                                             │
│  Multimodal Handler                         │
│  (image → Llama 4 Vision, audio → Whisper)  │
│         ↓                                   │
│  Intent Classifier (Groq structured output) │
│         ↓                                   │
│  ┌─────────────────────────────────────┐    │
│  │         CrewAI Crew                 │    │
│  │  SafetyAgent → DoctorAgent          │    │
│  │  SafetyAgent → TherapistAgent       │    │
│  │  (sequential, context-sharing)      │    │
│  └─────────────────────────────────────┘    │
│         ↓                                   │
│  SQLite Session Memory  ←→  LangSmith       │
└─────────────────────────────────────────────┘
         │
         ▼
   ChatResponse → back to WhatsApp/Web
```

---

## 🤖 Agentic Design — The 3-Agent Crew

Every message passes through **all 3 agents in sequence**, regardless of channel:

| Agent | Role | Tools |
|---|---|---|
| **SafetyAgent** | Crisis detection — checks every message for suicidal ideation, self-harm, emergencies | `emergency_call` (Twilio) |
| **DoctorAgent** | Medical guidance — symptoms, home care, red flags. Never diagnoses | None |
| **TherapistAgent** | Emotional support — CBT techniques, coping strategies, therapist search | `find_nearby_therapists` (Google Maps) |

**Why sequential?** SafetyAgent runs first on every message — no medical or therapy response is ever sent before checking if it's a crisis. This is a guardrail, not just a feature.

---

## 🔌 MCP Server — Therapist Directory

The therapist finder is exposed as an **MCP (Model Context Protocol) server** — a standard by Anthropic for AI tool communication.

```
TherapistAgent
    ↓ calls
MCP Server (mcp_server/therapist_directory.py)
    ↓ exposes 3 tools
  search_by_location()
  get_therapist_details()
  list_specialties()
    ↓ queries
Google Maps Places API
    ↓ returns
Real clinic listings with ratings, hours, addresses
```

**Why MCP?** The therapist search logic is completely decoupled from the agent. Any future agent or external app can call the same MCP server — the data source is swappable without touching agent code.

---

## 📊 LLMOps — Observability with LangSmith

Every request is traced end-to-end in LangSmith:

```
handle_request()
    │ LangSmith parent run
    ├── intent_classifier() → logged
    ├── safety_task()       → logged  
    ├── doctor/therapist_task() → logged
    └── response            → logged with latency, tokens, intent
```

- **Intent distribution** — see what % of messages are MEDICAL vs THERAPY vs CRISIS or MIXED
- **Latency tracking** — per-agent execution time
- **Error monitoring** — failed runs captured with full traceback
- **Dataset building** — conversations logged for future fine-tuning

---

## 🛠️ Tech Stack

| Category | Technology | Why |
|---|---|---|
| **LLM** | Groq — Llama 3.3 70B | Fastest inference, free tier |
| **Vision** | Groq — Llama 4 Scout | Medical image analysis |
| **Speech** | Groq — Whisper large-v3 | Voice message transcription |
| **Agents** | CrewAI | Sequential multi-agent orchestration |
| **Intent** | Groq structured output (Pydantic) | Reliable JSON classification |
| **API** | FastAPI + Uvicorn | WhatsApp webhook receiver |
| **Web UI** | Streamlit 1.57 | Rapid multimodal chat UI |
| **Memory** | SQLite + aiosqlite | Lightweight session persistence |
| **WhatsApp** | Twilio | Webhook + media download + outbound SMS |
| **Maps** | Google Maps Places API | Real therapist/clinic listings |
| **Observability** | LangSmith | LLM tracing and monitoring |
| **Deployment** | Hugging Face Spaces (Docker) | Free, permanent URL, secrets management |
| **Proxy** | Nginx | Routes WebSocket + webhook on single port |
| **Protocol** | MCP (Model Context Protocol) | Decoupled tool server for therapist search |

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/safespace-ai.git
cd safespace-ai

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Fill in: GROQ_API_KEY, GOOGLE_MAPS_API_KEY, LANGSMITH_API_KEY
# Optional: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN (for WhatsApp)

# 4. Run Web UI
streamlit run interfaces/streamlit_ui/app.py

# 5. Run WhatsApp Backend (separate terminal)
python app.py
# Then expose with ngrok: ngrok http 8000
# Set Twilio webhook to: https://your-ngrok-url/whatsapp/webhook
```

---

## 🌐 Deployment (Hugging Face Spaces)

```bash
# Install HF CLI
pip install huggingface_hub

# Deploy (no GitHub push needed)
python deploy_to_hf.py --username YOUR_HF_USERNAME --token hf_xxxxx
```

Then add secrets in Space Settings:
- `GROQ_API_KEY`
- `GOOGLE_MAPS_API_KEY`
- `LANGSMITH_API_KEY`
- `TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN` (for WhatsApp)

**Live:** https://huggingface.co/spaces/HiteshiAglawe0505/safespace-ai

---

## 📁 Project Structure

```
safespace-ai/
├── app.py                          # FastAPI entrypoint
├── hf_app.py                       # HF Spaces: starts Nginx + FastAPI + Streamlit
├── Dockerfile                      # Docker config for HF deployment
├── requirements.txt
├── .env.example
│
├── core/
│   ├── config.py                   # Pydantic-settings: all env vars validated at startup
│   ├── schemas.py                  # ChatRequest, ChatResponse, Intent, MessageType
│   └── engine.py                   # handle_request(): routes all input types
│
├── agents/
│   ├── crew.py                     # CrewAI crew: wires agents + tasks + memory
│   ├── intent_classifier.py        # Groq structured output → Intent enum
│   ├── safety.py                   # SafetyAgent: crisis detection + emergency call
│   ├── doctor.py                   # DoctorAgent: medical guidance
│   └── therapist.py                # TherapistAgent: mental health + therapist search
│
├── tools/
│   ├── maps_tool.py                # FindTherapistsTool → Google Maps Places API
│   └── emergency_tool.py           # EmergencyCallTool → Twilio voice call
│
├── mcp_server/
│   └── therapist_directory.py      # MCP server: exposes therapist search as tools
│
├── multimodal/
│   ├── vision.py                   # describe_image() → Groq Llama 4 Vision
│   └── speech.py                   # transcribe_audio() → Groq Whisper
│
├── memory/
│   └── store.py                    # SQLite session store + history formatter
│
├── interfaces/
│   ├── whatsapp/
│   │   ├── webhook.py              # POST /whatsapp/webhook (Twilio)
│   │   └── sender.py               # send_whatsapp_message()
│   └── streamlit_ui/
│       └── app.py                  # Streamlit: text + image + voice UI
│
└── observability/
    └── tracer.py                   # LangSmith run logging
```

---

## 🔑 Key Technical Decisions

**1. Why CrewAI over LangChain agents?**
CrewAI's sequential crew ensures SafetyAgent always runs first — it's architecturally impossible to skip crisis detection. With LangChain, the router might send medical messages directly to DoctorAgent.

**2. Why Pydantic for intent classification?**
Groq's structured output with Pydantic guarantees the LLM returns a valid `Intent` enum — no string parsing, no hallucinated intent values, no try/catch spaghetti.

**3. Why SQLite over Redis/Postgres?**
Session memory only needs last 10 messages per user. SQLite is zero-infrastructure, async-capable, and sufficient for prototype scale. Easy to swap for Postgres in production.

**4. Why Nginx on HF Spaces?**
HF Docker Spaces expose only one port. Nginx routes `/whatsapp/*` to FastAPI and `/*` (including WebSocket) to Streamlit — both in the same container. A Python proxy blocks WebSocket upgrades; Nginx handles them natively.

**5. Why MCP for the therapist directory?**
Decouples the data source from the agent. The TherapistAgent doesn't know if data comes from Google Maps, a hospital database, or a static list — it just calls the MCP tool. The backend is swappable without touching agent code.

---

## 📸 Screenshots

| WhatsApp | Web App |
|---|---|
| Medical advice + image analysis | Streamlit dark UI |
| Therapist finder with real listings | Live mic recording |
| Crisis detection + emergency call | Intent badges (MEDICAL/THERAPY) |

---

## 👩‍💻 Author

**Hiteshi Aglawe**  
Aspiring GenAI / ML Engineer  
[GitHub](https://github.com/HiteshiAglawe0505) · [HuggingFace](https://huggingface.co/HiteshiAglawe0505)

---

## ⚠️ Disclaimer

SafeSpace AI is a **prototype assistant** — not a substitute for professional medical or mental health care. Always consult a qualified doctor or therapist for medical decisions. 
