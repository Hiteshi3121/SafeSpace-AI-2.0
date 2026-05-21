# 🌿 SafeSpace AI 2.0

> **AI-powered Medical & Mental Health Assistant**
> Multi-agent · Multimodal · WhatsApp + Web · Production Deployed

<img width="214" height="430" alt="image" src="https://github.com/user-attachments/assets/fd5050b6-9815-4692-8ade-fc85f66e67e2" />

<img width="214" height="430" alt="image" src="https://github.com/user-attachments/assets/c32e6c0e-65c0-489c-9fea-f34609224996" />

<img width="214" height="430" alt="image" src="https://github.com/user-attachments/assets/8815e3f6-32ac-4f35-a402-cbaeda8e09ac" />

<img width="214" height="430" alt="image" src="https://github.com/user-attachments/assets/b7a7953a-8cd3-44a1-a639-8a6f76515803" />

<img width="832" height="418" alt="image" src="https://github.com/user-attachments/assets/e157c9c4-a091-4601-aaa5-0e6cbccd8c2d" />


Output recording of Web UI channel - https://drive.google.com/file/d/1GpLaM2ymyDChYf9XFsl2B_R4af7kUAod/view?usp=drive_link
Output recording of WhatsApp channel via Mobile Phone - https://drive.google.com/file/d/1jdH45zFpfvuc1GiRQ0sa4__ra_WKWqrt/view?usp=drive_link
Output recoeding of WhatsApp (HF/backend) - https://drive.google.com/file/d/1sTEQGIYj2ztpMHG_-Fxp1Jts5BTpEnPG/view?usp=drive_link


## What is SafeSpace AI?

SafeSpace is a **production-deployed, multi-channel AI health assistant** that provides:

- 🩺 **Medical guidance** — symptom analysis, image reading (blood reports, X-rays, ECG, skin conditions), home care advice, red-flag detection
- 🧠 **Mental health support** — CBT-based therapy conversations, emotional support, crisis detection
- 🚨 **Emergency escalation** — auto-triggers a real Twilio voice call when suicidal ideation or crisis is detected
- 📍 **Therapist finder** — real clinic listings via Google Maps Places API through an MCP server
- 🖼️ **Medical image analysis** — upload a photo of a rash, wound, blood report, X-ray, or ECG
- 🎙️ **Voice message support** — speak your symptoms on WhatsApp or web; Whisper transcribes them

Available on **WhatsApp** (via Twilio) and a **Streamlit web app** — both powered by the same backend deployed on Hugging Face Spaces with Nginx routing everything through a single public port.

---

## 🏗️ Architecture

```
User (WhatsApp / Web Browser)
         │
         ▼
┌──────────────────────────────────────────┐
│         NGINX  (port 7860 — public)      │
│                                          │
│  /whatsapp/*  →  FastAPI   :8000         │
│  /mcp/*       →  MCP Server:8001  ← NEW  │
│  /*           →  Streamlit :8501         │
└──────────────────────────────────────────┘
         │
    ┌────┴──────────────┐
    │                   │
    ▼                   ▼
FastAPI              Streamlit
(WhatsApp webhook)   (Web UI)
    │                   │
    └────────┬──────────┘
             │ same engine for both channels
             ▼
┌──────────────────────────────────────────────────┐
│                   CORE ENGINE                    │
│                                                  │
│  Multimodal Handler                              │
│  (image → Llama 4 Scout Vision)                  │
│  (audio → Groq Whisper large-v3)                 │
│                  ↓                               │
│  Intent Classifier                               │
│  (Groq structured output + Pydantic)             │
│  → MEDICAL / THERAPY / MIXED / UNKNOWN           │
│                  ↓                               │
│  ┌──────────────────────────────────────────┐    │
│  │           CrewAI Crew (sequential)       │    │
│  │                                          │    │
│  │  SafetyAgent   ← always runs FIRST       │    │
│  │       ↓                                  │    │
│  │  DoctorAgent   or  TherapistAgent        │    │
│  │  (context from SafetyAgent shared)       │    │
│  └──────────────────────────────────────────┘    │
│                  ↓                               │
│  SQLite Session Memory  ←→  LangSmith Tracing    │
└──────────────────────────────────────────────────┘
         │
         ▼
   ChatResponse → WhatsApp (Twilio) / Streamlit UI


MCP Server (port 8001 — internal)
  TherapistAgent → maps_tool.py → POST /search
                → therapist_directory.py → Google Maps
```

---

## 🤖 Agentic Design — The 3-Agent Crew

Every message passes through **all agents in sequence**, regardless of channel:

| Agent | Role | Tools |
|---|---|---|
| **SafetyAgent** | Crisis detection — checks every message for suicidal ideation, self-harm, emergencies. Runs FIRST, always. | `emergency_call` (Twilio voice) |
| **DoctorAgent** | Medical guidance — symptoms, image interpretation, home care, red flags. Never diagnoses. | None |
| **TherapistAgent** | Emotional support — CBT techniques, coping strategies, therapist search | `find_nearby_therapists` (Google Maps via MCP) |

**Why sequential?** SafetyAgent runs first on every single message — no medical or therapy response is ever sent before checking if it is a crisis. This is an architectural guarantee, not just a prompt instruction.

**Why context sharing?** Both DoctorAgent and TherapistAgent receive the SafetyAgent output via `context=[safety_task]`. If safety flagged a concern, the second agent's response accounts for it automatically.

---

## 🔌 MCP Server — Therapist Directory (Live on HF)

The therapist finder is exposed as an **MCP (Model Context Protocol) server** — a standard by Anthropic for AI tool communication. The MCP server runs inside the same Docker container and is publicly reachable via Nginx routing.

```
TherapistAgent calls find_nearby_therapists tool
         ↓
maps_tool.py detects THERAPIST_MCP_URL is set
         ↓  HTTP POST to http://127.0.0.1:8001/search
MCP Server (mcp_server/therapist_directory.py)
         ↓  exposes 3 tools via /tools discovery endpoint
  search_by_location()   →  POST /search
  get_therapist_details() →  POST /details
  list_specialties()      →  GET  /specialties
         ↓
Google Maps Places API
         ↓
Real clinic listings with names, addresses, ratings, open/closed status
```

**MCP endpoints (publicly accessible):**
- `https://hiteshiaglawe0505-safespace-ai.hf.space/mcp/health`
- `https://hiteshiaglawe0505-safespace-ai.hf.space/mcp/tools`
- `https://hiteshiaglawe0505-safespace-ai.hf.space/mcp/docs`

**Why MCP?** The therapist search logic is completely decoupled from the agent. Any future agent, external app, or partner service can call the same MCP server. The data source (Google Maps today, Practo tomorrow) is swappable without touching a single line of agent code.

**Fallback design:** If the MCP server is temporarily unavailable (e.g., restarting), `maps_tool.py` automatically falls back to calling Google Maps directly. Users always receive results.

---

## 📊 LLMOps — Observability with LangSmith

Every request is traced in LangSmith at `smith.langchain.com` under project `safespace-ai`.

**What is logged per request (Traces tab):**

| Field | Example |
|---|---|
| Input: user_id | `web_a66a****` / `whatsapp:+91*******590` |
| Input: channel | `web` / `whatsapp` |
| Input: message_type | `text` / `image` / `audio` |
| Output: intent | `THERAPY` / `MEDICAL` / `MIXED` |
| Output: confidence | `0.95` |
| Output: text | Full response sent to user |
| Output: escalated | `false` / `true` (emergency call fired) |
| Output: latency_ms | `5770` |

**Tool calls (Runs tab → filter run_type=tool):**
- `find_nearby_therapists` — location, output clinic list, latency
- `emergency_call` — crisis reason, Twilio SID, latency

**Dataset:** Every request logs intent + confidence to the `safespace_intent_classification` dataset for future fine-tuning.

**Note on LLM call tracing:** CrewAI 1.14.4 uses LiteLLM directly (bypassing LangChain wrappers) and resets LiteLLM callbacks after each run — blocking third-party callback nesting. Agent-level LLM calls are observable at the trace level via latency but are not individually nested under the parent run. This is a known CrewAI 1.14.4 + LangSmith version compatibility constraint.

---

## 🛠️ Tech Stack

| Category | Technology | Why |
|---|---|---|
| **LLM** | Groq — Llama 3.3 70B | Fastest inference, free tier |
| **Vision** | Groq — Llama 4 Scout | Medical image analysis on same API |
| **Speech** | Groq — Whisper large-v3 | Voice message transcription |
| **Agents** | CrewAI 1.14.4 | Sequential multi-agent orchestration |
| **Intent** | Groq structured output + Pydantic | Reliable JSON classification, no string parsing |
| **API** | FastAPI + Uvicorn | Async WhatsApp webhook receiver |
| **Web UI** | Streamlit 1.57 | Native st.audio_input() mic, rapid multimodal UI |
| **Memory** | SQLite + aiosqlite | Lightweight async session persistence |
| **WhatsApp** | Twilio | Webhook + media download + voice calls |
| **Maps** | Google Maps Places API | Real therapist/clinic listings with India data |
| **Observability** | LangSmith | Request tracing, tool monitoring, intent datasets |
| **Deployment** | Hugging Face Spaces (Docker) | Free, permanent URL, built-in secrets management |
| **Proxy** | Nginx | Routes WebSocket + webhook + MCP on single port 7860 |
| **Protocol** | MCP (Model Context Protocol) | Decoupled tool server for therapist directory |

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/HiteshiAglawe0505/safespace-ai.git
cd safespace-ai

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Required: GROQ_API_KEY, GOOGLE_MAPS_API_KEY
# Optional: LANGSMITH_API_KEY, TWILIO_* (for WhatsApp), THERAPIST_MCP_URL

# 4. Run Web UI only
streamlit run interfaces/streamlit_ui/app.py

# 5. Run WhatsApp backend (separate terminal)
python app.py
# Expose with: ngrok http 8000
# Set Twilio webhook: https://your-ngrok-url/whatsapp/webhook

# 6. Run MCP server (optional, separate terminal)
python mcp_server/therapist_directory.py
# Add to .env: THERAPIST_MCP_URL=http://localhost:8001
```

---

## 🌐 Deployment (Hugging Face Spaces)

```bash
pip install huggingface_hub
python fix_hf_upload.py --username HiteshiAglawe0505 --token hf_xxxxx
```

Add secrets in Space Settings → Secrets:

| Secret | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | ✅ | LLM, vision, speech |
| `GOOGLE_MAPS_API_KEY` | ✅ | Therapist search |
| `LANGSMITH_API_KEY` | Recommended | Observability |
| `LANGSMITH_PROJECT` | Recommended | Project name |
| `LANGSMITH_TRACING` | Recommended | Enable tracing |
| `TWILIO_ACCOUNT_SID` | WhatsApp only | Webhook auth |
| `TWILIO_AUTH_TOKEN` | WhatsApp only | Webhook auth |
| `TWILIO_FROM_NUMBER` | WhatsApp only | Sender number |
| `TWILIO_WHATSAPP_NUMBER` | WhatsApp only | Sandbox number |
| `EMERGENCY_CONTACT` | WhatsApp only | Crisis call target |

`THERAPIST_MCP_URL` is **not** a secret — it is hardcoded as `http://127.0.0.1:8001` inside `hf_app.py` since the MCP server runs inside the same container.

**Live:** https://huggingface.co/spaces/HiteshiAglawe0505/safespace-ai

---

## 📁 Project Structure

```
safespace-ai/
├── app.py                          # Local dev: FastAPI entry point only
├── hf_app.py                       # HF Spaces: starts Nginx + FastAPI + MCP + Streamlit
├── Dockerfile
├── requirements.txt                # crewai==1.14.4 pinned (cache_breakpoint compatibility)
├── .env.example
│
├── core/
│   ├── config.py                   # pydantic-settings: all env vars validated at startup
│   ├── schemas.py                  # ChatRequest, ChatResponse, Intent, MessageType
│   └── engine.py                   # handle_request(): multimodal routing + LangSmith tracing
│
├── agents/
│   ├── crew.py                     # CrewAI crew: agents, tasks, location detection, kickoff
│   ├── intent_classifier.py        # Groq JSON mode + Pydantic → Intent enum
│   ├── safety.py                   # SafetyAgent: crisis detection + emergency call
│   ├── doctor.py                   # DoctorAgent: medical guidance, 80/20 response structure
│   └── therapist.py                # TherapistAgent: CBT support + therapist search
│
├── tools/
│   ├── maps_tool.py                # FindTherapistsTool → MCP server → Google Maps
│   └── emergency_tool.py           # EmergencyCallTool → Twilio voice API
│
├── mcp_server/
│   └── therapist_directory.py      # MCP server on :8001 — 3 tools over Google Maps Places API
│
├── multimodal/
│   ├── vision.py                   # describe_image() → Groq Llama 4 Scout
│   └── speech.py                   # transcribe_audio() → Groq Whisper large-v3
│
├── memory/
│   └── store.py                    # SQLite: save/load session, format last 10 msgs for prompt
│
├── interfaces/
│   ├── whatsapp/
│   │   ├── webhook.py              # POST /whatsapp/webhook — receives Twilio form data
│   │   └── sender.py               # send_whatsapp_message() → Twilio Messages API
│   └── streamlit_ui/
│       └── app.py                  # Chat UI: text + image upload + st.audio_input() mic
│
└── observability/
    └── tracer.py                   # LangSmith: parent runs, tool @traceable, intent dataset
```

---

## 🔑 Key Technical Decisions

**1. Why CrewAI sequential over LangChain agents?**
CrewAI's `Process.sequential` guarantees SafetyAgent always runs first — architecturally impossible to skip crisis detection. With a single LangChain agent, the router might send medical messages directly to DoctorAgent without a safety check.

**2. Why Pydantic structured output for intent?**
Groq JSON mode + Pydantic `IntentResult` guarantees a valid `Intent` enum with confidence and reasoning. No string parsing, no hallucinated values, immediate `ValidationError` at the boundary rather than silent failures inside agents.

**3. Why SQLite over Redis/Postgres?**
Session memory needs last 10 messages per user. SQLite is zero-infrastructure, async-capable via aiosqlite, and sufficient for prototype scale. The trade-off is ephemeral data on HF (wiped on container restart). Postgres is the production upgrade path with minimal code change.

**4. Why Nginx on HF Spaces?**
HF Docker Spaces expose exactly one port (7860). Nginx routes `/whatsapp/*` to FastAPI, `/mcp/*` to the MCP server, and `/*` including WebSocket to Streamlit — all in the same container. A Python httpx proxy blocks WebSocket upgrade headers (produces 403); Nginx handles them natively.

**5. Why MCP for the therapist directory?**
Decouples data source from agent code. TherapistAgent calls the MCP server via HTTP — it doesn't know or care whether data comes from Google Maps, Practo, or a hospital database. The backend is swappable without touching agent code.

**6. Why is `crewai==1.14.4` pinned?**
CrewAI 1.14.5+ adds a `cache_breakpoint` property to system messages. Groq does not support this property and rejects every request with a 400 error. Version 1.14.4 is the last version confirmed working with Groq's API.

**7. Why is `THERAPIST_MCP_URL` hardcoded as `127.0.0.1:8001`?**
The MCP server runs inside the same Docker container as the main app. Localhost communication is faster (no network hop), requires no HTTPS certificate, and needs no separate deployment. The public `/mcp/*` Nginx route exists for external callers; internal tool calls go direct.

---

## 🔮 Future Scope & Known Limitations

### Current Limitations (Prototype vs Production)

| Limitation | Impact | Root Cause |
|---|---|---|
| SQLite data is ephemeral | Users lose conversation history on every HF restart | SQLite lives in container filesystem, wiped on restart |
| Web user identity resets per tab | No memory continuity for web users across sessions | UUID generated fresh per `st.session_state`, no persistent login |
| CrewAI agent LLM calls not nested in LangSmith | LLM Calls monitoring tab is empty | CrewAI 1.14.4 resets LiteLLM callbacks, blocking LangSmith nesting |
| MCP server restarts intermittently | Occasional fallback to direct Google Maps | HF container orchestration kills subprocesses under memory pressure |
| No Twilio webhook signature validation | Spoofed webhook requests possible | Skipped for speed; critical for real deployment |
| WhatsApp is Twilio sandbox | Real users cannot message without joining sandbox | Free Twilio account limitation |
| Groq free tier rate limits | Vision/speech may get 429 errors under load | Model fallback list helps but not eliminated |

---

### Roadmap — Production Upgrades

**1. Persistent Database → PostgreSQL / Supabase**
Replace SQLite with a hosted PostgreSQL database (Railway, Supabase, Neon). Session memory survives container restarts. Users retain full conversation history indefinitely. Code change is minimal — `aiosqlite` and `asyncpg` have nearly identical interfaces.

**2. User Authentication → Login System**
Add a lightweight login so web users get a persistent, consistent `user_id` tied to their account rather than a new UUID per browser tab. Options: Google OAuth via `streamlit-google-auth`, or a simple email + magic link system. WhatsApp already has natural identity via phone number.

**3. RAG over Medical Knowledge Base**
Add a vector database (Qdrant, Pinecone) loaded with verified medical sources — DSM-5 criteria, drug interaction databases, symptom-to-condition mappings. Agents would retrieve relevant documents before responding, grounding answers in verified data rather than purely Llama's training data.

**4. Doctor Search for MEDICAL Intent**
The MCP server already accepts a `specialty` parameter. Add a `FindDoctorsTool` in `maps_tool.py` and attach it to `DoctorAgent`. Patients asking about specific medical conditions could receive listings for relevant specialists (cardiologist, dermatologist, neurologist) without any MCP server changes.

**5. Upgrade CrewAI → Native LangSmith LLM Tracing**
Once Groq resolves the `cache_breakpoint` compatibility issue (or a newer CrewAI version fixes it), upgrading to CrewAI 1.15+ would restore full parent-child LLM call nesting in LangSmith. The LLM Calls and Cost & Tokens monitoring tabs would then populate automatically.

**6. Fine-Tuning Pipeline**
LangSmith is already collecting every conversation as the `safespace_intent_classification` dataset. Once sufficient data accumulates, fine-tune a smaller model (Llama 3.1 8B) specifically on health-domain conversations. A fine-tuned 8B model could outperform the general 70B at a fraction of the inference cost.

**7. Multi-Language Support**
Groq Whisper large-v3 supports 50+ languages including Hindi, Marathi, Bengali, Tamil, and Telugu. Add a language detection step in `engine.py` and inject a language instruction into agent task descriptions. The infrastructure already supports it — just needs the detection + routing logic.

**8. WhatsApp Business API → Full Production**
Migrate from Twilio Sandbox to a WhatsApp Business Account. Removes the requirement for users to join a sandbox. Enables proactive messaging, message templates, and higher throughput limits.

**9. Horizontal Scaling**
Replace SQLite with PostgreSQL (item 1), add a message queue (Redis + Celery or AWS SQS) so the webhook returns 200 OK immediately and processes LLM responses asynchronously, then deploy multiple FastAPI/Streamlit instances behind a load balancer. The adapter pattern means both channels use the same brain with no refactoring needed.

**10. Twilio Webhook Signature Validation**
Add `X-Twilio-Signature` header verification in `webhook.py` using Twilio's HMAC-SHA1 algorithm. Prevents spoofed webhook requests from external actors. One-line change using `twilio.request_validator.RequestValidator`.

**11. User Feedback Loop → Model Improvement**
Add thumbs up / thumbs down buttons in the Streamlit UI that call `langsmith_client.create_feedback(run_id, score)`. Over time, builds a labeled dataset of good and bad responses. Enables prompt optimization and identifies which intents or message types produce weak responses.

**12. Voice Output → Text-to-Speech Responses**
For WhatsApp voice users, convert SafeSpace responses to audio using a TTS API (ElevenLabs, Google TTS) and send back as voice notes via Twilio's media messaging. Creates a fully voice-native experience for accessibility.

---

## 📸 Demo — What It Can Do

| Capability | Example |
|---|---|
| Medical symptom advice | "My left side of throat feels sore and heavy" → home remedies + warning signs |
| Blood report reading | Upload CBC report image → explains Hemoglobin levels, suggests doctor visit |
| X-ray interpretation | Upload child's chest X-ray → notes mottled lung appearance, recommends monitoring |
| ECG reading | Upload ECG strip → identifies leads, flags possible abnormal rhythms |
| Skin condition | Upload photo of rash → describes appearance, home care, when to seek help |
| Voice note | Send audio symptom description → Whisper transcribes → full medical guidance |
| Crisis detection | "I feel like killing myself" → emergency Twilio call placed + helpline numbers sent |
| Therapist finder | "find therapists near Gondia Maharashtra" → 5 real clinics with ratings and hours |
| CBT support | "I'm feeling low since a week" → validation + mindful breathing exercise |
| Memory | References previous messages in same conversation across WhatsApp sessions |

---

## 👩‍💻 Author

**Hiteshi Aglawe**
Aspiring GenAI / ML Engineer
[GitHub](https://github.com/HiteshiAglawe0505) · [HuggingFace](https://huggingface.co/HiteshiAglawe0505)

---

## ⚠️ Disclaimer

SafeSpace AI is a **prototype research assistant** — not a substitute for professional medical or mental health care. Always consult a qualified doctor or licensed therapist for medical decisions. The emergency call feature is a demonstration of crisis escalation architecture and should not be relied upon as a sole safety mechanism.
