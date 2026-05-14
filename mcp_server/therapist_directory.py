"""
mcp_server/therapist_directory.py

SafeSpace MCP Server — Therapist Directory
Backed by Google Maps Places API (real India clinic data).

WHAT IS THIS?
=============
A standalone FastAPI server that exposes therapist-search as MCP tools.
The main app's maps_tool.py calls this server when THERAPIST_MCP_URL is set.

WHY MCP ARCHITECTURE?
=====================
Without MCP: maps_tool.py → Google Maps API (hardwired inside the app)
With MCP:    maps_tool.py → THIS server → Google Maps API

The TherapistAgent doesn't know or care where data comes from.
You can swap Google Maps for any other source later without touching agent code.
This is the interview talking point: "I decoupled the data source from the agent."

RUNNING:
========
  python mcp_server/therapist_directory.py
  → Starts on http://localhost:8001
  → Swagger docs at http://localhost:8001/docs

Then in .env:
  THERAPIST_MCP_URL=http://localhost:8001

TOOLS EXPOSED (3):
==================
  POST /search          → search_by_location
  POST /details         → get_therapist_details
  GET  /specialties     → list_specialties
  GET  /tools           → MCP discovery (lists available tools)
"""

import logging
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | MCP | %(message)s"
)
logger = logging.getLogger(__name__)

PLACES_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

app = FastAPI(
    title="SafeSpace Therapist Directory MCP Server",
    description="MCP-compatible server exposing therapist search via Google Maps Places API.",
    version="1.0.0",
)

# In-memory cache: location → results list (avoids repeat API calls)
_cache: dict[str, list[dict]] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    location: str = Field(..., description="City or area, e.g. 'Nagpur'")
    specialty: str = Field(default="therapist psychiatrist counselor")


class DetailRequest(BaseModel):
    location: str
    result_index: int = Field(..., description="0-based index from previous search")


# ── MCP Tool endpoints ────────────────────────────────────────────────────────

@app.get("/tools", summary="MCP discovery — lists available tools")
def list_tools():
    """
    MCP clients call this to discover what tools this server provides.
    Like a menu — before ordering, you check what's available.
    """
    return {
        "server": "SafeSpace Therapist Directory",
        "tools": [
            {
                "name": "search_by_location",
                "description": "Search for mental health professionals near a city using Google Maps.",
                "endpoint": "POST /search",
                "inputs": {"location": "string", "specialty": "string (optional)"}
            },
            {
                "name": "get_therapist_details",
                "description": "Get full details for one result from a previous search.",
                "endpoint": "POST /details",
                "inputs": {"location": "string", "result_index": "int (0-based)"}
            },
            {
                "name": "list_specialties",
                "description": "List mental health specialties and available helplines.",
                "endpoint": "GET /specialties",
            }
        ]
    }


@app.post("/search", summary="Tool: search_by_location")
def search_by_location(req: SearchRequest):
    """
    MCP Tool 1: search_by_location
    Calls Google Maps Places API to find mental health professionals.
    Returns both structured data and a pre-formatted string for AI agents.
    """
    logger.info("MCP search: location='%s'", req.location)

    # Load Google Maps key from environment
    import os
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return {
            "location": req.location,
            "count": 0,
            "results": [],
            "result": (
                "GOOGLE_MAPS_API_KEY not set in MCP server environment. "
                "Add it to .env and restart the MCP server."
            )
        }

    # Check cache
    cache_key = req.location.lower().strip()
    if cache_key not in _cache:
        _cache[cache_key] = _google_maps_search(api_key, req.location)

    results = _cache[cache_key]

    if not results:
        formatted = (
            f"No mental health professionals found near {req.location} on Google Maps. "
            "Try Practo.com or iCall: 9152987821."
        )
    else:
        lines = [f"Mental health professionals near {req.location}:\n"]
        for i, p in enumerate(results[:5], 1):
            entry = f"{i}. {p['name']}\n   📍 {p['address']}"
            if p.get("rating"):
                entry += f"\n   ⭐ {p['rating']}/5"
            if p.get("open_now") is True:
                entry += "  🟢 Open now"
            elif p.get("open_now") is False:
                entry += "  🔴 Closed now"
            lines.append(entry)
        lines.append("\n💡 Also try Practo.com | 📞 iCall: 9152987821")
        formatted = "\n\n".join(lines)

    return {
        "location": req.location,
        "count": len(results),
        "results": results[:5],
        "result": formatted,
        "data_source": "Google Maps Places API"
    }


@app.post("/details", summary="Tool: get_therapist_details")
def get_therapist_details(req: DetailRequest):
    """MCP Tool 2: Returns full details for one result from a previous search."""
    cache_key = req.location.lower().strip()
    results = _cache.get(cache_key, [])

    if not results:
        return {"error": f"No cached results for '{req.location}'. Run /search first."}
    if req.result_index >= len(results):
        return {"error": f"Index {req.result_index} out of range. Search returned {len(results)} results."}

    p = results[req.result_index]
    return {
        "name":        p.get("name"),
        "address":     p.get("address"),
        "rating":      p.get("rating", "Not rated"),
        "open_now":    p.get("open_now", "Unknown"),
        "place_id":    p.get("place_id"),
        "google_maps": f"https://www.google.com/maps/place/?q=place_id:{p.get('place_id')}" if p.get("place_id") else None,
    }


@app.get("/specialties", summary="Tool: list_specialties")
def list_specialties():
    """MCP Tool 3: Lists mental health specialties searched and Indian helplines."""
    return {
        "specialties_searched": [
            "Psychiatrist",
            "Psychologist",
            "Therapist / Counselor",
            "Mental Health Clinic",
        ],
        "data_source": "Google Maps Places API",
        "india_helplines": {
            "iCall":                "9152987821 (Mon–Sat 8am–10pm)",
            "Vandrevala Foundation": "1860-2662-345 (24/7)",
            "NIMHANS":              "080-46110007",
            "Snehi":                "044-24640050",
        }
    }


@app.get("/health")
def health():
    import os
    return {
        "status": "ok",
        "server": "SafeSpace Therapist Directory MCP",
        "google_maps_key_set": bool(os.environ.get("GOOGLE_MAPS_API_KEY")),
    }


# ── Google Maps helper ────────────────────────────────────────────────────────

def _google_maps_search(api_key: str, location: str) -> list[dict]:
    query = f"psychiatrist psychologist therapist mental health clinic in {location}"
    try:
        resp = requests.get(
            PLACES_TEXT_SEARCH_URL,
            params={"query": query, "key": api_key, "language": "en"},
            timeout=10,
        )
        data   = resp.json()
        status = data.get("status")

        if status == "REQUEST_DENIED":
            logger.error(
                "Google Maps REQUEST_DENIED: %s\n"
                "Fix: Enable 'Places API' at console.cloud.google.com → APIs & Services → Library",
                data.get("error_message", "")
            )
            return []
        if status not in ("OK", "ZERO_RESULTS"):
            logger.warning("Google Maps status: %s", status)
            return []

        return [
            {
                "name":     p.get("name", "Unknown"),
                "address":  p.get("formatted_address", "Address not available"),
                "rating":   p.get("rating"),
                "open_now": p.get("opening_hours", {}).get("open_now"),
                "place_id": p.get("place_id"),
            }
            for p in data.get("results", [])
        ]
    except Exception as e:
        logger.warning("Google Maps search failed: %s", e)
        return []


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    key_status = "✅ Set" if os.environ.get("GOOGLE_MAPS_API_KEY") else "❌ NOT SET — add GOOGLE_MAPS_API_KEY to .env"
    print(f"""
{'='*60}
  SafeSpace Therapist Directory MCP Server
  Running on: http://localhost:8001
  Swagger docs: http://localhost:8001/docs
  Google Maps key: {key_status}
{'='*60}
""")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")