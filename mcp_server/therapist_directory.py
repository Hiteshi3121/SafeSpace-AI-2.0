"""
mcp_server/therapist_directory.py

SafeSpace MCP Server — Therapist Directory

WHAT IS THIS FILE?
==================
This is a standalone server that exposes therapist-search as MCP tools.

MCP (Model Context Protocol) is a standard invented by Anthropic so that
AI agents can discover and call tools in a standardised way — like a USB
standard for AI tools.

BEFORE MCP (what we had):
  maps_tool.py was hardwired INSIDE the SafeSpace app.
  Only SafeSpace could use it.

AFTER MCP (what we have now):
  This server runs separately.
  ANY agent — CrewAI, LangChain, Claude Desktop, even a future
  mobile app — can call these tools via the same standard protocol.

HOW THE FLOW WORKS:
===================
User asks → TherapistAgent decides to find therapists
    → calls find_nearby_therapists tool (maps_tool.py)
        → maps_tool.py calls THIS MCP server via HTTP
            → MCP server geocodes 
                → returns formatted results back up the chain

RUNNING THIS SERVER:
====================
  python mcp_server/therapist_directory.py
  → Starts on http://localhost:8001

Then add to your .env:
  THERAPIST_MCP_URL=http://localhost:8001

The maps_tool.py will automatically route through MCP when this is set.

TOOLS EXPOSED:
==============
1. search_by_location   — main search, takes city + optional specialty
2. get_therapist_details — get full info for one result by index
3. list_specialties     — list what mental health specialties we search for
"""

import logging
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | MCP-Server | %(message)s"
)
logger = logging.getLogger(__name__)

# ── FastAPI app (this IS the MCP server) ─────────────────────────────────────
app = FastAPI(
    title="SafeSpace Healthcare Directory MCP Server",
    description=(
        "MCP-compatible server that exposes therapist search tools "
        "backed by Google Maps Places API. "
        "Requires GOOGLE_MAPS_API_KEY."
    ),
    version="1.0.0",
)

# Google Maps Places API (no rate limits at our scale, real India data)
PLACES_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

# In-memory cache — avoids duplicate API calls within the same session
_cache: dict[str, str] = {}

# Google Maps API key — read at request time so it picks up HF injected env
def _get_api_key() -> str:
    import os
    from core.config import get_settings
    try:
        return get_settings().google_maps_api_key
    except Exception:
        return os.environ.get("GOOGLE_MAPS_API_KEY", "")


# ── Request / Response schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    location: str = Field(..., description="City or area, e.g. 'Nagpur' or 'Mumbai Maharashtra'")
    specialty: str = Field(
        default="therapist psychiatrist counselor",
        description="Mental health specialty to look for"
    )
    radius_km: int = Field(default=5, ge=1, le=20, description="Search radius in km")


class SearchResponse(BaseModel):
    location:       str
    count:          int
    results:        list[dict]
    result:         str          # pre-formatted string for the AI agent to use directly
    data_source:    str = "OpenStreetMap via Overpass API"


class DetailRequest(BaseModel):
    location:   str
    result_index: int = Field(..., description="0-based index from a previous search result")


# ── MCP tool endpoints ────────────────────────────────────────────────────────

@app.get("/tools", summary="List available MCP tools")
def list_tools():
    """
    MCP discovery endpoint.
    Clients call this to learn what tools this server provides.
    This is the MCP equivalent of a restaurant showing you its menu.
    """
    return {
        "tools": [
            {
                "name": "search_by_location",
                "description": (
                    "Search for mental health professionals (therapists, psychiatrists, "
                    "counselors) near a given city or area using OpenStreetMap data."
                ),
                "endpoint": "POST /search",
                "input_schema": {
                    "location": "string — city/area name",
                    "specialty": "string — optional, defaults to general mental health",
                    "radius_km": "int — optional, 1-20, defaults to 5",
                }
            },
            {
                "name": "get_therapist_details",
                "description": "Get detailed info for a specific result from a previous search.",
                "endpoint": "POST /details",
                "input_schema": {
                    "location": "string — same city used in search",
                    "result_index": "int — 0-based index of the result",
                }
            },
            {
                "name": "list_specialties",
                "description": "List all mental health specialties and OSM tags this server searches for.",
                "endpoint": "GET /specialties",
            }
        ]
    }


@app.post("/search", response_model=SearchResponse, summary="Tool: search_by_location")
def search_by_location(req: SearchRequest):
    """
    MCP Tool 1: search_by_location
    ================================
    Queries Google Maps Places API for mental health providers near the location.
    No geocoding step needed — Google Maps handles location resolution internally.
    Returns both structured data (for apps) and a pre-formatted string
    (so AI agents can paste it directly into their response).
    """
    logger.info("MCP search_by_location: location='%s' specialty='%s'", req.location, req.specialty)

    # Check cache first
    cache_key = f"{req.location.lower()}:{req.specialty.lower()}"
    if cache_key in _cache:
        logger.info("Cache hit for '%s'", req.location)
        return SearchResponse(
            location=req.location,
            count=1,
            results=[],
            result=_cache[cache_key],
        )

    # Google Maps Places search — no geocoding needed
    api_key = _get_api_key()
    if not api_key:
        fallback = (
            f"Google Maps API key not configured. "
            f"For therapists near {req.location}, try Practo.com "
            f"or call iCall: 9152987821."
        )
        return SearchResponse(location=req.location, count=0, results=[], result=fallback)

    formatted = _google_maps_search(api_key, req.location)

    # Cache the result
    _cache[cache_key] = formatted

    return SearchResponse(
        location=req.location,
        count=1,
        results=[],
        result=formatted,
    )


@app.post("/details", summary="Tool: get_therapist_details")
def get_therapist_details(req: DetailRequest):
    """
    MCP Tool 2: get_therapist_details
    ===================================
    Returns full details for one result from a previous search.
    Useful when the user asks "tell me more about the first one".
    """
    cache_key_prefix = f"{req.location}"
    # Find matching cached results
    matching = None
    for key, results in _cache.items():
        if results:
            matching = results
            break

    if not matching or req.result_index >= len(matching):
        return {"error": f"No cached results for '{req.location}'. Please search first."}

    p = matching[req.result_index]
    return {
        "name":    p.get("name"),
        "address": p.get("address"),
        "phone":   p.get("phone") or "Not listed",
        "website": p.get("website") or "Not listed",
        "coordinates": {
            "lat": p.get("lat"),
            "lon": p.get("lon"),
        },
        "maps_link": (
            f"https://www.openstreetmap.org/?mlat={p.get('lat')}&mlon={p.get('lon')}"
            if p.get("lat") else None
        )
    }


@app.get("/specialties", summary="Tool: list_specialties")
def list_specialties():
    """
    MCP Tool 3: list_specialties
    ==============================
    Returns all OSM tags and specialties this server searches for.
    Useful for showing the user what kinds of professionals are included.
    """
    return {
        "specialties_searched": [
            "Psychotherapist (healthcare=psychotherapist)",
            "Psychiatrist (healthcare=psychiatrist)",
            "Clinical Psychologist (healthcare:speciality=psychology)",
            "Mental Health Clinic (healthcare:speciality=mental_health)",
            "General Doctor / Counselor (name contains 'mental|counsel|therap')",
        ],
        "data_source": "OpenStreetMap via Overpass API",
        "coverage": "Global — data quality varies by region",
        "india_helplines": {
            "iCall":               "9152987821",
            "Vandrevala Foundation": "1860-2662-345",
            "NIMHANS":             "080-46110007",
            "Snehi":               "044-24640050",
        }
    }


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "server": "SafeSpace Healthcare Directory MCP", "version": "1.0.0"}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _google_maps_search(api_key: str, location: str) -> str:
    """
    Calls Google Maps Places API textsearch endpoint.
    Returns a formatted string ready for the AI agent to use.
    Same logic as maps_tool.py — centralised here in the MCP server.
    """
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
            logger.error("Google Maps REQUEST_DENIED: %s", data.get("error_message"))
            return _fallback(location)
        if status == "ZERO_RESULTS":
            logger.info("Google Maps ZERO_RESULTS for '%s'", location)
            return _fallback(location)
        if status != "OK":
            logger.warning("Google Maps unexpected status '%s' for '%s'", status, location)
            return _fallback(location)

        places = data.get("results", [])[:5]
        if not places:
            return _fallback(location)

        lines = [f"Mental health professionals near {location}:\n"]
        for i, p in enumerate(places, 1):
            name     = p.get("name", "Unknown")
            address  = p.get("formatted_address", "Address not available")
            rating   = p.get("rating")
            open_now = p.get("opening_hours", {}).get("open_now")

            entry = f"{i}. {name}\n   📍 {address}"
            if rating:
                entry += f"\n   ⭐ {rating}/5"
            if open_now is True:
                entry += "  🟢 Open now"
            elif open_now is False:
                entry += "  🔴 Closed now"
            lines.append(entry)

        lines.append(
            "\n💡 For appointments and reviews visit Practo.com\n"
            "📞 iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345"
        )
        logger.info("Google Maps returned %d results for '%s'", len(places), location)
        return "\n\n".join(lines)

    except requests.exceptions.Timeout:
        logger.warning("Google Maps timeout for '%s'", location)
        return _fallback(location)
    except Exception as e:
        logger.warning("Google Maps error for '%s': %s", location, e)
        return _fallback(location)


def _fallback(location: str) -> str:
    return (
        f"I couldn't find listings near {location} right now. "
        "Please try:\n"
        "   • Practo.com → search 'psychiatrist' or 'psychologist'\n"
        "   • 1mg.com → Mental Health section\n\n"
        "📞 Free helplines:\n"
        "   • iCall: 9152987821 (Mon–Sat 8am–10pm)\n"
        "   • Vandrevala Foundation: 1860-2662-345 (24/7)"
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SafeSpace Therapist Directory MCP Server")
    print("  Running on: http://localhost:8001")
    print("  Tools docs: http://localhost:8001/docs")
    print("  Add to .env: THERAPIST_MCP_URL=http://localhost:8001")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")