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
            → MCP server geocodes + calls Overpass API
                → returns formatted results back up the chain

WHY THIS IS INTERVIEW-WORTHY:
===============================
1. You separated business logic (therapist search) from agent logic
2. The search service is now independently deployable and testable
3. Any future agent or app can use it — you're thinking at scale
4. MCP is Anthropic's open standard — shows you follow the ecosystem

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
        "backed by OpenStreetMap / Overpass API. "
        "No API keys required."
    ),
    version="1.0.0",
)

# OSM endpoints
NOMINATIM_URL   = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL    = "https://overpass-api.de/api/interpreter"
OSM_HEADERS     = {"User-Agent": "SafeSpace-MCP/1.0 (mental health tool)"}
SEARCH_RADIUS_M = 5000

# ── In-memory cache so we don't hammer Overpass on repeated queries ───────────
_cache: dict[str, list[dict]] = {}


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
    Geocodes the city, then queries Overpass for mental health providers.
    Returns both structured data (for apps) and a pre-formatted string
    (so AI agents can paste it directly into their response).
    """
    logger.info("MCP search_by_location: location='%s' specialty='%s'", req.location, req.specialty)

    radius_m = req.radius_km * 1000

    # Geocode
    lat, lon = _geocode(req.location)
    if lat is None:
        no_result_text = (
            f"I couldn't find '{req.location}' on the map. "
            "Please try a major city name like 'Nagpur' or 'Mumbai'."
        )
        return SearchResponse(
            location=req.location, count=0, results=[], result=no_result_text
        )

    # Check cache
    cache_key = f"{lat:.3f},{lon:.3f},{radius_m}"
    if cache_key not in _cache:
        _cache[cache_key] = _overpass_search(lat, lon, radius_m)
    results = _cache[cache_key]

    # Format for agent
    if not results:
        formatted = (
            f"No mental health professionals found near {req.location} on OpenStreetMap. "
            "OSM data can be sparse. Try Practo.com, 1mg.com, or iCall: 9152987821."
        )
    else:
        lines = [f"Mental health professionals near {req.location}:\n"]
        for i, p in enumerate(results[:5], 1):
            entry = f"{i}. {p['name']}\n   📍 {p['address']}"
            if p.get("phone"):
                entry += f"\n   📞 {p['phone']}"
            if p.get("website"):
                entry += f"\n   🌐 {p['website']}"
            lines.append(entry)
        lines.append("\n💡 Data from OpenStreetMap. Also try Practo.com for more options.")
        formatted = "\n\n".join(lines)

    return SearchResponse(
        location=req.location,
        count=len(results),
        results=results[:5],
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


# ── Internal helpers (same logic as maps_tool.py, centralised here) ───────────

def _geocode(location: str):
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": location, "format": "json", "limit": 1},
            headers=OSM_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.warning("Geocoding failed for '%s': %s", location, e)
    return None, None


def _overpass_search(lat: float, lon: float, radius_m: int) -> list[dict]:
    query = f"""
    [out:json][timeout:25];
    (
      node["healthcare"="psychotherapist"](around:{radius_m},{lat},{lon});
      node["healthcare"="psychiatrist"](around:{radius_m},{lat},{lon});
      node["amenity"="doctors"]["healthcare:speciality"~"psychiatry|psychology|mental_health",i](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"]["healthcare:speciality"~"psychiatry|psychology|mental_health",i](around:{radius_m},{lat},{lon});
      node["amenity"="doctors"]["name"~"psychiatr|psycholog|counsel|therap|mental",i](around:{radius_m},{lat},{lon});
    );
    out body 10;
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        return [_parse_element(el) for el in elements if el.get("tags", {}).get("name")]
    except Exception as e:
        logger.warning("Overpass query failed: %s", e)
        return []


def _parse_element(el: dict) -> dict:
    tags = el.get("tags", {})
    addr_parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", ""),
        tags.get("addr:postcode", ""),
    ]
    address = ", ".join(p for p in addr_parts if p) or tags.get("addr:full", "")
    return {
        "name":    tags.get("name", "Unknown"),
        "address": address or "Address not listed",
        "phone":   tags.get("phone") or tags.get("contact:phone", ""),
        "website": tags.get("website") or tags.get("contact:website", ""),
        "lat":     el.get("lat"),
        "lon":     el.get("lon"),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SafeSpace Therapist Directory MCP Server")
    print("  Running on: http://localhost:8001")
    print("  Tools docs: http://localhost:8001/docs")
    print("  Add to .env: THERAPIST_MCP_URL=http://localhost:8001")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")