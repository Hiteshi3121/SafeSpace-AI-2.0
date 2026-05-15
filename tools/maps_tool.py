"""
tools/maps_tool.py

CrewAI Tool that finds nearby therapists/psychiatrists.

DATA SOURCE — OpenStreetMap via Overpass API (FREE, no API key needed)
======================================================================
We replaced Google Maps Places API with the Overpass API which queries
OpenStreetMap data. 100% free, no billing, no API key required.

HOW IT WORKS (2 steps):
  1. Geocode the city name → lat/lon using Nominatim (OSM's free geocoder)
     e.g. "Nagpur" → (21.1458, 79.0882)

  2. Query Overpass for healthcare nodes within 5km radius:
     - amenity=doctors with healthcare:speciality=psychiatry/psychology
     - healthcare=psychotherapist
     - amenity=clinic (general mental health clinics)

WHY THIS IS STILL A CrewAI TOOL (not hardcoded):
==================================================
The TherapistAgent's LLM brain decides WHEN to call this tool.
It reasons: "User wants a therapist near Nagpur → call find_nearby_therapists"
We just changed the underlying data source — the agent behavior is identical.

MCP BRIDGE:
===========
When the MCP server (mcp_server/therapist_directory.py) is running,
this tool calls it via HTTP instead of Overpass directly.
Same CrewAI tool interface, swappable backend.
"""

import logging
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Nominatim geocoder — free OSM service, requires a User-Agent header
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
HEADERS       = {"User-Agent": "SafeSpace-AI/2.0 (mental health assistant)"}

# Radius in metres to search around the geocoded location
SEARCH_RADIUS_M = 5000


class TherapistSearchInput(BaseModel):
    """Input schema for the therapist finder tool."""
    location: str = Field(
        ...,
        description="City or area to search in, e.g. 'Nagpur', 'Mumbai Maharashtra'"
    )
    specialty: str = Field(
        default="therapist psychiatrist counselor",
        description="Type of mental health professional to search for"
    )


class FindTherapistsTool(BaseTool):
    """
    Finds nearby therapists, psychiatrists, and counselors using
    OpenStreetMap (Overpass API) — completely free, no API key needed.
    Use this when the user asks for professional mental health support,
    therapy recommendations, or help finding a doctor near them.
    """
    name: str = "find_nearby_therapists"
    description: str = (
        "Searches OpenStreetMap for nearby therapists, psychiatrists, and counselors. "
        "Use when the user explicitly asks for professional help, therapy recommendations, "
        "or when you determine they need in-person professional support. "
        "Input: location (city/area) and optional specialty type."
    )
    args_schema: type[BaseModel] = TherapistSearchInput

    def _run(self, location: str, specialty: str = "therapist psychiatrist counselor") -> str:
        """
        Two-step search:
          1. Geocode location name → coordinates via Nominatim
          2. Query Overpass for mental health professionals nearby
        Falls back to MCP server if THERAPIST_MCP_URL is configured.
        """

        # ── MCP bridge: if MCP server is running, delegate to it ──────────────
        mcp_url = getattr(settings, "therapist_mcp_url", "")
        if mcp_url:
            return _call_mcp_server(mcp_url, location, specialty)

        # ── Direct Overpass path ───────────────────────────────────────────────
        logger.info("Therapist search via Overpass for location='%s'", location)

        lat, lon = _geocode(location)
        if lat is None:
            return (
                f"I couldn't find the location '{location}' on the map. "
                "Please try a nearby major city name (e.g. 'Nagpur', 'Mumbai')."
            )

        results = _overpass_search(lat, lon)

        if not results:
            return (
                f"No mental health professionals found on OpenStreetMap near {location}. "
                "OpenStreetMap data can be sparse in some areas. "
                "You can also try: Practo.com, 1mg.com, or search 'therapist near me' on Google Maps. "
                "iCall helpline: 9152987821 | Vandrevala Foundation: 1860-2662-345"
            )

        lines = [f"Mental health professionals near {location} (via OpenStreetMap):\n"]
        for i, place in enumerate(results[:5], 1):
            name    = place.get("name", "Unnamed Clinic")
            addr    = place.get("address", "Address not listed on map")
            phone   = place.get("phone", "")
            website = place.get("website", "")

            entry = f"{i}. {name}\n   📍 {addr}"
            if phone:
                entry += f"\n   📞 {phone}"
            if website:
                entry += f"\n   🌐 {website}"
            lines.append(entry)

        lines.append(
            "\n💡 Results from OpenStreetMap. For more options try Practo.com"
        )
        logger.info("Overpass returned %d results for '%s'", len(results), location)
        return "\n\n".join(lines)


# ── Helper functions ──────────────────────────────────────────────────────────

def _geocode(location: str) -> tuple[float | None, float | None]:
    """Convert a city/area name to (lat, lon) using Nominatim."""
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": location, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.warning("Nominatim geocoding failed for '%s': %s", location, e)
    return None, None


def _overpass_search(lat: float, lon: float) -> list[dict]:
    """
    Query Overpass for mental health providers within SEARCH_RADIUS_M.
    We cast a wide net with multiple OSM tag combinations.
    """
    query = f"""
    [out:json][timeout:25];
    (
      node["healthcare"="psychotherapist"](around:{SEARCH_RADIUS_M},{lat},{lon});
      node["healthcare"="psychiatrist"](around:{SEARCH_RADIUS_M},{lat},{lon});
      node["amenity"="doctors"]["healthcare:speciality"~"psychiatry|psychology|mental_health",i](around:{SEARCH_RADIUS_M},{lat},{lon});
      node["amenity"="clinic"]["healthcare:speciality"~"psychiatry|psychology|mental_health",i](around:{SEARCH_RADIUS_M},{lat},{lon});
      node["amenity"="doctors"]["name"~"psychiatr|psycholog|counsel|therap|mental",i](around:{SEARCH_RADIUS_M},{lat},{lon});
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
    """Extract clean fields from an Overpass node."""
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


def _call_mcp_server(mcp_url: str, location: str, specialty: str) -> str:
    """
    Delegate the search to the MCP server when it's running.
    The MCP server calls Overpass internally — same data, clean architecture.
    Falls back to direct Overpass if MCP server is unreachable.
    """
    try:
        resp = requests.post(
            f"{mcp_url.rstrip('/')}/search",
            json={"location": location, "specialty": specialty},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("result", "No results from MCP server.")
    except Exception as e:
        logger.warning("MCP server call failed (%s), falling back to direct Overpass: %s", mcp_url, e)
        lat, lon = _geocode(location)
        if lat is None:
            return f"Could not find location '{location}'."
        results = _overpass_search(lat, lon)
        if not results:
            return f"No therapists found near {location}. Try Practo.com or search Google Maps."
        lines = [f"Mental health professionals near {location}:\n"]
        for i, p in enumerate(results[:5], 1):
            lines.append(f"{i}. {p['name']}\n   📍 {p['address']}")
        return "\n\n".join(lines)


# Singleton instance — imported directly by TherapistAgent


# ══════════════════════════════════════════════════════════════════════════════
# FindDoctorsTool — same MCP server, medical specialties
# Used by DoctorAgent when user asks for nearby doctors/specialists
# ══════════════════════════════════════════════════════════════════════════════

class DoctorSearchInput(BaseModel):
    location: str = Field(
        ...,
        description="City or area to search in, e.g. 'Nagpur', 'Mumbai'"
    )
    specialty: str = Field(
        default="doctor physician specialist clinic",
        description="Type of doctor to search for, e.g. 'cardiologist', 'dermatologist'"
    )


class FindDoctorsTool(BaseTool):
    """
    Finds nearby doctors, physicians, and medical specialists using Google Maps.
    Use this when the user asks for nearby doctors or specialists after
    receiving medical guidance. Ask for their city first if not provided.
    """
    name: str = "find_nearby_doctors"
    description: str = (
        "Searches Google Maps for nearby doctors, physicians, and specialists. "
        "Use ONLY when the user explicitly says yes to finding a doctor and provides a location. "
        "Input: location (city name) and specialty based on their health concern."
    )
    args_schema: type[BaseModel] = DoctorSearchInput

    def _run(self, location: str, specialty: str = "doctor physician clinic") -> str:

        # Route through MCP server if configured
        mcp_url = getattr(settings, "therapist_mcp_url", "")
        if mcp_url:
            result = _try_mcp(mcp_url, location, specialty)
            if result:
                return result

        # Direct Google Maps search
        api_key = getattr(settings, "google_maps_api_key", "")
        if not api_key:
            return _no_api_fallback(location)

        # Build a specific query based on specialty
        query = f"{specialty} doctor hospital clinic in {location}"
        logger.info("Doctor search via Google Maps: '%s' near '%s'", specialty, location)
        return _google_maps_search(api_key, location, specialty)


# Singleton instances
find_therapists_tool = FindTherapistsTool()
find_doctors_tool    = FindDoctorsTool()