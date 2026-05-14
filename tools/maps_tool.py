"""
tools/maps_tool.py

Finds nearby therapists/psychiatrists using Google Maps Places API.

WHY GOOGLE MAPS (not Overpass/OpenStreetMap):
=============================================
OpenStreetMap has almost no mental health clinic data for Indian cities.
Overpass queries return 0 results for Nagpur, Mumbai, Pune etc.
Google Maps has real, verified clinic listings for India — it's the
right tool for this use case.

SETUP (one-time, 5 minutes):
=============================
1. Go to: console.cloud.google.com
2. Select your project (or create one)
3. APIs & Services → Library → search "Places API" → Enable it
   ⚠️  You need "Places API" — NOT "Maps JavaScript API"
4. APIs & Services → Credentials → copy your API key
5. Add to .env:  GOOGLE_MAPS_API_KEY=your_key_here

FREE TIER:
  Google gives $200 free credit/month.
  Places Text Search costs $0.032/call → you get ~6,000 free searches/month.
  No charges unless you exceed $200/month.
  Add billing to activate the free credit:
  console.cloud.google.com → Billing → Link a billing account

MCP BRIDGE:
===========
When THERAPIST_MCP_URL is set in .env and the MCP server is running,
this tool routes through the MCP server instead of calling Google Maps directly.
Leave THERAPIST_MCP_URL empty to call Google Maps directly (simpler for local dev).
"""

import logging
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

PLACES_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL     = "https://maps.googleapis.com/maps/api/place/details/json"


class TherapistSearchInput(BaseModel):
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
    Finds nearby therapists, psychiatrists, and counselors using Google Maps.
    Use this when the user asks for professional mental health support,
    therapy recommendations, or mentions a city and wants a therapist there.
    """
    name: str = "find_nearby_therapists"
    description: str = (
        "Searches Google Maps for nearby therapists, psychiatrists, and counselors. "
        "Use when the user explicitly asks for professional help, therapy recommendations, "
        "or mentions a specific location/city and wants a therapist there. "
        "Input: location (city/area name) and optional specialty type."
    )
    args_schema: type[BaseModel] = TherapistSearchInput

    def _run(self, location: str, specialty: str = "therapist psychiatrist counselor") -> str:

        # ── Route through MCP server if configured ─────────────────────────
        mcp_url = getattr(settings, "therapist_mcp_url", "")
        if mcp_url:
            result = _try_mcp(mcp_url, location, specialty)
            if result:
                return result
            # MCP unreachable — fall through to Google Maps directly

        # ── Google Maps Places API ─────────────────────────────────────────
        api_key = getattr(settings, "google_maps_api_key", "")
        if not api_key:
            logger.warning("GOOGLE_MAPS_API_KEY not set in .env")
            return _no_api_fallback(location)

        logger.info("Therapist search via Google Maps for location='%s'", location)
        return _google_maps_search(api_key, location, specialty)


# ── Google Maps helper ────────────────────────────────────────────────────────

def _google_maps_search(api_key: str, location: str, specialty: str) -> str:
    """
    Calls Google Maps Places Text Search API.
    Query: "psychiatrist OR psychologist OR therapist in <city>"
    Returns formatted results string for the AI agent to use directly.
    """
    query = f"psychiatrist psychologist therapist mental health clinic in {location}"

    try:
        resp = requests.get(
            PLACES_TEXT_SEARCH_URL,
            params={
                "query": query,
                "key":   api_key,
                "language": "en",
            },
            timeout=10,
        )
        data   = resp.json()
        status = data.get("status")

        # ── Handle API errors with clear messages ──────────────────────────
        if status == "REQUEST_DENIED":
            error_msg = data.get("error_message", "")
            logger.error("Google Maps REQUEST_DENIED: %s", error_msg)
            # This is the most common setup mistake — log clearly
            return (
                "⚠️ Google Maps API not configured correctly. "
                "The Places API needs to be enabled in Google Cloud Console. "
                f"Error: {error_msg}\n\n"
                + _helpline_fallback(location)
            )

        if status == "OVER_DAILY_LIMIT" or status == "OVER_QUERY_LIMIT":
            logger.error("Google Maps quota exceeded")
            return (
                "Google Maps quota reached for today. "
                "Here are alternative ways to find help:\n\n"
                + _helpline_fallback(location)
            )

        if status == "INVALID_REQUEST":
            logger.error("Google Maps INVALID_REQUEST for location='%s'", location)
            return f"I couldn't search for that location. Try a city name like 'Nagpur' or 'Mumbai'."

        if status == "ZERO_RESULTS":
            return (
                f"No mental health professionals found on Google Maps near {location}. "
                "Try searching directly on Google Maps or Practo.com.\n\n"
                + _helpline_fallback(location)
            )

        if status != "OK":
            logger.warning("Google Maps unexpected status: %s", status)
            return _no_api_fallback(location)

        # ── Format results ─────────────────────────────────────────────────
        places = data.get("results", [])[:5]
        if not places:
            return _no_api_fallback(location)

        lines = [f"Mental health professionals near {location}:\n"]
        for i, place in enumerate(places, 1):
            name    = place.get("name", "Unknown")
            address = place.get("formatted_address", "Address not available")
            rating  = place.get("rating")
            open_now = place.get("opening_hours", {}).get("open_now")

            entry = f"{i}. {name}\n   📍 {address}"
            if rating:
                entry += f"\n   ⭐ {rating}/5"
            if open_now is True:
                entry += "  🟢 Open now"
            elif open_now is False:
                entry += "  🔴 Closed now"
            lines.append(entry)

        lines.append(
            "\n💡 For appointments and verified reviews, visit Practo.com\n"
            "📞 iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345"
        )
        logger.info("Google Maps returned %d results for '%s'", len(places), location)
        return "\n\n".join(lines)

    except requests.exceptions.Timeout:
        logger.warning("Google Maps request timed out for '%s'", location)
        return _no_api_fallback(location)
    except Exception as e:
        logger.warning("Google Maps call failed for '%s': %s", location, e)
        return _no_api_fallback(location)


# ── MCP bridge ────────────────────────────────────────────────────────────────

def _try_mcp(mcp_url: str, location: str, specialty: str) -> str | None:
    """Call MCP server. Returns None if unreachable so caller can fallback."""
    try:
        resp = requests.post(
            f"{mcp_url.rstrip('/')}/search",
            json={"location": location, "specialty": specialty},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("result")
    except Exception as e:
        logger.warning("MCP server unreachable (%s): %s — falling back to Google Maps", mcp_url, e)
        return None


# ── Fallback helpers ──────────────────────────────────────────────────────────

def _helpline_fallback(location: str) -> str:
    return (
        f"Ways to find a therapist near {location}:\n"
        "   • Practo.com → search 'psychiatrist' or 'psychologist'\n"
        "   • 1mg.com → Mental Health section\n"
        "   • iCliniq.com → online + in-person\n\n"
        "📞 Free helplines:\n"
        "   • iCall: 9152987821 (Mon–Sat, 8am–10pm)\n"
        "   • Vandrevala Foundation: 1860-2662-345 (24/7)\n"
        "   • NIMHANS: 080-46110007"
    )


def _no_api_fallback(location: str) -> str:
    return (
        f"I wasn't able to search Google Maps right now. "
        + _helpline_fallback(location)
    )


# Singleton instance — imported directly by TherapistAgent
find_therapists_tool = FindTherapistsTool()