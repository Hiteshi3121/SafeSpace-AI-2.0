"""
tools/maps_tool.py — Therapist finder using Google Maps Places API

DATA SOURCE: Google Maps Places API (real clinic data for India)
SETUP: Add GOOGLE_MAPS_API_KEY to .env and enable Places API in Google Cloud Console
"""

import logging
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from core.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

PLACES_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"


class TherapistSearchInput(BaseModel):
    location: str = Field(..., description="City or area, e.g. 'Nagpur', 'Wadi Nagpur'")
    specialty: str = Field(default="therapist psychiatrist counselor")


class FindTherapistsTool(BaseTool):
    """
    Finds nearby therapists, psychiatrists, and counselors using Google Maps.
    Use when user asks for professional mental health support near a location.
    """
    name: str = "find_nearby_therapists"
    description: str = (
        "Searches Google Maps for nearby therapists, psychiatrists, and counselors. "
        "Use when user asks for professional help or mentions a city and wants a therapist. "
        "Input: location (city/area name)."
    )
    args_schema: type[BaseModel] = TherapistSearchInput

    def _run(self, location: str, specialty: str = "therapist psychiatrist counselor") -> str:
        # MCP bridge
        mcp_url = getattr(settings, "therapist_mcp_url", "")
        if mcp_url:
            result = _try_mcp(mcp_url, location, specialty)
            if result:
                return result

        # Google Maps
        api_key = getattr(settings, "google_maps_api_key", "")
        if not api_key:
            return _fallback(location)

        return _google_maps_search(api_key, location)


def _google_maps_search(api_key: str, location: str) -> str:
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
            return _fallback(location)
        if status != "OK":
            logger.warning("Google Maps status: %s", status)
            return _fallback(location)

        places = data.get("results", [])[:5]
        if not places:
            return _fallback(location)

        lines = [f"Mental health professionals near {location}:\n"]
        for i, p in enumerate(places, 1):
            name    = p.get("name", "Unknown")
            address = p.get("formatted_address", "Address not available")
            rating  = p.get("rating")
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


def _try_mcp(mcp_url: str, location: str, specialty: str) -> str | None:
    try:
        resp = requests.post(
            f"{mcp_url.rstrip('/')}/search",
            json={"location": location, "specialty": specialty},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("result")
    except Exception as e:
        logger.warning("MCP server unreachable: %s", e)
        return None


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


# Singleton
find_therapists_tool = FindTherapistsTool()