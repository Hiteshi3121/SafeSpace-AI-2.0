"""
debug_therapist.py — Run from project root

USAGE:
    cd AI_SafeSpace_2.0
    python debug_therapist.py
"""
import sys, os, re
print("\n" + "="*60)
print("  SafeSpace Therapist Finder — Debug Script")
print("="*60)

# TEST 1: Config
print("\n[TEST 1] Config...")
try:
    from core.config import get_settings
    settings = get_settings()
    print(f"   ✅ Config OK")
    print(f"   GROQ key:             {'SET ✅' if settings.groq_api_key else 'MISSING ❌'}")
    print(f"   GOOGLE_MAPS_API_KEY:  {'SET ✅' if settings.google_maps_api_key else 'MISSING ⚠️'}")
    print(f"   THERAPIST_MCP_URL:    '{getattr(settings,'therapist_mcp_url','')}'")
except Exception as e:
    print(f"   ❌ FAILED: {e}"); sys.exit(1)

# TEST 2: Import
print("\n[TEST 2] Importing maps_tool...")
try:
    from tools.maps_tool import find_therapists_tool, _google_maps_search
    print("   ✅ maps_tool imported OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}"); sys.exit(1)

# TEST 3: Google Maps API — returns a formatted string
print("\n[TEST 3] Testing Google Maps Places API...")
if not settings.google_maps_api_key:
    print("   ⏭️  Skipped — GOOGLE_MAPS_API_KEY not set in .env")
else:
    result = _google_maps_search(settings.google_maps_api_key, "Nagpur", "therapist psychiatrist")
    if "REQUEST_DENIED" in result:
        print("   ❌ REQUEST_DENIED — Places API not enabled in Google Cloud Console")
        print("      Fix: console.cloud.google.com → APIs → Enable 'Places API'")
    elif "OVER" in result or "quota" in result.lower():
        print("   ❌ Quota/billing issue — check Google Cloud Console billing")
    elif any(x in result for x in ["1.", "2.", "psychiatr", "psycholog", "therap", "clinic", "mental"]):
        print("   ✅ Google Maps working! Results preview:")
        print("   " + "-"*50)
        for line in result.split("\n")[:12]:
            print(f"   {line}")
        print("   " + "-"*50)
    else:
        print(f"   ⚠️  Unexpected response: {result[:200]}")

# TEST 4: Full tool (bypasses Groq — tests the tool alone)
print("\n[TEST 4] Calling find_therapists_tool._run() directly...")
orig_mcp = os.environ.get("THERAPIST_MCP_URL", "")
os.environ["THERAPIST_MCP_URL"] = ""  # force direct path for this test
try:
    result = find_therapists_tool._run(location="Nagpur", specialty="therapist psychiatrist")
    print("   ✅ Tool ran OK. Response:")
    print("   " + "-"*50)
    for line in result.split("\n"):
        print(f"   {line}")
    print("   " + "-"*50)
except Exception as e:
    print(f"   ❌ Tool FAILED: {e}")
finally:
    os.environ["THERAPIST_MCP_URL"] = orig_mcp

# TEST 5: Groq rate limit
print("\n[TEST 5] Groq API availability...")
try:
    import litellm
    litellm.completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5,
        api_key=settings.groq_api_key,
    )
    print("   ✅ Groq responding — not rate limited")
except Exception as e:
    if "rate_limit" in str(e).lower() or "429" in str(e):
        wait = re.search(r'try again in ([\w.]+)', str(e))
        print(f"   ⏳ Groq RATE LIMITED — wait {wait.group(1) if wait else 'a few minutes'} then retry")
        print("      This is what causes 'trouble processing' in the app. Code is fine.")
    else:
        print(f"   ❌ Groq error: {e}")

# TEST 6: Full crew run
print("\n[TEST 6] Full CrewAI crew run (needs Groq + Google Maps)...")
print("         Takes 15-30 seconds...")
import asyncio

async def test_crew():
    try:
        from core.schemas import UserSession
        from agents.crew import run_crew
        session = UserSession(user_id="debug_test")
        result = await run_crew(
            user_text="find me a therapist near Nagpur",
            session=session,
            user_id="debug_test",
        )
        print(f"   ✅ Crew ran OK!")
        print(f"   Intent:   {result.intent}")
        print(f"   Response preview:")
        for line in result.text.split("\n")[:8]:
            print(f"   {line}")
    except Exception as e:
        if "rate_limit" in str(e).lower():
            print("   ⏳ Groq rate limited — wait and retry. Tool itself works fine (TEST 4 passed).")
        else:
            print(f"   ❌ Crew FAILED: {type(e).__name__}: {e}")

asyncio.run(test_crew())

print("\n" + "="*60)
print("  If TEST 3 and 4 passed → therapist finder is working ✅")
print("  If TEST 6 fails with rate limit → just wait and retry")
print("="*60 + "\n")