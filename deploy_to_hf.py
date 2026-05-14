"""
deploy_to_hf.py  —  Run from project root, no GitHub needed.

USAGE:
    python deploy_to_hf.py --username YOUR_HF_USERNAME --token hf_xxxxxx

GET YOUR TOKEN:
    https://huggingface.co/settings/tokens → New token → Role: Write

WHAT IT DOES:
    1. Creates HF Space with docker SDK (correct modern approach for Streamlit)
    2. Copies HF_README.md → README.md temporarily
    3. Uploads project files (skips .env, __pycache__, .db, PDFs, venv)
    4. Restores original README.md
    5. Prints live URL + secret setup instructions
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--token",    required=True)
    parser.add_argument("--space",    default="safespace-ai")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import HfApi

    api     = HfApi(token=args.token)
    repo_id = f"{args.username}/{args.space}"
    root    = Path(__file__).parent

    # ── Step 1: Create Space with docker SDK ──────────────────────────────────
    # NOTE: HF API no longer accepts 'streamlit' as space_sdk.
    # We use 'docker' instead — our Dockerfile runs Streamlit on port 7860.
    # This is the correct modern approach for Streamlit on HF Spaces.
    print(f"\n📦 Creating Space: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )
        print("   ✅ Space created (docker SDK)")
    except Exception as e:
        print(f"   ℹ️  Space may already exist: {e}")
        print("   Continuing with upload...")

    # ── Step 2: Swap README for HF version ────────────────────────────────────
    hf_readme  = root / "HF_README.md"
    readme     = root / "README.md"
    readme_bak = root / "_README_bak.md"

    if readme.exists():
        shutil.copy(readme, readme_bak)
        print("   📝 Backed up README.md")
    if hf_readme.exists():
        shutil.copy(hf_readme, readme)
        print("   📝 Using HF_README.md as README.md for upload")

    # ── Step 3: Upload ────────────────────────────────────────────────────────
    print(f"\n🚀 Uploading to {repo_id} ...")
    print("   Skipping: .env, __pycache__, .db files, PDFs, venv\n")

    try:
        api.upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                "**/__pycache__/**",
                "**/*.pyc",
                "**/*.pyo",
                "**/data/*.db",
                "**/*.pdf",
                "**/.env",                   # NEVER upload secrets
                "**/HF_README.md",
                "**/_README_bak.md",
                "**/deploy_to_hf.py",
                "**/debug_therapist.py",
                "**/.git/**",
                "**/.venv/**",
                "**/venv/**",
                "**/*.egg-info/**",
                "**/node_modules/**",
            ],
            commit_message="Deploy SafeSpace AI 2.0",
        )
        print("\n✅ Upload complete!")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        raise
    finally:
        # Restore original README
        if readme_bak.exists():
            shutil.copy(readme_bak, readme)
            readme_bak.unlink()
            print("   📝 Restored original README.md")

    # ── Step 4: Print next steps ──────────────────────────────────────────────
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"""
{'='*60}
🌿 SafeSpace AI upload complete!

   Space URL:  {url}
   Build logs: {url}/logs
   Secrets:    {url}/settings

REQUIRED — add these secrets NOW at {url}/settings:

   GROQ_API_KEY          your Groq API key
   GOOGLE_MAPS_API_KEY   your Google Maps key

OPTIONAL:
   LANGSMITH_API_KEY     for LangSmith tracing
   LANGSMITH_PROJECT     safespace-ai

The Space will BUILD automatically (takes 3-5 min first time).
Watch progress at: {url}/logs

If build fails, check logs and paste errors here.
{'='*60}
""")


if __name__ == "__main__":
    main()