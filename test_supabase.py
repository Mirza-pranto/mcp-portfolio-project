import os
import toml
from supabase import create_client

# 1. Load secrets manually to eliminate Streamlit/MCP process issues
try:
    secrets = toml.load(".streamlit/secrets.toml")
    url = secrets["SUPABASE_URL"].strip().rstrip("/")
    # Clean up accidental /rest/v1 appends
    if url.endswith("/rest/v1"):
        url = url[:-8].rstrip("/")
    key = secrets["SUPABASE_KEY"].strip()
    print(f"Connecting to URL: {url}")
except Exception as e:
    print(f"❌ Failed to load secrets.toml: {e}")
    exit(1)

# 2. Initialize Client
try:
    supabase = create_client(url, key)
    print("✅ Supabase client initialized.")
except Exception as e:
    print(f"❌ Client initialization failed: {e}")
    exit(1)

# 3. Test Insert
try:
    print("Attempting test insertion into 'tickets' table...")
    response = supabase.table("tickets").insert({"description": "Direct script test ticket", "status": "Open"}).execute()
    print("🎉 SUCCESS! Inserted row:", response.data)
except Exception as e:
    print("❌ DIRECT ERROR:", str(e))