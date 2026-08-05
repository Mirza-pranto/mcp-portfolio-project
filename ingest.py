import os
from supabase import create_client
from sentence_transformers import SentenceTransformer

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def add_document(title: str, content: str):
    embedding = embedder.encode(content).tolist()
    supabase.table("kb_articles").insert({
        "title": title,
        "content": content,
        "embedding": embedding
    }).execute()
    print(f"Added: {title}")

# Example usage:
add_document(
    "VPN Setup Guide", 
    "To connect to the corporate VPN, open Cisco AnyConnect, enter vpn.company.com, and use your Okta credentials to log in. If error 412 appears, restart the service."
)