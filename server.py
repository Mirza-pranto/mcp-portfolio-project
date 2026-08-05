# server.py
import os
from fastmcp import FastMCP
from mcp.server.fastmcp import Context
from supabase import create_client

# Initialize the MCP Server
mcp = FastMCP("SupportTicketingSystem")

raw_url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
if raw_url.endswith("/rest/v1"):
    raw_url = raw_url[:-8].rstrip("/")

SUPABASE_URL = raw_url if raw_url else None
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Expose Tools via the @mcp.tool decorator
@mcp.tool
def get_ticket_status(ticket_id: int, ctx: Context | None = None) -> str:
    """Retrieves the status of a specific support ticket by its ID."""
    if ctx:
        ctx.info(f"Fetching ticket status for ID {ticket_id}")

    try:
        if supabase is None:
            raise RuntimeError("Supabase client is not configured. Set SUPABASE_URL and SUPABASE_KEY.")

        response = supabase.table("tickets").select("status, description").eq("id", ticket_id).single().execute()
        result = response.data

        if result:
            return f"Ticket {ticket_id} is currently {result['status']}. Description: {result['description']}"
        return f"Ticket {ticket_id} not found."
    except Exception as e:
        if ctx:
            ctx.error(f"Database lookup failed: {str(e)}")
        return f"ERROR: Database lookup failed - {str(e)}"

@mcp.tool
def create_ticket(description: str, ctx: Context | None = None) -> str:
    """Creates a new high-priority support ticket. Use this when the user reports a new critical bug."""
    if ctx:
        ctx.info("Starting ticket creation operation")

    try:
        if supabase is None:
            raise RuntimeError("Supabase client is not configured. Set SUPABASE_URL and SUPABASE_KEY.")

        response = supabase.table("tickets").insert({"status": "Open", "description": description}).execute()
        new_id = response.data[0]["id"]

        if ctx:
            ctx.info(f"Ticket created successfully with ID {new_id}")
        return f"Successfully created Ticket #{new_id}."
    except Exception as e:
        if ctx:
            ctx.error(f"Database insertion failed: {str(e)}")
        return f"ERROR: Database insertion failed - {str(e)}"

@mcp.tool
def get_recent_tickets(limit: int = 5, ctx: Context | None = None) -> str:
    """Retrieves a list of the most recent support tickets to summarize the current queue."""
    if ctx:
        ctx.info(f"Fetching the latest {limit} tickets")

    try:
        if supabase is None:
            raise RuntimeError("Supabase client is not configured. Set SUPABASE_URL and SUPABASE_KEY.")

        response = supabase.table("tickets").select("id", "status", "description").order("id", desc=True).limit(limit).execute()
        rows = response.data

        if not rows:
            return "No tickets found."

        formatted = []
        for row in rows:
            formatted.append(f"Ticket {row['id']} - {row['status']}: {row['description']}")
        return "\n".join(formatted)
    except Exception as e:
        if ctx:
            ctx.error(f"Database query failed: {str(e)}")
        return f"ERROR: Database insertion failed - {str(e)}"

if __name__ == "__main__":
    # Runs the server over stdio, the standard way MCP clients communicate with local servers.
    mcp.run()