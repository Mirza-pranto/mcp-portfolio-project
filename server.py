# server.py
import os
from fastmcp import FastMCP
from mcp.server.fastmcp import Context
from supabase import create_client
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer

# Initialize the MCP Server
mcp = FastMCP("SupportTicketingSystem")

raw_url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
if raw_url.endswith("/rest/v1"):
    raw_url = raw_url[:-8].rstrip("/")

SUPABASE_URL = raw_url if raw_url else None
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None


# Load the embedding model (downloads automatically the first time)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Searches the internal IT knowledge base for guides, policies, and troubleshooting steps."""
    try:
        # Convert the text query into a vector embedding
        query_embedding = embedder.encode(query).tolist()
        
        # Call the Supabase matching function
        response = supabase.rpc("match_kb_articles", {
            "query_embedding": query_embedding,
            "match_threshold": 0.3, # Adjust between 0.0 and 1.0 for strictness
            "match_count": 3
        }).execute()
        
        if not response.data:
            return "No relevant internal documentation found."
            
        # Format the results for the LLM
        results = ["Here are the relevant internal documents:\n"]
        for doc in response.data:
            results.append(f"Title: {doc['title']}\nContent: {doc['content']}\nRelevance: {doc['similarity']:.2f}\n---")
            
        return "\n".join(results)
    except Exception as e:
        return f"ERROR searching knowledge base: {str(e)}"

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
def create_ticket(description: str, email: str, ctx: Context | None = None) -> str:
    """Creates a new high-priority support ticket. Requires the user's email."""
    try:
        response = supabase.table("tickets").insert({
            "status": "Open", 
            "description": description,
            "user_email": email  # New column added here
        }).execute()
        
        new_id = response.data[0]["id"]
        return f"Successfully created Ticket #{new_id} for {email}."
    except Exception as e:
        return f"ERROR: Database insertion failed - {str(e)}"
    
    
@mcp.tool
def get_user_tickets(email: str, ctx: Context | None = None) -> str:
    """Retrieves all past and present support tickets for a specific user email."""
    try:
        response = supabase.table("tickets").select("id, status, description, created_at").eq("user_email", email).order("id", desc=True).execute()
        
        if not response.data:
            return f"No previous tickets found for {email}."
            
        # Format the data nicely for the AI to read
        formatted_history = f"Ticket History for {email}:\n"
        for ticket in response.data:
            formatted_history += f"- Ticket #{ticket['id']} ({ticket['status']}): {ticket['description']}\n"
            
        return formatted_history
    except Exception as e:
        return f"ERROR: Failed to fetch history - {str(e)}"
    
    
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


@mcp.tool()
def web_search(query: str, max_results: int = 5, ctx: Context | None = None) -> str:
    """Search the web using DuckDuckGo to find real-time information, technical documentation, or troubleshooting guides."""
    if ctx:
        ctx.info(f"Performing web search for query: {query}")
    try:
        # Use a specific region and a short timeout to keep searches responsive
        ddgs = DDGS(timeout=10)
        results = list(ddgs.text(query, max_results=max_results, region="wt-wt"))
        if not results:
            return "No search results found for that query."

        output = []
        for i, r in enumerate(results, 1):
            title = r.get('title') or r.get('title_no_format') or ''
            href = r.get('href') or r.get('url') or ''
            snippet = r.get('body') or r.get('snippet') or ''
            output.append(f"{i}. {title}\n   URL: {href}\n   Snippet: {snippet}\n")
        return "\n".join(output)
    except Exception as e:
        if ctx:
            ctx.error(f"Web search failed: {str(e)}")
        return f"ERROR: Web search failed - {str(e)}"


if __name__ == "__main__":
    # Runs the server over stdio, the standard way MCP clients communicate with local servers.
    mcp.run()