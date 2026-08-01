# server.py
import sqlite3
from fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("SupportTicketingSystem")

def init_db():
    """Sets up a mock database for tickets."""
    conn = sqlite3.connect('tickets.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tickets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT, description TEXT)''')
    # Insert a dummy ticket if empty
    c.execute("SELECT count(*) FROM tickets")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO tickets (status, description) VALUES ('Open', 'Database 502 error reported by user')")
    conn.commit()
    conn.close()

init_db()

# Expose Tools via the @mcp.tool decorator
@mcp.tool
def get_ticket_status(ticket_id: int) -> str:
    """Retrieves the status of a specific support ticket by its ID."""
    conn = sqlite3.connect('tickets.db')
    c = conn.cursor()
    c.execute("SELECT status, description FROM tickets WHERE id=?", (ticket_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return f"Ticket {ticket_id} is currently {result[0]}. Description: {result[1]}"
    return f"Ticket {ticket_id} not found."

@mcp.tool
def create_ticket(description: str) -> str:
    """Creates a new high-priority support ticket. Use this when the user reports a new critical bug."""
    conn = sqlite3.connect('tickets.db')
    c = conn.cursor()
    c.execute("INSERT INTO tickets (status, description) VALUES ('Open', ?)", (description,))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return f"Successfully created Ticket #{new_id}."

if __name__ == "__main__":
    # Runs the server over stdio, the standard way MCP clients communicate with local servers.
    mcp.run()