# Multimodal IT Support Agent (MCP Engine)

## Overview
This project is a multimodal IT support assistant that lets users upload error screenshots or describe support issues in plain language. The app uses Groq LLM function calling to route tasks to an isolated FastMCP server, which interacts with a local SQLite database over stdio for ticket lookup and creation.

## Key Features
- Multimodal analysis of screenshots and text-based support requests
- Model Context Protocol (MCP) integration over stdio
- SQLite-backed ticket management
- Streamlit-based user interface

## Architecture Diagram
```text
User UI (Streamlit) -> LLM (Groq) -> MCP Client -> FastMCP Server -> SQLite DB
```

## Setup & Execution
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the MCP server:
   ```bash
   python server.py
   ```

4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes
- Replace the placeholder API key in the app with a valid Groq or Gemini key as needed.
- The SQLite database file will be created locally as `tickets.db`.
