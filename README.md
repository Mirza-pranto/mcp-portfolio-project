# ⚡ Omni‑Support AI – Intelligent IT Support Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-00BFFF?logo=groq)](https://groq.com)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP-7B2FBE)](https://github.com/jlowin/fastmcp)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3FCF8E?logo=supabase)](https://supabase.com)

**Omni‑Support AI** is an end‑to‑end IT support platform that combines a conversational AI agent with a full‑featured admin dashboard. It leverages **Groq’s ultra‑fast LLMs**, **Supabase** for data persistence and authentication, and a **Model Context Protocol (MCP)** server to orchestrate tools like ticket management, knowledge base search, and web search.

---

## ✨ Features

### 🔐 Authentication & Role‑Based Access
- **Supabase Auth** – secure email/password login and sign‑up.
- **Dual‑role system** – *Users* get a chat interface, *Agents* get a full administrative dashboard.
- Session persistence across page reloads.

### 💬 Multi‑Threaded Chat Sessions
- Sidebar with **“New Chat”**, chat history list, and **“Delete Chat”**.
- Each conversation is stored as a separate session (auto‑titled from the first user message).
- Seamless switching between threads – all messages persist in Supabase.

### 🤖 AI‑Powered Support Chat
- **Vision‑capable** – users can upload screenshots (error logs, UI issues) and the model analyzes them.
- **Voice input** – record audio; transcribed on‑the‑fly with Groq’s Whisper.
- **Multi‑turn tool calling** – the agent decides when to call tools (search KB, create ticket, web search, etc.).
- **Text‑to‑Speech** – AI responses can be read aloud using gTTS (optional).

### 🛠️ Actionable Agent Dashboard
- **Ticket Queue** – view all tickets with metrics (total, new, open).
- **Manage Tickets** – update status, add internal notes, and resolve issues.
- **Proactive KB Generation** – when a ticket is marked *Resolved*, automatically draft a knowledge base article from the issue + resolution notes using Groq’s LLM.

### 📚 Hybrid Knowledge Base
- **Vector + Full‑Text Search** – powered by Supabase’s `pgvector` and PostgreSQL full‑text search (RPC `hybrid_search_kb`).
- Upload PDF/TXT manuals; they are chunked, embedded, and stored for semantic retrieval.
- Manage (delete) articles directly from the dashboard.

### 🔌 MCP Tool Integrations
Built‑in tools (served by `server.py`):
- `get_ticket_status`, `create_ticket`, `get_user_tickets`, `get_recent_tickets`
- `search_knowledge_base` (hybrid)
- `web_search` (DuckDuckGo)

---

## 🧰 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | [Streamlit](https://streamlit.io) |
| **LLM / Vision** | [Groq](https://groq.com) (Llama 3.2 Vision, llama-3.3‑70b, Whisper‑large‑v3‑turbo) |
| **MCP Framework** | [FastMCP](https://github.com/jlowin/fastmcp) |
| **Database & Auth** | [Supabase](https://supabase.com) (PostgreSQL + pgvector) |
| **Embeddings** | [sentence‑transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`) |
| **TTS** | [gTTS](https://gtts.readthedocs.io) |
| **Web Search** | [duckduckgo‑search](https://pypi.org/project/duckduckgo-search/) |
| **PDF parsing** | [PyPDF](https://pypi.org/project/pypdf/) |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- A [Supabase](https://supabase.com) project (with `pgvector` enabled)
- A [Groq API key](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/omni-support-ai.git
cd omni-support-ai