import os
import asyncio
import json
import base64
import tempfile
import io
import re
from datetime import datetime, timedelta

import streamlit as st
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from groq import Groq
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
from supabase import create_client
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------------
# 1. Environment & Supabase client
# -------------------------------------------------------------------
if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
    os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]
else:
    st.warning("⚠️ Supabase keys missing from st.secrets! Check .streamlit/secrets.toml")

supabase_client = None
if "SUPABASE_URL" in os.environ and "SUPABASE_KEY" in os.environ:
    supabase_client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

env_status = {
    "SUPABASE_URL": "SUPABASE_URL" in os.environ,
    "SUPABASE_KEY": "SUPABASE_KEY" in os.environ,
}

# -------------------------------------------------------------------
# 2. Embedder & Groq
# -------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

try:
    api_key = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError, Exception):
    api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("API Key not found. Please set GROQ_API_KEY in `.streamlit/secrets.toml`")
    st.stop()

client = Groq(api_key=api_key)

st.set_page_config(page_title="Omni-Support AI", layout="centered", page_icon="⚡")
st.title("⚡ IT Support Agent (Groq Edition)")
st.caption("Powered by Llama 3.2 Vision & Model Context Protocol (MCP)")

# -------------------------------------------------------------------
# 3. TTS helpers
# -------------------------------------------------------------------
def clean_text_for_speech(text: str, max_chars: int = 350) -> str:
    text = re.sub(r'```[\s\S]*?```', ' Code snippet omitted. ', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[\*#_~]', '', text)
    text = re.sub(r'\n+', ' ', text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(' ', 1)[0] + "... Please see screen for full details."
    return text

def generate_tts_audio(text: str) -> io.BytesIO | None:
    try:
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            cleaned_text = "Here is the summary of your request."
        tts = gTTS(text=cleaned_text, lang="en", tld="com")
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"⚠️ Voice output unavailable: {str(e)}")
        return None

# -------------------------------------------------------------------
# 4. MCP tool execution
# -------------------------------------------------------------------
async def execute_mcp_tool(tool_name: str, tool_args: dict):
    mcp_env = os.environ.copy()
    transport = PythonStdioTransport("server.py", env=mcp_env)
    mcp_client = Client(transport)
    async with mcp_client:
        result = await mcp_client.call_tool(tool_name, tool_args)
        return result.data

# -------------------------------------------------------------------
# 5. Tool definitions
# -------------------------------------------------------------------
support_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "Retrieves the status of a specific support ticket by its ID.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "integer"}},
                "required": ["ticket_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Creates a new high-priority support ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["description", "email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_tickets",
            "description": "Retrieves a list of the most recent support tickets to summarize the current queue.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 5}}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_tickets",
            "description": "Retrieves all past and present support tickets for a specific user email.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the company's internal IT knowledge base for troubleshooting steps, setup guides, and policies. Always use this BEFORE searching the public web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query based on the user's issue."}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo to find real-time information, technical documentation, or troubleshooting guides.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query string"}},
                "required": ["query"]
            }
        }
    }
]

# -------------------------------------------------------------------
# 6. Session state initialisation
# -------------------------------------------------------------------
if "user_session" not in st.session_state:
    st.session_state.user_session = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_voice_input" not in st.session_state:
    st.session_state.is_voice_input = False

# -------------------------------------------------------------------
# 7. Authentication helper functions
# -------------------------------------------------------------------
def sign_in(email: str, password: str):
    try:
        res = supabase_client.auth.sign_in_with_password({"email": email, "password": password})
        return res
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

def sign_up(email: str, password: str):
    try:
        res = supabase_client.auth.sign_up({"email": email, "password": password})
        return res
    except Exception as e:
        st.error(f"Sign-up failed: {str(e)}")
        return None

def get_user_profile(user_id: str):
    try:
        res = supabase_client.table("user_profiles").select("role").eq("id", user_id).execute()
        if res.data:
            return res.data[0].get("role", "user")
        else:
            # No profile yet – create one with default role 'user'
            supabase_client.table("user_profiles").insert({"id": user_id, "role": "user"}).execute()
            return "user"
    except Exception as e:
        st.error(f"Failed to fetch user role: {e}")
        return "user"

def sign_out():
    supabase_client.auth.sign_out()
    st.session_state.user_session = None
    st.session_state.user_role = None
    st.session_state.messages = []
    st.rerun()

# -------------------------------------------------------------------
# 8. Login / Signup screen (shown when not authenticated)
# -------------------------------------------------------------------
if not st.session_state.user_session:
    st.markdown("## 👋 Welcome to Omni-Support")
    st.markdown("Please sign in or create an account.")

    mode = st.radio("", ["Login", "Sign Up"], horizontal=True)
    with st.form("auth_form"):
        email = st.text_input("Email", placeholder="your@email.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submit = st.form_submit_button("Continue")

        if submit:
            if not email or not password:
                st.warning("Please fill in all fields.")
            elif mode == "Login":
                res = sign_in(email, password)
                if res:
                    st.session_state.user_session = res
                    st.session_state.user_role = get_user_profile(res.user.id)
                    st.rerun()
            else:  # Sign Up
                res = sign_up(email, password)
                if res:
                    # After sign-up, automatically create profile and log in
                    st.session_state.user_session = res
                    st.session_state.user_role = get_user_profile(res.user.id)
                    st.success("Account created! You are now logged in.")
                    st.rerun()
    st.stop()

# -------------------------------------------------------------------
# 9. Once authenticated: show sidebar with user info & sign out
# -------------------------------------------------------------------
user_email = st.session_state.user_session.user.email
is_agent = (st.session_state.user_role == "agent")

with st.sidebar:
    st.markdown(f"**👤 {user_email}**")
    if is_agent:
        st.badge("Agent", color="green")
    else:
        st.badge("User", color="blue")

    if st.button("🚪 Sign Out"):
        sign_out()

    st.divider()

    if not is_agent:
        # User mode: show clear chat and voice recorder
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.markdown("### 🎤 Voice Input")
        audio_bytes = audio_recorder(text="Click to record...", icon_size="2x")
    else:
        # Agent mode: no extra controls (dashboard takes over)
        st.markdown("✅ **Agent Dashboard active**")

# -------------------------------------------------------------------
# 10. Define render functions for agent dashboard and chat
# -------------------------------------------------------------------
def render_agent_dashboard():
    """Full‑screen agent dashboard with three tabs."""
    st.markdown("## 🛠️ Agent Dashboard")

    tab1, tab2, tab3 = st.tabs(["🎫 Ticket Queue", "📚 Knowledge Base", "⚙️ System Status"])

    # ---------- TAB 1: Ticket Queue + Manage ----------
    with tab1:
        # Metrics row
        col1, col2, col3 = st.columns(3)
        total_tickets = recent_tickets = open_tickets = 0
        if supabase_client:
            try:
                total_res = supabase_client.table("tickets").select("*", count="exact").execute()
                total_tickets = total_res.count if hasattr(total_res, 'count') else len(total_res.data)
                yesterday = (datetime.now() - timedelta(days=1)).isoformat()
                recent_res = supabase_client.table("tickets").select("*", count="exact").gt("created_at", yesterday).execute()
                recent_tickets = recent_res.count if hasattr(recent_res, 'count') else len(recent_res.data)
                open_res = supabase_client.table("tickets").select("*", count="exact").neq("status", "closed").execute()
                open_tickets = open_res.count if hasattr(open_res, 'count') else len(open_res.data)
            except Exception as e:
                st.error(f"Could not fetch metrics: {e}")

        col1.metric("📊 Total Tickets", total_tickets)
        col2.metric("🕒 New (24h)", recent_tickets)
        col3.metric("🟢 Open Tickets", open_tickets)

        # Refresh button
        if st.button("🔄 Refresh Queue", key="refresh_queue"):
            st.rerun()

        # Full ticket table
        if supabase_client:
            try:
                res = supabase_client.table("tickets").select("*").order("id", desc=True).execute()
                if res.data:
                    st.dataframe(res.data, use_container_width=True, hide_index=True)
                else:
                    st.info("No tickets found.")
            except Exception as e:
                st.error(f"Failed to load queue: {e}")
        else:
            st.error("Database connection missing.")

        # ---------- Manage Ticket section ----------
        st.divider()
        st.subheader("📝 Manage Ticket")

        if supabase_client:
            # Get list of ticket IDs for selection
            try:
                id_res = supabase_client.table("tickets").select("id").order("id", desc=True).execute()
                ticket_ids = [str(t["id"]) for t in id_res.data] if id_res.data else []
            except Exception:
                ticket_ids = []

            if ticket_ids:
                selected_id_str = st.selectbox("Select a ticket by ID:", ticket_ids, key="ticket_selector")
                if selected_id_str:
                    ticket_id = int(selected_id_str)
                    # Fetch ticket details
                    try:
                        detail_res = supabase_client.table("tickets").select("*").eq("id", ticket_id).execute()
                        if detail_res.data:
                            ticket = detail_res.data[0]
                            st.text(f"**User Email:** {ticket.get('user_email', 'N/A')}")
                            st.text(f"**Description:** {ticket.get('description', 'N/A')}")
                            st.text(f"**Current Status:** {ticket.get('status', 'N/A')}")
                            st.text(f"**Created At:** {ticket.get('created_at', 'N/A')}")
                            st.text(f"**Internal Notes:** {ticket.get('internal_notes', '')}")

                            # Update controls
                            new_status = st.selectbox(
                                "Update Status",
                                ["Open", "In Progress", "Resolved", "Closed"],
                                index=["Open", "In Progress", "Resolved", "Closed"].index(ticket.get("status", "Open")),
                                key="status_update"
                            )
                            new_notes = st.text_area(
                                "Internal Notes (append or overwrite)",
                                value=ticket.get("internal_notes", ""),
                                key="notes_update"
                            )

                            if st.button("✏️ Update Ticket", key="update_ticket_btn"):
                                try:
                                    supabase_client.table("tickets").update({
                                        "status": new_status,
                                        "internal_notes": new_notes
                                    }).eq("id", ticket_id).execute()
                                    st.toast("✅ Ticket updated successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Update failed: {e}")
                        else:
                            st.info("Ticket not found.")
                    except Exception as e:
                        st.error(f"Could not load ticket details: {e}")
            else:
                st.info("No tickets to manage.")
        else:
            st.error("Database connection missing.")

    # ---------- TAB 2: Knowledge Base Management ----------
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Add Content")
            uploaded_kb_files = st.file_uploader(
                "Upload IT Manuals (PDF/TXT)",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                key="kb_uploader"
            )
            if st.button("Ingest Files", key="ingest_files"):
                if not uploaded_kb_files:
                    st.warning("Please select at least one file.")
                elif supabase_client is None:
                    st.error("Supabase client not available.")
                else:
                    with st.spinner("Extracting and embedding documents..."):
                        embedder = load_embedder()
                        for file in uploaded_kb_files:
                            text_content = ""
                            if file.name.endswith(".txt"):
                                text_content = file.getvalue().decode("utf-8")
                            elif file.name.endswith(".pdf"):
                                reader = PdfReader(file)
                                for page in reader.pages:
                                    text_content += page.extract_text() + "\n"
                            text_content = text_content.strip()
                            if not text_content:
                                continue
                            chunk_size = 1000
                            chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
                            for i, chunk in enumerate(chunks):
                                if not chunk.strip():
                                    continue
                                embedding = embedder.encode(chunk).tolist()
                                supabase_client.table("kb_articles").insert({
                                    "title": f"{file.name} (Part {i+1})",
                                    "content": chunk,
                                    "embedding": embedding
                                }).execute()
                        st.success("✅ Files successfully ingested!")
                        st.rerun()

        with col2:
            st.subheader("🗑️ Manage Content")
            if supabase_client:
                try:
                    kb_res = supabase_client.table("kb_articles").select("id, title").order("id", desc=True).execute()
                    if kb_res.data:
                        kb_options = {f"[{item['id']}] {item['title']}": item['id'] for item in kb_res.data}
                        selected_article = st.selectbox(
                            "Select a document chunk to remove:",
                            options=list(kb_options.keys()),
                            key="kb_delete_select"
                        )
                        if st.button("Delete Selected", key="delete_kb_btn"):
                            article_id = kb_options[selected_article]
                            supabase_client.table("kb_articles").delete().eq("id", article_id).execute()
                            st.success(f"✅ Deleted {selected_article}")
                            st.rerun()
                    else:
                        st.info("Knowledge base is empty.")
                except Exception as e:
                    st.error(f"Error loading KB articles: {e}")
            else:
                st.error("Database connection missing.")

    # ---------- TAB 3: System Status ----------
    with tab3:
        st.subheader("🖥️ System Status")
        st.markdown("**Supabase Connection:**")
        if supabase_client:
            st.success("✅ Connected")
        else:
            st.error("❌ Not connected")
        st.markdown("**Groq API Key:**")
        if api_key:
            st.success("✅ Loaded")
        else:
            st.error("❌ Missing")
        st.markdown("**MCP Server:**")
        st.success("✅ Ready (server.py expected)")
        st.markdown("**Environment:**")
        st.json(env_status)

# -------------------------------------------------------------------
# 11. Chat interface (for non‑agent users)
# -------------------------------------------------------------------
def render_user_chat():
    # System prompt uses the authenticated user's email
    system_prompt = (
        f"You are a helpful IT support agent. You are talking to user: {user_email}. "
        "Always use this email when creating tickets or looking up their history.\n"
        "Tool usage rules:\n"
        "1. Always search the internal knowledge base first using `search_knowledge_base`.\n"
        "2. IF `search_knowledge_base` returns no relevant results or fails, you MUST immediately call `web_search` to find the solution on the public web."
    )

    # Initialise messages if empty or if system prompt changed (e.g., after login)
    if not st.session_state.messages:
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
    elif st.session_state.messages[0]["content"] != system_prompt:
        # Update system prompt if user changed (should not happen within a session)
        st.session_state.messages[0] = {"role": "system", "content": system_prompt}

    # Display chat history (skip system and tool messages)
    for message in st.session_state.messages:
        if message.get("role") in ["tool", "system"]:
            continue
        with st.chat_message(message["role"]):
            if isinstance(message.get("content"), str):
                st.markdown(message["content"])
            elif isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "text":
                        st.markdown(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image"):
                            _, encoded = image_url.split(",", 1)
                            image_bytes = base64.b64decode(encoded)
                            st.image(image_bytes)
                        else:
                            st.image(image_url)

    # Input area
    uploaded_image = st.file_uploader("Upload Error Screenshot", type=["png", "jpg", "jpeg"])
    user_input = st.chat_input("Describe your issue...")
    voice_input = None

    # Voice input from sidebar
    if 'audio_bytes' in locals() and audio_bytes and ("last_audio" not in st.session_state or st.session_state.last_audio != audio_bytes):
        st.session_state.last_audio = audio_bytes
        with st.spinner("Transcribing audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            with open(tmp_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=("audio.wav", f.read()),
                    model="whisper-large-v3-turbo",
                )
            voice_input = transcription.text
            os.remove(tmp_path)
            st.toast(f"Transcribed: {voice_input}")

    # Determine active input
    if voice_input:
        active_input = voice_input
        st.session_state.is_voice_input = True
    elif user_input:
        active_input = user_input
        st.session_state.is_voice_input = False
    else:
        active_input = None

    if active_input or uploaded_image:
        active_text = active_input if active_input else ""
        if not active_text.strip() and uploaded_image is None:
            st.warning("Please provide a text description, record audio, or upload a screenshot.")
            st.stop()

        content = []
        if active_text.strip():
            content.append({"type": "text", "text": active_text})
        elif uploaded_image:
            content.append({"type": "text", "text": "Please analyze this uploaded screenshot and assist me."})

        if uploaded_image is not None:
            base64_image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        st.session_state.messages.append({"role": "user", "content": content})

        with st.chat_message("user"):
            if active_text:
                st.markdown(active_text)
            elif uploaded_image:
                st.markdown("*Uploaded an image.*")
            if uploaded_image is not None:
                st.image(uploaded_image)

        VISION_MODEL = "qwen/qwen3.6-27b"  # Ensure this model is available

        with st.spinner("Analyzing request and executing tools..."):
            try:
                MAX_TURNS = 5
                turn_count = 0
                response = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=st.session_state.messages,
                    tools=support_tools,
                    tool_choice="auto",
                    temperature=0.0
                )
                final_answer = None

                while turn_count < MAX_TURNS:
                    response_message = response.choices[0].message
                    tool_calls = response_message.tool_calls

                    if tool_calls:
                        assistant_msg = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                        st.session_state.messages.append(assistant_msg)
                        with st.chat_message("assistant"):
                            st.info("🛠️ Tool call requested")

                        for tool_call in tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            st.info(f"🛠️ **Agent Decision:** Trigger tool `{tool_name}` with parameters `{tool_args}`")
                            with st.spinner(f"Executing `{tool_name}` on MCP Server..."):
                                mcp_output = asyncio.run(execute_mcp_tool(tool_name, tool_args))
                            safe_mcp_output = str(mcp_output) if not isinstance(mcp_output, str) else mcp_output
                            st.session_state.messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": safe_mcp_output,
                            })
                            with st.expander(f"🔍 Developer Logs: Raw MCP Output ({tool_name})"):
                                if isinstance(mcp_output, str) and mcp_output.startswith("ERROR:"):
                                    st.error(mcp_output)
                                else:
                                    st.code(mcp_output)

                        response = client.chat.completions.create(
                            model=VISION_MODEL,
                            messages=st.session_state.messages,
                            tools=support_tools,
                            tool_choice="auto",
                            temperature=0.0
                        )
                        turn_count += 1
                    else:
                        final_answer = response_message.content or ""
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        break
                else:
                    st.warning(f"⚠️ Reached maximum tool call turns ({MAX_TURNS}). Showing partial response.")
                    final_answer = response_message.content if response_message.content else "Max turns reached without final answer."
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                if final_answer is not None:
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                        if st.session_state.get("is_voice_input", False):
                            with st.spinner("Generating voice response..."):
                                audio_fp = generate_tts_audio(final_answer)
                                if audio_fp:
                                    st.audio(audio_fp, format="audio/mp3", autoplay=True)
                            st.session_state.is_voice_input = False

            except Exception as e:
                st.error(f"❌ **API Error:** {str(e)}")

# -------------------------------------------------------------------
# 12. Main routing: agent dashboard vs user chat
# -------------------------------------------------------------------
if is_agent:
    render_agent_dashboard()
else:
    render_user_chat()