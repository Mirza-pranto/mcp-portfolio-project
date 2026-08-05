import os
import asyncio
import json
import base64
import tempfile
import io
import re

import streamlit as st
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from groq import Groq
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
from supabase import create_client
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Propagate Supabase secrets into the process environment so the MCP server can access them.
if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
    os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]
    st.info("✅ Supabase credentials loaded from st.secrets into process environment.")
else:
    st.warning("⚠️ Supabase keys missing from st.secrets! Check .streamlit/secrets.toml")

# Initialize Supabase client for the frontend dashboard
supabase_client = None
if "SUPABASE_URL" in os.environ and "SUPABASE_KEY" in os.environ:
    supabase_client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# Diagnostic environment reporting
env_status = {
    "SUPABASE_URL": "SUPABASE_URL" in os.environ,
    "SUPABASE_KEY": "SUPABASE_KEY" in os.environ,
}
st.write(f"Debug: SUPABASE_URL set = {env_status['SUPABASE_URL']}, SUPABASE_KEY set = {env_status['SUPABASE_KEY']}")

@st.cache_resource
def load_embedder():
    """Caches the embedding model so it doesn't reload on every UI click."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# 1. Setup Groq API Key
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

# 2. TTS Helper Functions
def clean_text_for_speech(text: str, max_chars: int = 350) -> str:
    """Strips markdown formatting/code and limits length to prevent Google TTS rate limits."""
    text = re.sub(r'```[\s\S]*?```', ' Code snippet omitted. ', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[\*#_~]', '', text)
    text = re.sub(r'\n+', ' ', text).strip()
    
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(' ', 1)[0] + "... Please see screen for full details."
    return text

def generate_tts_audio(text: str) -> io.BytesIO | None:
    """Safely converts text to an in-memory MP3 stream using gTTS with error fallback."""
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

# 3. Async MCP Execution with explicit subprocess environment inheritance
async def execute_mcp_tool(tool_name: str, tool_args: dict):
    mcp_env = os.environ.copy()
    transport = PythonStdioTransport("server.py", env=mcp_env)
    mcp_client = Client(transport)
    async with mcp_client:
        result = await mcp_client.call_tool(tool_name, tool_args)
        return result.data

# 4. Define Tools using the standard OpenAI/Groq JSON Schema
support_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "Retrieves the status of a specific support ticket by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "integer"}
                },
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
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 5
                    }
                }
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
                "properties": {
                    "email": {"type": "string"}
                },
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
                "properties": {
                    "query": {"type": "string", "description": "The search query based on the user's issue."}
                },
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
                "properties": {
                    "query": {"type": "string", "description": "The search query string"}
                },
                "required": ["query"]
            }
        }
    }
]

# 5. Streamlit UI Elements & State Initialization
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# --- NEW: The Login Wall ---
if not st.session_state.user_email:
    st.markdown("## 👋 Welcome to Omni-Support")
    st.markdown("Please sign in to access your support history and create new tickets.")
    
    with st.form("login_form"):
        email_input = st.text_input("Email Address", placeholder="name@company.com")
        submit_btn = st.form_submit_button("Start Chat")
        
        if submit_btn and email_input:
            st.session_state.user_email = email_input
            st.rerun()
            
    st.stop() # This stops the rest of the app from loading until they log in!

# Updated system prompt with explicit tool‑usage rules
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful IT support agent. You are talking to user: {user_email}. "
    "Always use this email when creating tickets or looking up their history.\n"
    "Tool usage rules:\n"
    "1. Always search the internal knowledge base first using `search_knowledge_base`.\n"
    "2. IF `search_knowledge_base` returns no relevant results or fails, you MUST immediately call `web_search` to find the solution on the public web."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(user_email=st.session_state.user_email)
        }
    ]

if "is_voice_input" not in st.session_state:
    st.session_state.is_voice_input = False

if "is_agent" not in st.session_state:
    st.session_state.is_agent = False
    
with st.sidebar:
    st.markdown("### Controls")
    if st.button("Clear Chat"):
        # Reset the chat but keep the user logged in
        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(user_email=st.session_state.user_email)
            }
        ]
        st.session_state.is_voice_input = False
        st.rerun()
        
    st.divider()
    st.markdown("### 🎤 Voice Input")
    audio_bytes = audio_recorder(text="Click to record...", icon_size="2x")        
    
    st.divider()
    st.markdown("### 🔒 Agent Portal")
    
    # Check if the user is logged in as an agent
    if not st.session_state.is_agent:
        st.caption("Staff only. Enter PIN to view queue.")
        agent_pin = st.text_input("PIN", type="password", key="pin_input")
        
        # Fallback to '0000' if AGENT_PIN is missing from secrets
        correct_pin = st.secrets.get("AGENT_PIN", "0000") 
        
        if st.button("Login"):
            if agent_pin == correct_pin:
                st.session_state.is_agent = True
                st.success("Unlocked!")
                st.rerun()
            else:
                st.error("Invalid PIN.")
                
    # If logged in, show Ticket Queue and Knowledge Base tools
    else:
        if st.button("Log Out Agent"):
            st.session_state.is_agent = False
            st.rerun()
            
        st.markdown("### 📋 Live Ticket Queue")
        if st.button("Refresh Queue"):
            st.rerun()
            
        if supabase_client:
            try:
                res = (supabase_client.table("tickets")
                       .select("id, status, description, user_email")
                       .order("id", desc=True)
                       .limit(10)
                       .execute())
                if res.data:
                    st.dataframe(res.data, use_container_width=True, hide_index=True)
                else:
                    st.caption("No active tickets found.")
            except Exception as e:
                st.caption(f"Unable to load queue: {e}")
        else:
            st.error("Database connection missing.")
            
        st.divider()

        # --- Manage & Delete Knowledge Base Articles ---
        st.markdown("### 🗑️ Manage Knowledge Base")
        
        if supabase_client:
            try:
                # Fetch all KB articles to display in the dropdown
                kb_res = (supabase_client.table("kb_articles")
                          .select("id, title")
                          .order("id", desc=True)
                          .execute())
                
                if kb_res.data:
                    # Create a dictionary to map the display name to the database ID
                    kb_options = {f"[{item['id']}] {item['title']}": item['id'] for item in kb_res.data}
                    
                    selected_article = st.selectbox(
                        "Select a document chunk to remove:", 
                        options=list(kb_options.keys())
                    )
                    
                    if st.button("Delete Selected"):
                        article_id = kb_options[selected_article]
                        # Delete the specific row from Supabase
                        supabase_client.table("kb_articles").delete().eq("id", article_id).execute()
                        st.success(f"✅ Deleted {selected_article}")
                        st.rerun() # Refresh the UI immediately
                else:
                    st.caption("Knowledge base is currently empty.")
            except Exception as e:
                st.caption(f"Unable to load KB articles: {e}")    

        st.divider()

        # --- Knowledge Base Uploader ---
        st.markdown("### 📚 Update Knowledge Base")
        uploaded_kb_files = st.file_uploader("Upload IT Manuals", type=["pdf", "txt"], accept_multiple_files=True)
        
        if st.button("Ingest Files") and uploaded_kb_files:
            with st.spinner("Extracting and embedding documents..."):
                embedder = load_embedder()
                
                for file in uploaded_kb_files:
                    text_content = ""
                    
                    # Extract text based on file type
                    if file.name.endswith(".txt"):
                        text_content = file.getvalue().decode("utf-8")
                    elif file.name.endswith(".pdf"):
                        reader = PdfReader(file)
                        for page in reader.pages:
                            text_content += page.extract_text() + "\n"
                            
                    # Clean up spacing
                    text_content = text_content.strip()
                    if not text_content:
                        continue
                        
                    # Chunk the text into ~1000 character segments
                    chunk_size = 1000
                    chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
                    
                    for i, chunk in enumerate(chunks):
                        if not chunk.strip(): 
                            continue
                            
                        # Generate vector embedding for the chunk
                        embedding = embedder.encode(chunk).tolist()
                        
                        # Insert into Supabase
                        supabase_client.table("kb_articles").insert({
                            "title": f"{file.name} (Part {i+1})",
                            "content": chunk,
                            "embedding": embedding
                        }).execute()
                        
                st.success("✅ Files successfully processed and added to the Knowledge Base!")
                st.rerun() # Refreshes sidebar so new items show up in the Delete dropdown right away


def render_message_content(content):
    if isinstance(content, str):
        st.markdown(content)
    elif isinstance(content, list):
        for item in content:
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

for message in st.session_state.messages:
    # Skip rendering system messages and tool outputs in the main chat UI
    if message.get("role") in ["tool", "system"]:
        continue
    with st.chat_message(message["role"]):
        render_message_content(message.get("content"))

uploaded_image = st.file_uploader("Upload Error Screenshot", type=["png", "jpg", "jpeg"])
user_input = st.chat_input("Describe your issue...")
voice_input = None

# Process Voice Input via Groq Whisper
if audio_bytes and ("last_audio" not in st.session_state or st.session_state.last_audio != audio_bytes):
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

# Determine active input type and set tracking flag
if voice_input:
    active_input = voice_input
    st.session_state.is_voice_input = True
elif user_input:
    active_input = user_input
    st.session_state.is_voice_input = False
else:
    active_input = None

# Handle state where only an image is uploaded (without text)
if active_input or uploaded_image:
    
    # Default text fallback if only an image is uploaded
    active_text = active_input if active_input else ""
    if not active_text.strip() and uploaded_image is None:
        st.warning("Please provide a text description, record audio, or upload a screenshot.")
        st.stop()

    content = []
    
    if active_text.strip():
        content.append({"type": "text", "text": active_text})
    elif uploaded_image:
        # Provide the LLM with instructions if the user only provided an image
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

    VISION_MODEL = "qwen/qwen3.6-27b" # Note: Ensure this model identifier matches Groq's exact current list!

    with st.spinner("Analyzing request and executing tools..."):
        try:
            # ---- MULTI‑TURN TOOL EXECUTION LOOP ----
            MAX_TURNS = 5
            turn_count = 0

            # Initial LLM call
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=st.session_state.messages,
                tools=support_tools,
                tool_choice="auto",
                temperature=0.0
            )

            # We'll collect tool calls in this loop; final answer will be set when no tool_calls.
            final_answer = None

            while turn_count < MAX_TURNS:
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                if tool_calls:
                    # 1. Append assistant message with tool_calls
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

                    # Show a placeholder in the UI that tool calls are happening
                    with st.chat_message("assistant"):
                        st.info("🛠️ Tool call requested")

                    # 2. Execute each tool and append results
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        st.info(f"🛠️ **Agent Decision:** Trigger tool `{tool_name}` with parameters `{tool_args}`")

                        with st.spinner(f"Executing `{tool_name}` on MCP Server..."):
                            mcp_output = asyncio.run(execute_mcp_tool(tool_name, tool_args))

                        # Safely convert tool responses to string to prevent API schema validation errors
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

                    # 3. Next LLM call with the updated messages (including tool responses)
                    response = client.chat.completions.create(
                        model=VISION_MODEL,
                        messages=st.session_state.messages,
                        tools=support_tools,
                        tool_choice="auto",
                        temperature=0.0
                    )
                    turn_count += 1
                    # Continue the loop to see if the LLM calls more tools

                else:
                    # No tool_calls → final answer ready
                    final_answer = response_message.content or ""
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    break

            else:
                # Loop ended because MAX_TURNS reached without a final answer
                st.warning(f"⚠️ Reached maximum tool call turns ({MAX_TURNS}). Showing partial response.")
                # Use the last response content if available, otherwise a fallback
                final_answer = response_message.content if response_message.content else "Max turns reached without final answer."
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            # 4. Render the final assistant message (already appended) and handle TTS
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