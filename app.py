import streamlit as st
import os
import asyncio
import json
import base64
import tempfile
import io
import re
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from groq import Groq
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
from supabase import create_client

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
                    "description": {"type": "string"}
                },
                "required": ["description"]
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
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_voice_input" not in st.session_state:
    st.session_state.is_voice_input = False

if "is_agent" not in st.session_state:
    st.session_state.is_agent = False

with st.sidebar:
    st.markdown("### Controls")
    if st.button("Clear Chat"):
        st.session_state.messages = []
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
                
    # If logged in, show the Live Ticket Queue
    else:
        if st.button("Log Out"):
            st.session_state.is_agent = False
            st.rerun()
            
        st.markdown("### 📋 Live Ticket Queue")
        if st.button("Refresh Queue"):
            st.rerun()
            
        if supabase_client:
            try:
                # Fetch the latest 10 open tickets
                res = (supabase_client.table("tickets")
                       .select("id, status, description")
                       .order("id", desc=True)
                       .limit(10)
                       .execute())
                
                if res.data:
                    # Display as a clean, interactive dataframe
                    st.dataframe(res.data, use_container_width=True, hide_index=True)
                else:
                    st.caption("No active tickets found.")
            except Exception as e:
                st.caption(f"Unable to load queue: {e}")
        else:
            st.error("Database connection missing.")

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
    if message.get("role") == "tool":
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

if active_input is not None:
    if not active_input.strip() and uploaded_image is None:
        st.warning("Please provide a text description, record audio, or upload a screenshot.")
        st.stop()

    content = []
    if active_input:
        content.append({"type": "text", "text": active_input})

    if uploaded_image is not None:
        base64_image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    st.session_state.messages.append({"role": "user", "content": content})

    with st.chat_message("user"):
        if active_input:
            st.markdown(active_input)
        if uploaded_image is not None:
            st.image(uploaded_image)

    VISION_MODEL = "qwen/qwen3.6-27b"

    with st.spinner("Analyzing request and executing tools..."):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=st.session_state.messages,
                tools=support_tools,
                tool_choice="auto",
                temperature=0.0
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
                st.session_state.messages.append(assistant_message)

                with st.chat_message("assistant"):
                    st.info("🛠️ Tool call requested")

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    st.info(f"🛠️ **Agent Decision:** Trigger tool `{tool_name}` with parameters `{tool_args}`")

                    with st.spinner(f"Executing `{tool_name}` on MCP Server..."):
                        mcp_output = asyncio.run(execute_mcp_tool(tool_name, tool_args))

                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": mcp_output,
                    })

                    if isinstance(mcp_output, str) and mcp_output.startswith("ERROR:"):
                        st.error(f"⚠️ **MCP Server Error:**\n\n{mcp_output}")
                    else:
                        st.success(f"✅ **MCP Server Output:**\n\n{mcp_output}")

                follow_up_response = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=st.session_state.messages,
                    tools=support_tools,
                    tool_choice="auto",
                    temperature=0.0
                )
                final_content = follow_up_response.choices[0].message.content or ""
                st.session_state.messages.append({"role": "assistant", "content": final_content})

                with st.chat_message("assistant"):
                    st.markdown(final_content)
                    if st.session_state.get("is_voice_input", False):
                        with st.spinner("Generating voice response..."):
                            audio_fp = generate_tts_audio(final_content)
                            if audio_fp:
                                st.audio(audio_fp, format="audio/mp3", autoplay=True)
                        st.session_state.is_voice_input = False
            else:
                assistant_text = response_message.content or ""
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                with st.chat_message("assistant"):
                    st.markdown(assistant_text)
                    if st.session_state.get("is_voice_input", False):
                        with st.spinner("Generating voice response..."):
                            audio_fp = generate_tts_audio(assistant_text)
                            if audio_fp:
                                st.audio(audio_fp, format="audio/mp3", autoplay=True)
                        st.session_state.is_voice_input = False

        except Exception as e:
            st.error(f"❌ **API Error:** {str(e)}")