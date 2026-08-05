# app.py
import streamlit as st
import os
import asyncio
import json
import base64
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from groq import Groq

# Propagate Supabase secrets into the process environment so the MCP server can access them.
if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    os.environ["SUPABASE_URL"] = st.secrets["SUPABASE_URL"]
    os.environ["SUPABASE_KEY"] = st.secrets["SUPABASE_KEY"]
    st.info("✅ Supabase credentials loaded from st.secrets into process environment.")
else:
    st.warning("⚠️ Supabase keys missing from st.secrets! Check .streamlit/secrets.toml")

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

# 2. Async MCP Execution with explicit subprocess environment inheritance
async def execute_mcp_tool(tool_name: str, tool_args: dict):
    mcp_env = os.environ.copy()
    transport = PythonStdioTransport("server.py", env=mcp_env)
    mcp_client = Client(transport)
    async with mcp_client:
        result = await mcp_client.call_tool(tool_name, tool_args)
        return result.data

# 3. Define Tools using the standard OpenAI/Groq JSON Schema
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
    }
]

# 4. Streamlit UI Elements
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### Controls")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


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

if user_input is not None:
    if not user_input.strip() and uploaded_image is None:
        st.warning("Please provide a text description or upload a screenshot.")
        st.stop()

    content = []
    if user_input:
        content.append({"type": "text", "text": user_input})

    if uploaded_image is not None:
        base64_image = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    st.session_state.messages.append({"role": "user", "content": content})

    with st.chat_message("user"):
        if user_input:
            st.markdown(user_input)
        if uploaded_image is not None:
            st.image(uploaded_image)

    with st.spinner("Analyzing request and executing tools..."):
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.6-27b",
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
                    model="qwen/qwen3.6-27b",
                    messages=st.session_state.messages,
                    tools=support_tools,
                    tool_choice="auto",
                    temperature=0.0
                )
                final_content = follow_up_response.choices[0].message.content or ""
                st.session_state.messages.append({"role": "assistant", "content": final_content})

                with st.chat_message("assistant"):
                    st.markdown(final_content)
            else:
                assistant_text = response_message.content or ""
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

                with st.chat_message("assistant"):
                    st.markdown(assistant_text)

        except Exception as e:
            st.error(f"❌ **API Error:** {str(e)}")