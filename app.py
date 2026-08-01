# app.py
import streamlit as st
import os
import asyncio
import json
import base64
from fastmcp import Client
from groq import Groq

# 1. Setup Groq API Key
if "GROQ_API_KEY" not in os.environ:
    # Do not hardcode secrets here. Set it in your terminal before running streamlit:
    # PowerShell: $env:GROQ_API_KEY="your_new_key_here"
    pass

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

client = Groq()

st.set_page_config(page_title="Omni-Support AI", layout="centered", page_icon="⚡")
st.title("⚡ IT Support Agent (Groq Edition)")
st.caption("Powered by Llama 3.2 Vision & Model Context Protocol (MCP)")

# 2. Async MCP Execution (Unchanged)
async def execute_mcp_tool(tool_name: str, tool_args: dict):
    mcp_client = Client("server.py")
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
    }
]

# 4. Streamlit UI Elements
uploaded_image = st.file_uploader("Upload Error Screenshot", type=["png", "jpg", "jpeg"])
user_text = st.text_input("Describe your issue or query:", placeholder="e.g., What is the status of ticket 1?")

if st.button("Submit to Support Agent", type="primary"):
    with st.spinner("Analyzing request and executing tools..."):
        
        # 5. Prepare Message Content (Text + Base64 Image)
        content = []
        if user_text:
            content.append({"type": "text", "text": user_text})
        
        if uploaded_image:
            # Convert uploaded image to base64 for Groq
            base64_image = base64.b64encode(uploaded_image.getvalue()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            
        if not content:
            st.warning("Please provide a text description or upload a screenshot.")
            st.stop()
            
        # 6. Send request to Groq API
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=[{"role": "user", "content": content}],
                tools=support_tools,
                tool_choice="auto",
                temperature=0.0
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            # 7. Check if Groq decided to invoke an MCP Tool
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    # Groq returns arguments as a JSON string, so we must parse it
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    st.info(f"🛠️ **Agent Decision:** Trigger tool `{tool_name}` with parameters `{tool_args}`")
                    
                    # Execute the real FastMCP client call!
                    with st.spinner(f"Executing `{tool_name}` on MCP Server..."):
                        mcp_output = asyncio.run(execute_mcp_tool(tool_name, tool_args))
                    
                    # Display real execution results from the SQLite DB
                    st.success(f"✅ **MCP Server Output:**\n\n{mcp_output}")
            else:
                # Standard conversational response
                st.markdown(response_message.content)

        except Exception as e:
            st.error(f"❌ **API Error:** {str(e)}")