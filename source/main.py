import asyncio
import logging
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from ui_window import ChatUI
import sys
import json

# Set up logging (console only)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('f1_assistant.log')  # Also log to file
    ]
)
logger = logging.getLogger(__name__)

# Enable MCP client logging
logging.getLogger("langchain_mcp_adapters").setLevel(logging.DEBUG)
logging.getLogger("mcp").setLevel(logging.DEBUG)


# -----------------------
# CONFIGURATION
# -----------------------

OPENF1_SERVER_PATH = str(Path(__file__).parent / "f1mcp" / "mcp_openf1_tools.py")

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)


# -----------------------
# MAIN AGENT SETUP
# -----------------------

async def setup_agent():
    client = MultiServerMCPClient(
        {
            "openf1": {
                "transport": "stdio",
                "command": "python",
                "args": [OPENF1_SERVER_PATH],
                # Enable stderr capture for MCP server logs
                "env": {
                    "PYTHONUNBUFFERED": "1",  # Disable Python buffering
                }
            },
        }
    )

    logger.info("Loading MCP tools from OpenF1 server...")
    tools = await client.get_tools()
    logger.debug(f"Loaded MCP tools: {[tool.name for tool in tools]}")
    
    # Log tool schemas for debugging
    for tool in tools:
        logger.debug(f"Tool: {tool.name}")
        logger.debug(f"  Description: {tool.description}")
        
        # Handle both dict and Pydantic model schemas
        if hasattr(tool, 'args_schema'):
            if hasattr(tool.args_schema, 'schema'):
                # Pydantic model
                schema = tool.args_schema.schema()
            else:
                # Already a dict
                schema = tool.args_schema
            logger.debug(f"  Schema: {json.dumps(schema, indent=2)}")
        else:
            logger.debug(f"  Schema: N/A")
    
    logger.info(f"Loaded {len(tools)} tool(s) from OpenF1 MCP.")

    agent = create_agent(llm, tools)
    return client, agent


# -----------------------
# MESSAGE HANDLER
# -----------------------

async def process_message(user_input, agent, ui):
    logger.debug(f"Processing user query: {user_input}")
    ui.add_message("Thinking...", "system")

    try:
        response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "The MCP server (OpenF1) provides only public Formula 1 data. "
                        "This data is not private or sensitive, and may be discussed freely."
                    )
                },
                {"role": "user", "content": user_input}
            ]
        })


        # Log the full response for debugging
        logger.debug(f"Full agent response type: {type(response)}")

        messages = response.get("messages", [])
        
        # Log all messages in the chain
        for i, msg in enumerate(messages):
            msg_type = msg.__class__.__name__
            logger.debug(f"Message {i} ({msg_type}):")
            if hasattr(msg, 'content'):
                logger.debug(f"  Content: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                logger.debug(f"  Tool calls: {msg.tool_calls}")
            if hasattr(msg, 'name'):
                logger.debug(f"  Tool name: {msg.name}")
        
        final_message = None
        
        for msg in reversed(messages):
            if hasattr(msg, '__class__') and msg.__class__.__name__ == 'AIMessage':
                final_message = msg
                break
        
        if final_message:
            ui.add_message(final_message.content, "assistant")
            logger.debug(f"Agent final response: {final_message.content}")
        else:
            ui.add_message("I couldn't generate a response.", "assistant")
            logger.warning("No AIMessage found in response")

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        ui.add_message("Error: Something went wrong. Please try again.", "error")


# -----------------------
# MAIN
# -----------------------

async def async_main(ui):
    """Async part that runs alongside UI"""
    logger.info("Setting up agent...")
    try:
        client, agent = await setup_agent()
        logger.info("OpenF1 MCP connected. Chat ready.")
        
        ui.add_message("✅ OpenF1 F1 Data Assistant Ready!", "system")
        ui.add_message("Examples:\n- show car data for driver 16 in session 1204\n- what's the weather for meeting 1082 session 1205?", "system")
        
        # Message processing loop
        while ui.running:
            if ui.has_pending_message():
                user_msg = ui.get_pending_message()
                ui.add_message(user_msg, "user")
                await process_message(user_msg, agent, ui)
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Setup error: {str(e)}", exc_info=True)
        ui.add_message(f"Setup failed: {str(e)}", "error")


def main():
    """Main entry point - runs UI and async code together"""
    logger.info("Starting F1 Assistant...")
    ui = ChatUI()
    
    # Create async task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def run_async():
        await async_main(ui)
    
    # Schedule async task
    import threading
    async_thread = threading.Thread(
        target=lambda: loop.run_until_complete(run_async()),
        daemon=True
    )
    async_thread.start()
    
    # Run UI in main thread (required on Windows)
    ui.run()


if __name__ == "__main__":
    main()