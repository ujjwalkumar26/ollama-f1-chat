import asyncio
import logging
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from ui_window import ChatUI
import sys
import json

# Set up logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('f1_assistant.log', encoding='utf-8')
    ]
)
# Force UTF-8 for stdout on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
logger = logging.getLogger(__name__)

# Enable MCP client logging
logging.getLogger("langchain_mcp_adapters").setLevel(logging.DEBUG)
logging.getLogger("mcp").setLevel(logging.DEBUG)


# -----------------------
# CONFIGURATION
# -----------------------

OPENF1_SERVER_PATH = str(Path(__file__).parent / "f1mcp" / "mcp_openf1_tools.py")
CHROMA_DB_PATH = str(Path(__file__).parent.parent / "chroma_db")

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.5,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


# -----------------------
# VECTOR STORE SETUP
# -----------------------

def setup_vectorstore():
    """Initialize ChromaDB vector store"""
    logger.info(f"Loading ChromaDB from {CHROMA_DB_PATH}")
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name="personal_knowledge"
        )
        logger.info("ChromaDB loaded successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading ChromaDB: {str(e)}", exc_info=True)
        return None


# -----------------------
# MAIN AGENT SETUP
# -----------------------

def get_search_tool():
    """Initialize DuckDuckGo search tool"""
    search = DuckDuckGoSearchRun()
    return Tool(
        name="Internet Search",
        func=search.run,
        description="Useful for searching information on the internet. Use this when you need to find current or factual Formula 1 information. Do not use for anything other than F1/Formula 1 related queries.",
    )

def get_system_prompt():
    """Return the system prompt as a plain string"""
    return """You are a helpful Formula 1 assistant with access to F1 public data through specialized tools.
            Your role:
            - Answer Formula 1 related questions using the available tools
            - Use the OpenF1 API tools to fetch real-time F1 data (sessions, drivers, meetings, etc.)
            - Use Internet Search for current F1 news and information not available in the API
            - Provide accurate, factual information about Formula 1

            Guidelines:
            - Always check if you have relevant knowledge before using tools
            - Use tools when you need specific data or current information
            - Be concise and informative in your responses
            - If you're unsure, say so rather than making up information
            - When a tool returns results, YOU MUST USE THEM in your answer

            The MCP server (OpenF1) provides only public Formula 1 data. This data is not private or sensitive, and may be discussed freely."""


async def setup_agent():
    client = MultiServerMCPClient(
        {
            "Openf1API": {
                "transport": "stdio",
                "command": "python",
                "args": [OPENF1_SERVER_PATH],
                "env": {
                    "PYTHONUNBUFFERED": "1",
                }
            },
        }
    )

    logger.info("Loading MCP tools from OpenF1 server...")
    tools = await client.get_tools()
    tools.append(get_search_tool())
    logger.debug(f"Loaded MCP tools: {[tool.name for tool in tools]}")
    
    # Log tool schemas for debugging
    for tool in tools:
        logger.debug(f"Tool: {tool.name}")
        logger.debug(f"  Description: {tool.description}")
        
        if hasattr(tool, 'args_schema'):
            if hasattr(tool.args_schema, 'schema'):
                schema = tool.args_schema.model_json_schema()
            else:
                schema = tool.args_schema
            logger.debug(f"  Schema: {json.dumps(schema, indent=2)}")
        else:
            logger.debug(f"  Schema: N/A")
    
    logger.info(f"Loaded {len(tools)} tool(s) from OpenF1 MCP.")

    # Get the system prompt as a string
    system_prompt = get_system_prompt()

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,  # Pass as string, not template
        checkpointer=InMemorySaver(),
    )
    return client, agent


# -----------------------
# KNOWLEDGE RETRIEVAL
# -----------------------

def retrieve_relevant_knowledge(query, vectorstore, k=3):
    """Retrieve relevant documents from vector store"""
    if not vectorstore:
        return ""
    
    try:
        logger.debug(f"Searching vector store for: {query}")
        docs = vectorstore.similarity_search(query, k=k)
        
        if not docs:
            logger.debug("No relevant documents found in vector store")
            return ""
        
        # Format the retrieved knowledge
        knowledge_context = "\n\n".join([
            f"[Knowledge {i+1}]\n{doc.page_content}\nSource: {doc.metadata.get('source', 'Unknown')}"
            for i, doc in enumerate(docs)
        ])
        
        logger.debug(f"Retrieved {len(docs)} relevant documents")
        return knowledge_context
    except Exception as e:
        logger.error(f"Error retrieving from vector store: {str(e)}", exc_info=True)
        return ""


# -----------------------
# MESSAGE HANDLER
# -----------------------

async def process_message(user_input, agent, ui, vectorstore):
    logger.debug(f"Processing user query: {user_input}")
    ui.add_message("Thinking...", "system")

    try:
        # Retrieve relevant knowledge from vector store
        relevant_knowledge = retrieve_relevant_knowledge(user_input, vectorstore)
        
        # Build the user message with context if available
        user_message = user_input
        if relevant_knowledge:
            user_message = f"""Context from Personal Knowledge Base (data up to 2023):
            {relevant_knowledge}

            User Question: {user_input}

            Please use the above context if relevant to answer the question."""
            logger.debug("Added personal knowledge to user message")
        
        response = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": user_message}
                ],
            },
            {"configurable": {"thread_id": "1"}}
        )

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
        # Initialize vector store
        vectorstore = setup_vectorstore()
        
        # Setup agent
        client, agent = await setup_agent()
        logger.info("OpenF1 MCP connected. Chat ready.")
        
        ui.add_message("✅ OpenF1 F1 Data Assistant Ready!", "system")
        if vectorstore:
            ui.add_message("✅ Personal Knowledge Base Connected!", "system")
        else:
            ui.add_message("⚠️ Personal Knowledge Base not available", "system")
        ui.add_message("Examples:\n- show car data for driver 16 in session 1204\n- what's the weather for meeting 1082 session 1205?", "system")
        
        # Message processing loop
        while ui.running:
            if ui.has_pending_message():
                user_msg = ui.get_pending_message()
                ui.add_message(user_msg, "user")
                await process_message(user_msg, agent, ui, vectorstore)
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