# F1 Data Chat Assistant

An interactive chat interface for querying Formula 1 data using OpenF1 API. This project uses LangChain, Ollama, ChromaDb for Vector search and Model Context Protocol (MCP) to create a natural language interface for F1 data queries.

## Features

- Real-time F1 data queries using OpenF1 API
- Natural language interface powered by Ollama LLM
- Support for multiple data types:
  - Driver information
  - Car telemetry data
  - Lap times
  - Session information
  - Weather data

## Requirements

- Python 3.8+
- Ollama installed locally
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ollama-f1-chat.git
cd ollama-f1-chat
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure Ollama is running with the required model:

```bash
ollama run llama3.2:1b
```

## Usage

Run the main application:

```bash
python source/main.py
```

Example queries:

- "show car data for driver 16 in session 1204"
- "what's the weather for meeting 1082 session 1205?"
- "tell me about lap 3 for driver 55 session 1204"

## Project Structure

- `source/main.py` - Main application entry point
- `source/f1mcp/` - MCP server implementation for OpenF1 API
- `requirements.txt` - Python dependencies

## License

[MIT License](LICENSE)
