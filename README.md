# MCP Client-Server Python Example

This project demonstrates a simple client-server [MCP](https://modelcontextprotocol.io/introduction).

---

## What is MCP (Model Context Protocol)?

> **MCP** is an open protocol introduced by Anthropic to enable large language models (LLMs) to interact with external tools, APIs, and resources in a standardized, extensible way.  
> It facilitates secure, multi-channel communication between AI models and external systems, supporting advanced agentic workflows and tool use.  
>  
> 🔗 [Anthropic's announcement](https://www.anthropic.com/news/introducing-the-model-context-protocol)  
> 🔗 [MCP documentation](https://github.com/anthropics/mcp)

---

## Features

- **MCP Server:** Exposes tools (e.g., addition) and resources (e.g., greetings) via SSE.
- **MCP Client:** Connects to the server, lists available tools, and interacts using OpenAI's chat completions.
- **OpenAI Integration:** Uses OpenAI's GPT models to process user queries and call server tools as needed.

---

## Requirements

- Python 3.12+
- [MCP Python SDK](https://pypi.org/project/mcp/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Uvicorn](https://www.uvicorn.org/) (for running the server)
- [python-dotenv](https://pypi.org/project/python-dotenv/) (for loading environment variables)
- [uv](https://github.com/astral-sh/uv) (fast Python package installer and resolver)

## Project Structure

```
.
├── client.py      # MCP client implementation
├── server.py      # MCP server implementation
├── pyproject.toml # Project metadata and dependencies
├── .env           # Environment variables (not committed)
└── README.md      # This file
```

**Install dependencies with [uv](https://github.com/astral-sh/uv):**
```bash
uv sync
```
*(This will install all dependencies as specified in `uv.lock`.)*

---

## Setup

1. **Environment Variables**

   Create a `.env` file in the project directory:
   ```
   OPENAI_API_KEY=your-openai-api-key
   MCP_SSE_URL=http://localhost:8080/sse
   ```

2. **Start the Server**

   ```bash
   uv run server.py --host 0.0.0.0 --port 8080
   ```
   The server exposes tools and resources via SSE at `/sse`.

3. **Run the Client**

   In another terminal:
   ```bash
   uv run client.py
   ```
   The client will connect to the server, list available tools, and start an interactive chat loop.

---

## Usage

- Type your queries in the client prompt.
- The client will use OpenAI to process your query and call server tools if needed.
- Type `quit` to exit the client.


