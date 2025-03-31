# Simple LangGraph Application

A basic conversational agent built with LangGraph, showcasing state management and graph-based workflows.

## Features

- Simple question-answering agent
- State management for conversation context
- Visual debugging with LangGraph Studio

## Setup

1. Clone this repository
2. Create a `.env` file with your API keys (see `.env.example`)
3. Install dependencies:
   ```
   pip install -e .
   ```
4. Run the application:
   ```
   langgraph dev
   ```
5. Access the LangGraph Studio Web UI at the URL provided in the terminal output

## Structure

- `app/agent.py`: Main agent definition with graph workflow
- `langgraph.json`: LangGraph application configuration
- `pyproject.toml`: Project metadata and dependencies
- `requirements.txt`: Direct dependencies

## License

MIT
