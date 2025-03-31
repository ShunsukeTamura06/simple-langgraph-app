"""
Simple conversation agent using LangGraph.

This module implements a basic conversation agent with memory
using LangGraph's state management capabilities.
"""

import os
from typing import Annotated, Any, Dict, List, Literal, TypedDict, Union

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Define the state for our conversation
class ConversationState(TypedDict):
    """State for the conversation agent."""

    messages: List[BaseMessage]
    config: Dict[str, Any]


# Define a model for configuring the agent
class AgentConfig(BaseModel):
    """Configuration for the conversation agent."""

    model: str = Field(
        default="gpt-4o", 
        description="The LLM model to use for the conversation"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="The temperature for the LLM generation"
    )


def get_llm(config: Dict[str, Any]):
    """Get the appropriate LLM based on the configuration."""
    model = config.get("model", "gpt-4o")
    temperature = config.get("temperature", 0.7)
    
    if model.startswith("gpt"):
        return ChatOpenAI(model=model, temperature=temperature)
    elif model.startswith("claude"):
        return ChatAnthropic(model=model, temperature=temperature)
    else:
        # Default to OpenAI
        return ChatOpenAI(model="gpt-4o", temperature=temperature)


def agent_node(state: ConversationState) -> Annotated[Union[Dict, Command[Literal[END]]], "agent"]:
    """Process user input and generate a response."""
    config = state.get("config", {})
    messages = state["messages"]
    
    # Get the appropriate LLM
    llm = get_llm(config)
    
    # Generate a response
    ai_message = llm.invoke(messages)
    
    # Return the AI's response
    return {"messages": [ai_message]}


def human_node(state: ConversationState) -> Annotated[Union[Dict, Command[Literal["agent"]]], "human"]:
    """Handle human interaction, which can be simulated or interrupted for real input."""
    # For development and debugging, we can interrupt here to get real human input
    user_input = interrupt(value="Waiting for user input...")
    
    # Create a human message
    message = HumanMessage(content=user_input)
    
    # Return the human message and direct to the agent node
    return Command(
        goto="agent",
        update={"messages": [message]},
    )


def setup_graph(config: Dict[str, Any] = None) -> StateGraph:
    """Create the conversation graph."""
    if config is None:
        config = {}
    
    # Create a new graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("human", human_node)
    
    # Set the entry point
    workflow.set_entry_point("human")
    
    # Add edges
    workflow.add_edge("human", "agent")
    workflow.add_edge("agent", "human")
    
    # Compile the graph
    return workflow.compile()


# Create and export the graph
conversation_graph = setup_graph()


# This allows for running the agent directly for testing
if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver
    
    # Initialize the graph with a MemorySaver for persistence
    graph = conversation_graph.with_checkpointer(MemorySaver())
    
    # Start the conversation with an initial config
    config = {"model": "gpt-4o", "temperature": 0.7}
    
    # Run the graph with the initial state
    for event in graph.stream({
        "messages": [],
        "config": config
    }):
        if "messages" in event:
            for message in event["messages"]:
                if isinstance(message, AIMessage):
                    print(f"AI: {message.content}")
                elif isinstance(message, HumanMessage):
                    print(f"Human: {message.content}")
