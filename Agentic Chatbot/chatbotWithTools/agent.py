import os
from typing import Annotated, Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pymongo import MongoClient
from typing_extensions import TypedDict
from memory import get_facts
from tools import tools

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API", "")


# Define Agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id:  Optional[str]

_llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 0.5)
_llm_with_tools = _llm.bind_tools(tools)


# Define Nodes 
def agent_node(state: AgentState) -> dict:
    """Main agent node: injects long-term memory then calls the LLM."""
    user_id = state.get("user_id", "default_user")
    facts = get_facts(user_id)
    memory_context = "\n".join(f"- {f}" for f in facts) if facts else "None yet."

    system = SystemMessage(
        content=(
            "You are a helpful assistant with memory.\n\n"
            f"What you know about this user:\n{memory_context}\n\n"
            "Use tools when you need current information (weather, web search).\n"
            "If the user shares something important about themselves, note it."
        )
    )
    response = _llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}

# Routing for tools
def should_use_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "__end__"

def build_graph():
    """Compile and return the LangGraph workflow with a MongoDB checkpointer."""
    tool_node = ToolNode(tools=tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_use_tools,
        {"tools": "tools", "__end__": END},
    )
    graph.add_edge("tools", "agent")

    mongo_client  = MongoClient(os.getenv("MONGO_URI"))
    checkpointer  = MongoDBSaver(
        client = mongo_client,
        db_name = os.getenv("MONGO_DB", "chatbot_db")
    )
    return graph.compile(checkpointer = checkpointer)