import os 
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END 
from langgraph.checkpoint.postgres import PostgresSaver
 
load_dotenv()
groq_api = os.getenv("GROQ_API")

# Define the state
class State(TypedDict):
    messages: Annotated[list[BaseMessage] , add_messages]
    summary: Optional[str]

model = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.0,
    api_key = groq_api
)

# how many last message we want to keep
keep_last_n = 3

# Format: postgresql://user:password@host:port/dbname
POSTGRES_URI = "postgresql://postgres:postgres@localhost:5442/postgres"

# chat node
def chat_node(state: State):
    """
    This is the main chat node or agent node.
    
    - If we have summary then we need to pass summary as System message + current message
    - If we don't have any summary then just pass the current messages
    """
    summary = state.get("summary" , "")
    messages = state["messages"]
    
    # if we have summary
    if summary:
        system_message = SystemMessage(
            content = (
                "You are a helpful assistant. "
                "Here is a summary of the conversation so far — "
                "use it as background context:\n\n"
                f"{summary}"
            )
        )
        input_messages = [system_message] + messages 
    else:
        input_messages = messages
    
    # now call the llm
    response = model.invoke(input_messages)
    return {
        "messages": [response]
    }


# Summarize node
def summarize(state: State):
    """
    Summarize node -> triggered after keep_last_n messages accumulate.
    """
    messages = state["messages"]  
    summary = state.get("summary" , "")
    
    # Determine which messages to summarize (all except last keep_last_n)
    if len(messages) <= keep_last_n:
        # No need to summarize
        return {}
    
    messages_to_summarize = messages[ : -keep_last_n]
    
    # Build the summarization prompt
    if summary:
        summary_prompt = (
            f"This is the summary of the conversation so far:\n{summary}\n\n"
            "Extend the summary by incorporating the new messages below. "
            "Be concise — capture key facts, decisions, and context only.\n\n"
            f"Messages to incorporate:\n{messages_to_summarize}"
        )
    else:
        summary_prompt = (
            "Summarize the following conversation. "
            "Be concise — capture key facts, decisions, and context only.\n\n"
            f"Messages to summarize:\n{messages_to_summarize}"
        )
    
    # Get new summary from LLM
    new_summary_msg = model.invoke([HumanMessage(content=summary_prompt)])
    new_summary = new_summary_msg.content
    
    # Create RemoveMessage objects for messages to delete
    messages_to_delete = []
    for msg in messages_to_summarize:
        if getattr(msg , 'id' , None) and msg.id:
            messages_to_delete.append(RemoveMessage(id = msg.id))
    
    return {
        "messages": messages_to_delete,
        "summary": new_summary
    }
    
# rounting function which will decide to summarize or not
def should_summarize(state: State):
    messages = state["messages"]
    if len(messages) > keep_last_n:
        return "summarize" # summarize node name
    
    return END 

def build_graph(checkpointer: PostgresSaver):
    graph = StateGraph(State)
    
    # add nodes
    graph.add_node("chat" , chat_node)
    graph.add_node("summarize" , summarize)
    
    # add edges
    graph.add_edge(START , "chat")
    graph.add_conditional_edges(
        "chat" , should_summarize,
        {"summarize" : "summarize" , END : END}
    )
    graph.add_edge("summarize" , END)
    
    workflow = graph.compile(checkpointer = checkpointer)
    return workflow

# globally declare the checkpointer
_cm = PostgresSaver.from_conn_string(POSTGRES_URI)
checkpointer = _cm.__enter__()
checkpointer.setup()

def get_graph():
    """Returns a compiled graph linked to the live checkpointer pool."""
    return build_graph(checkpointer)

import uuid
def run_chat(thread_id: str | None = None):
    print("[DEBUG] entered run_chat")
    graph = get_graph()
    print("[DEBUG] graph loaded")
    # Reuse a thread_id to resume a previous conversation,
    # or generate a new one to start fresh.
    if thread_id is None:
        thread_id = str(uuid.uuid4())
 
    config = {"configurable": {"thread_id": thread_id}}
 
    print(f"\n{'='*15}")
    print(f"  Chatbot  |  thread_id: {thread_id}")
    print(f"  (type 'quit' to exit, 'show state' to inspect)\n")
    print(f"{'='*15}\n")
    
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
    
            # Debug helper 
            if user_input.lower() == "show state":
                snapshot = graph.get_state(config)
                state = snapshot.values
                print("\n---- Current State ----------------------------------")
                print(f"  Messages in context : {len(state.get('messages', []))}")
                print(f"  Summary exists      : {bool(state.get('summary', ''))}")
                if state.get("summary"):
                    print(f"\n  Summary:\n  {state['summary']}\n")
                print("---------------------------------------------------------\n")
                continue
    
            # Normal turn
            result = graph.invoke(
                {"messages": [HumanMessage(content = user_input)]},
                config = config,
            )
    
            # The last message is always the assistant's reply
            ai_message = result["messages"][-1]
            print(f"\nAssistant: {ai_message.content}\n")
    
            # Show a hint when summarization just fired
            snapshot = graph.get_state(config)
            if snapshot.values.get("summary"):
                msg_count = len(snapshot.values.get("messages", []))
                print(f"  [Summary active — {msg_count} messages in context]\n")
    finally:
        _cm.__exit__(None, None, None)
                    
if __name__ == "__main__":
    print("Starting....")
    run_chat(thread_id = "0d4193bf-c3c6-4dc0-857b-8134ec2c2722")