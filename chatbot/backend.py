from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages


model = ChatOllama(model = "qwen2.5:1.5b") 

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]

def chatNode(state: ChatState):
    prompt = state["messages"]
    response = model.invoke(prompt)
    return {
        "messages": [response]
    }

checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chatNode" , chatNode)
graph.add_edge(START , "chatNode")
graph.add_edge("chatNode" , END)
chatFlow = graph.compile(checkpointer = checkpointer)