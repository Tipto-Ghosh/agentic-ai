from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

model = ChatOllama(model = "qwen2.5:1.5b") 

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]

def chatNode(state: ChatState):
    prompt = state["messages"]
    response = model.invoke(prompt)
    return {
        "messages": [response]
    }

connection = sqlite3.connect(database = "chatbot.db" , check_same_thread = False)
checkpointer = SqliteSaver(conn = connection)

graph = StateGraph(ChatState)
graph.add_node("chatNode" , chatNode)
graph.add_edge(START , "chatNode")
graph.add_edge("chatNode" , END)
chatFlow = graph.compile(checkpointer = checkpointer)