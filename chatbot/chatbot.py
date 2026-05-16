from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import warnings
warnings.filterwarnings("ignore")


# create the state 
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage] , add_messages]

chat_model = ChatOllama(model = "phi4-mini:latest")

def chat_node(state: ChatState):
    global chat_model
    # take user query from state
    messages = state['messages']
    # send query to llm 
    response = chat_model.invoke(messages)
    # store the reponse
    return {
        "messages": [AIMessage(response.content)]
    }

thread_id = "1"
checkpoint = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat_node" , chat_node)

graph.add_edge(START , "chat_node")
graph.add_edge("chat_node" , END)

chatbot = graph.compile(checkpointer = checkpoint)
config = {
    "configurable": {"thread_id" : thread_id}
}
while True:
    human_message = input("User:")
    if human_message.strip().lower() in ['exit', 'done', 'bye', 'quit']:
        break
    
    state = {
        "messages": [HumanMessage(content = human_message)]
    }
     
    result = chatbot.invoke(state , config = config)
    model_response = result['messages'][-1].content
    
    
    print(f"AI: {model_response}")