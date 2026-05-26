import os 
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list , add_messages]

async def main():
    groq_api = os.getenv("GROQ_API")
    model = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature = 0.0,
        api_key = groq_api
    )
    
    # create mcp client
    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp"
            }
        }
    )
    tools = await client.get_tools()
    # print(tools)
    
    tool_node = ToolNode(tools)
    model_with_tools = model.bind_tools(tools)
    
    # create a rounting function to call tools
    def tool_route(state: AgentState):
        # extract the messages from the state
        messages = state["messages"]
        # extract the last message
        last_message = messages[-1]
        
        if hasattr(last_message , "tool_calls") and last_message.tool_calls:
            return "tools" # tools node name in the graph
        
        return END 
    
    # create a function for model node 
    async def model_node(state: AgentState):
        # extract the messages from the state
        messages = state["messages"]
        
        response = await model_with_tools.ainvoke(messages)
        return {
            "messages": [response]
        }
        
    graph = StateGraph(AgentState)
    
    graph.add_node("model_node" , model_node)
    graph.add_node("tools" , tool_node)
    
    graph.add_edge(START , "model_node")
    graph.add_conditional_edges(
        "model_node", tool_route,
        {
            "tools": "tools", END : END
        }
    )
    
    graph.add_edge("tools" , "model_node")
    
    workflow = graph.compile()
    
    final_state = await workflow.ainvoke(
        {
            "messages": [
                SystemMessage(
                    content=(
                        "You are a helpful weather assistant. "
                        "Always use the tool results to answer the user directly."
                        "Based the result that you are getting from the tool, make a small report."
                    )
                ),
                ("human", "What's the weather in Dhaka today?")
            ]
        }
    )
    
    messages = final_state["messages"]
    final_response = messages[-1].content
    print(final_response)
    
if __name__ == "__main__":
    asyncio.run(main())