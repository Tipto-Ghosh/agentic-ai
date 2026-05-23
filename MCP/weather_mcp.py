import asyncio
import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
import nest_asyncio
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama


class AgentState(TypedDict):
    messages: Annotated[list , add_messages]

async def main():
    # create the mcp client
    client = MultiServerMCPClient({
            "weather": {
                "transport": "stdio",
                "command": "D:/Tipto/agentic-ai/MCP/mcp-openweather/mcp-weather.exe",
                "args": [],
                "env": {"OWM_API_KEY" : os.getenv("OPEN_WEATHER_API")}
            }
        }
    )

    # check tools that we have on this server
    tools = await client.get_tools()
    
    # get the model
    # get the groq model
    # llm = ChatGroq(
    #     model = "llama-3.3-70b-versatile",
    #     temperature = 0.5, api_key =  os.getenv("GROQ_API")
    # )
    
    llm = ChatOllama(
        model = "qwen2.5:1.5b",
        temperature = 0.4
    )
    # bind the llm with tools
    llm_with_tools = llm.bind_tools(tools)
    
    
    async def call_model(state: AgentState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {
            "messages": [response]
        }
    
    graph = StateGraph(AgentState)
    graph.add_node("call_model" , call_model)
    graph.add_node("tools" , ToolNode(tools))
    
    graph.add_edge(START , "call_model")
    graph.add_conditional_edges("call_model" , tools_condition)
    graph.add_edge("tools" , "call_model")
    
    workflow = graph.compile()
    
    inputs = {
        "messages": [
            ("human", "Today's weather in Dhaka now.")
        ]
    }
    print("\n--- Agent Response --- \n")
    
    async for msg, metadata in workflow.astream(inputs, stream_mode = "messages"):
        if metadata.get("langgraph_node") == "call_model" and msg.content and not getattr(msg , "tool_calls" , None):
            print(msg.content, end = "", flush = True)
            
    print("\n\n--- Execution Complete ---")
                
        
if __name__ == "__main__": 
    asyncio.run(main())