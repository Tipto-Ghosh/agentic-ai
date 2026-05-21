import streamlit as st
from backend import chatFlow
import uuid
from langchain_core.messages import AIMessageChunk,HumanMessage,AIMessage

# generate a new thread id
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    # Generate a new thread ID and reset chat history
    new_thread_id = generate_thread_id()
    st.session_state["thread_id"] = new_thread_id
    
    if new_thread_id not in st.session_state["all_threads"]:
        st.session_state["all_threads"].append(new_thread_id)
    st.session_state["chat_history"] = []

def load_conversation(thread_id):
    config = {"configurable": {"thread_id": str(thread_id)}}
    state = chatFlow.get_state(config = config)
    
    if not state or not state.values:
        return []
    
    messages = state.values.get("messages" , [])
    history = []
    for msg in messages:
        if isinstance(msg , HumanMessage):
            role = "user"
        elif isinstance(msg , AIMessage):
            role = "assistant"
        else:
            continue
        
        history.append({
            "role": role , "content": msg.content
        })
    
    return history


st.title("Chatbot with Temporary Memory")
    
if "all_threads" not in st.session_state:
    st.session_state["all_threads"] = []


# initialize the current active thread id
if "thread_id" not in st.session_state:
    current_thread_id = generate_thread_id()
    st.session_state["thread_id"] = current_thread_id
    st.session_state["all_threads"].append(current_thread_id)

st.session_state["chat_history"] = load_conversation(st.session_state["thread_id"])
# sidebar ui
st.sidebar.title("Chat History")
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My chats")
# display threads in reverse order
for thread in reversed(st.session_state["all_threads"]):
    if thread == st.session_state["thread_id"]:
        st.sidebar.markdown(f"**Thread ID: {str(thread)} (Active)**")
    else:
        # inactive threads
        if st.sidebar.button(f"Load Thread {str(thread)}" , key = str(thread)):
            # load this thread
            st.session_state["thread_id"] = thread
            # get the chat history for this thread
            st.session_state["chat_history"] = load_conversation(thread)
            st.rerun()
              
# render all messages for the current active thread
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
user_input = st.chat_input("Type here")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # get the llm response
    config = {"configurable": {"thread_id" : str(st.session_state["thread_id"])}}
    
    # stream generator function
    def response_generator():
        stream_response = chatFlow.stream(
            input = {"messages": [("user"  , user_input)]},
            config = config,
            stream_mode = "messages"
        )
        for chunk , metadata in stream_response:
            if isinstance(chunk , (AIMessageChunk , AIMessage)) and chunk.content:
                yield chunk.content
                
    with st.chat_message("assistant"):
        ai_message = st.write_stream(response_generator())
    
    st.rerun()