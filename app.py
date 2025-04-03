import streamlit as st
from data1 import run_career_agent

# Page Configurations
st.set_page_config(page_title="Skill Sync AI", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    .chat-box {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 10px;
        width: 75%;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .user-msg {
        background-color: #ECECEC;
        margin-left: auto;
        text-align: left;
    }
    .assistant-msg {
        background-color: #FFF0F5;
        margin-right: auto;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with centered logo
with st.sidebar:
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.image("career_logo.png", width=150)  # SkillSync AI logo
    st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("💡 Skill Sync AI")
    st.write("Get insights on skills, jobs, and learning paths.")
    st.divider()
    st.write("🔍 Ask anything about careers!")
    
    # Sample Prompts
    st.subheader("📌 Sample Prompts")
    sample_prompts = [
        "My name is Pallavi Pillai and roll no 28. Give course recommendations for Management Consultant.",
        "Now Give job recommendation for the role of Management Consultant.",
        "My name is Nitin Patel and roll no 41. Give skills required for Data Scientist."
    ]
    
    for prompt in sample_prompts:
        if st.button(prompt):
            st.session_state["query"] = prompt

# Title
st.markdown("<h1 style='text-align: center;'>🎓 Skill Sync AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Your AI guide for career growth.</p>", unsafe_allow_html=True)
st.divider()


# Initialize chat session state if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Input and Processing
user_input = st.chat_input("Type your question here...")
if user_input:
    # Append user's query
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process the user's query
    with st.spinner("Thinking..."):
        output = run_career_agent(user_input)
    st.session_state.messages.append({"role": "assistant", "content": output})

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-box user-msg'><b>User:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        # Render assistant messages with Markdown for clickable links
        st.markdown(f"<div class='chat-box assistant-msg'><b>SkillSync AI:</b> </div>", unsafe_allow_html=True)
        st.markdown(msg["content"], unsafe_allow_html=True)  # This ensures Markdown rendering



