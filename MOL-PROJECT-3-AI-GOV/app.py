import streamlit as st
from datetime import datetime
import pandas as pd
from evedently_demo import get_response, upload_report 

st.set_page_config(
    page_title="MOL IT – AI Governance",
    layout="wide"
)

# ---------- NAVY & WHITE THEME CSS ----------
st.markdown(
    """
    <style>
    /* 1. Narrow the entire body & Background */
    .main .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 10rem;
    }
    
    /* 2. Heading Styles */
    .main-header {
        text-align: center;
        color: #002147; /* Navy Blue */
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 3rem;
    }

    /* 3. Catchy Feature Blocks */
    .feature-box {
        background: White;
        padding: 14px 16px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 160px;
        width: 350px;
        text-align: center;
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
        border-color: #002147;
        box-shadow: 0 10px 15px -3px rgba(0, 33, 71, 0.2);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    .feature-title {
        color: #002147;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* 4. Chat Input Styling */
    div[data-testid="stChatInput"] {
        max-width: 1500px;
        margin: 0 auto;
        background-color: white;
        border-radius: 15px;
    }

    /* Fix for scrolling overlap */
    .stChatMessage {
        background-color: #f8fafc !important;
        border-radius: 10px !important;
        border: 1px solid #f1f5f9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- MAIN UI ----------

# Centered Navy Heading
st.markdown("<h1 class='main-header'>🛡️ MOL IT – AI Governance</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Securing the future of maritime intelligence</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- FEATURE BLOCKS ----------
if not st.session_state.messages:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('''
            <div class="feature-box">
                <div class="feature-icon">📦</div>
                <div class="feature-title">Shipping</div>
                <div style="font-size: 0.85rem; color: #475569;">Compliance, logistics, and maritime policy.</div>
            </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown('''
            <div class="feature-box">
                <div class="feature-icon">🔐</div>
                <div class="feature-title">Safety</div>
                <div style="font-size: 0.85rem; color: #475569;">Risk mitigation and governance audits.</div>
            </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown('''
            <div class="feature-box">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Reporting</div>
                <div style="font-size: 0.85rem; color: #475569;">Data monitoring and dashboard logs.</div>
            </div>
        ''', unsafe_allow_html=True)

# ---------- CHAT LOGIC ----------
# Container for messages to ensure they stay within the narrowed width
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("How can I assist you today?")

if user_input:
    # 1. Immediately show the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. Show a placeholder for the assistant while generating
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            # Execute your backend functions
            model_response, final_context, user_question, current_time, time_taken, model_used, input_tokens_used, output_tokens_used, total_tokens_used = get_response(user_input)
            upload_report(model_response, final_context, user_question, current_time, time_taken, model_used, input_tokens_used, output_tokens_used, total_tokens_used)
            
            # Display the final response
            st.markdown(model_response)
    
    # 3. Save to history and rerun to lock the state
    st.session_state.messages.append({"role": "assistant", "content": model_response})
    st.rerun()