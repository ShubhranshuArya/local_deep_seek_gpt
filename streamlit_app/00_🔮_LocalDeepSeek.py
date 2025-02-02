import streamlit as st
from utils.model_utils import load_model_and_processor


st.set_page_config(
    page_title="LocalDeepSeek",
    page_icon="🔮",
)

st.markdown(
    """
# 🔮 **LocalDeepSeek**

### **Local Image Generation and MultiModal Chatting with DeepSeek**

"""
)

vl_gpt, vl_chat_processor = load_model_and_processor()
