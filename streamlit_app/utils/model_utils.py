import torch
from transformers import AutoConfig, AutoModelForCausalLM
import streamlit as st
from janus.models import VLChatProcessor


@st.cache_resource
def load_model_and_processor(model_path="deepseek-ai/Janus-1.3B"):
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = "eager"

    vl_gpt_model = AutoModelForCausalLM.from_pretrained(
        model_path, language_config=language_config, trust_remote_code=True
    )
    vl_gpt_model = vl_gpt_model.to(
        torch.bfloat16 if torch.cuda.is_available() else torch.float16
    )
    if torch.cuda.is_available():
        vl_gpt_model = vl_gpt_model.cuda()

    vl_chat_proc = VLChatProcessor.from_pretrained(model_path)
    return vl_gpt_model, vl_chat_proc
