import streamlit as st

from utils.model_utils import load_model_and_processor
from utils.multimodal_utils import multimodal_understanding

st.set_page_config(
    page_title="NotSoDeepSeek",
    page_icon="🔮",
)

#####Sidebar Start#####

st.sidebar.markdown("## **Parameter Settings**")

seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)
top_p = st.sidebar.slider(
    "Top_P",
    min_value=0.0,
    max_value=1.0,
    value=0.95,
    step=0.05,
)
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
)

#####Sidebar End#####


#####Main Layout Start#####

st.title("Talk to your image")

st.subheader("Upload an Image")
uploaded_image = st.file_uploader(
    "",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

vl_gpt, vl_chat_processor = load_model_and_processor()

if uploaded_image:
    st.image(uploaded_image, use_container_width=True)

question = st.text_input("Question", value="Explain this meme...")

if st.button("Chat"):
    if not uploaded_image:
        st.warning("Please upload an image before chatting.")
    else:
        with st.spinner("Analyzing your image..."):
            answer = multimodal_understanding(
                vl_gpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                image=uploaded_image,
                question=question,
                seed=seed,
                top_p=top_p,
                temperature=temperature,
            )
        st.text_area("Response", value=answer, height=150)

#####Main Layout End#####
