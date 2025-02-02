import streamlit as st

from utils.image_gen_utils import generate_image
from utils.model_utils import load_model_and_processor

st.set_page_config(
    page_title="NotSoDeepSeek",
    page_icon="🔮",
)

#####Sidebar Start#####

st.sidebar.markdown("## **Parameter Settings**")

seed_t2i = st.sidebar.number_input(
    "Seed (Optional)",
    min_value=0,
    value=12345,
    step=1,
)
cfg_weight = st.sidebar.slider(
    "CFG Weight",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
)

#####Sidebar End#####


#####Main Layout Start#####

st.title("Create Images From Text")
prompt = st.text_area(
    "Prompt",
    value="A cute baby fox in autumn leaves, digital art, cinematic lighting...",
)

vl_gpt, vl_chat_processor = load_model_and_processor()

if st.button("Generate Images"):
    with st.spinner("Generating images... This may take a minute."):
        images = generate_image(
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            prompt=prompt,
            seed=seed_t2i,
            guidance=cfg_weight,
        )
    st.write("Generated Images:")
    cols = st.columns(2)
    idx = 0
    for i in range(2):  # 2 rows
        for j in range(2):  # 2 cols
            if idx < len(images):
                with cols[j]:
                    st.image(images[idx], use_container_width=True)
            idx += 1

# Tips / example prompts
with st.expander("Example Prompts"):
    st.write(
        "1. A cyberpunk samurai meditating in a neon-lit Japanese garden, cherry blossoms falling."
    )
    st.write(
        "2. A magical library with floating books, ethereal lighting, dust particles in the air, hyperrealistic detail."
    )
    st.write(
        "3. A steampunk-inspired coffee machine with brass gears and copper pipes, Victorian era style, morning light."
    )

#####Main Layout End#####
