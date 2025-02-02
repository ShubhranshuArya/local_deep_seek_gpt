import torch
import numpy as np
from PIL import Image


@torch.inference_mode()
def multimodal_understanding(
    vl_gpt,
    vl_chat_processor,
    image,
    question,
    seed,
    top_p,
    temperature,
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "Assistant", "content": ""},
    ]

    pil_image = Image.open(image).convert("RGB")
    prepared_inputs = vl_chat_processor(
        conversations=conversation, images=[pil_image], force_batchify=True
    ).to(
        "cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepared_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepared_inputs.attention_mask,
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
    )

    answer = vl_chat_processor.tokenizer.decode(
        outputs[0].cpu().tolist(), skip_special_tokens=True
    )
    return answer
