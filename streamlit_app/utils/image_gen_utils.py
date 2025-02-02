import torch
import numpy as np
from PIL import Image


@torch.inference_mode()
def generate(
    vl_gpt,
    vl_chat_processor,
    input_ids,
    width,
    height,
    temperature=1.0,
    parallel_size=5,
    cfg_weight=5.0,
    image_token_num_per_image=576,
    patch_size=16,
):
    cuda_device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cuda_device = "cuda"

    # Expand input tokens for conditional & unconditional branches
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(
        cuda_device
    )
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    # Convert tokens to embeddings
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int
    ).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True, past_key_values=pkv
        )
        pkv = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        # Classifier-Free Guidance
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        # Prepare the next image embeddings
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # Decode the image tokens
    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, width // patch_size, height // patch_size],
    )

    return generated_tokens.to(dtype=torch.int), patches


def unpack(decoded_patches, width, height, parallel_size=5):
    decoded_patches = (
        decoded_patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    )
    decoded_patches = np.clip((decoded_patches + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = decoded_patches
    return visual_img


@torch.inference_mode()
def generate_image(
    vl_gpt,
    vl_chat_processor,
    prompt,
    seed=None,
    guidance=5,
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    width = 384
    height = 384
    parallel_size = 5

    messages = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": ""},
    ]
    text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=messages,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    text += vl_chat_processor.image_start_tag
    tokenizer = vl_chat_processor.tokenizer

    input_ids = torch.LongTensor(tokenizer.encode(text))
    output, patches = generate(
        vl_gpt,
        vl_chat_processor,
        input_ids,
        width // 16 * 16,
        height // 16 * 16,
        cfg_weight=guidance,
        parallel_size=parallel_size,
    )

    images = unpack(patches, width // 16 * 16, height // 16 * 16)
    pil_images = [
        Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS)
        for i in range(parallel_size)
    ]
    return pil_images
