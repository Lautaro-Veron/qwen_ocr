"""
Script with original notebook parameters (torch_dtype="auto")
To compare performance vs FP16
"""

from GetModelList import get_qwen_models
import huggingface_hub
from my_timer import MyTimer, my_timer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    print("🔍 Exploring available Qwen models...")
    get_qwen_models("Qwen")

    print("\n📥 Loading Qwen2.5-VL model with original parameters...")

    # Load model (exact parameters from original notebook)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype="auto",  # Original parameter
        device_map="cuda"
    )

    print("📥 Loading processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Configure image and prompt (use relative path)
    image = r".\test_images\factura_ejemplo.png"
    prompt = "Extract all text found on the image, including handwritten signatures"

    print(f"📷 Processing image: {image}")
    print(f"💬 Prompt: {prompt}")

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    print("🔧 Preparing data for inference...")

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Prepare image(s) or video for the model
    image_inputs, video_inputs = process_vision_info(messages)

    # Use the processor to prepare the input data
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move data to GPU
    inputs = inputs.to("cuda")
    model = model.to("cuda")

    print("🚀 Generating response...")

    # For Inference - generate output (original notebook parameters)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

    # Create iterator of tuples between input and generated ids
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode model output into human readable format
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("\n" + "="*60)
    print("📋 EXTRACTED RESULT (original parameters):")
    print("="*60)
    print(output_text[0] if output_text else "No text extracted")
    print("="*60)

if __name__ == "__main__":
    main()