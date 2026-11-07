
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = Path(__file__).resolve().parent

tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
tokenizer.chat_template = (checkpoint / "chat_template.jinja").read_text()

model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
model.eval()

system_prompt = (checkpoint / "system_prompt.txt").read_text()

while True:
    text = input("\nYou: ")

    if text.lower() == "exit":
        break

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
