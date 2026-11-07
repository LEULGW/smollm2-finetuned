
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = Path(__file__).resolve().parent

tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
tokenizer.chat_template = (checkpoint / "chat_template.jinja").read_text()

model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
model.eval()

system_prompt = (checkpoint / "system_prompt.txt").read_text()

with open("test.jsonl", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

with open("smollm2_predictions.jsonl", "w", encoding="utf-8") as output:

    for i, example in enumerate(dataset, 1):
        print(f"Testing {i}/{len(dataset)}: {example['id']}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]}
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        reply = tokenizer.decode(
            result[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        try:
            prediction = json.loads(reply)
        except json.JSONDecodeError:
            prediction = None

        output.write(json.dumps({
            "id": example["id"],
            "prediction": prediction
        }) + "\n")

        output.flush()

print("Testing complete!")
print("Predictions saved to smollm2_predictions.jsonl")
