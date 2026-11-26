
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, temperature, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response

prompt = "Artificial Intelligence is"

print("PROMPT:", prompt)

temperatures = [0.2, 0.7, 1.0]

for temp in temperatures:
    print("\nTemperature:", temp)
    print(generate_response(prompt, temp))

prompts = [
    "Explain Artificial Intelligence in simple words:",
    "Write a creative story about Artificial Intelligence:",
    "Explain Artificial Intelligence in a formal academic style:"
]

for p in prompts:
    print("\nPrompt:", p)
    print("Response:", generate_response(p, 0.7))

print("\nPrompt experimentation completed successfully!")

