
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "I love learning artificial intelligence"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_length=50
    )

    decoder_input_ids = generated[:, :-1]

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        output_attentions=True
    )

translation = tokenizer.decode(generated[0], skip_special_tokens=True)

print("English Input:", text)
print("French Translation:", translation)

source_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
target_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])

attention = outputs.cross_attentions[-1][0].mean(dim=0).cpu().numpy()

plt.figure(figsize=(12, 7))

sns.heatmap(
    attention,
    xticklabels=source_tokens,
    yticklabels=target_tokens,
    cmap="viridis",
    annot=True,
    fmt=".2f"
)

plt.xlabel("English Input Tokens")
plt.ylabel("French Output Tokens")
plt.title("Attention Mechanism Visualization")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("Attention visualization completed successfully!")
