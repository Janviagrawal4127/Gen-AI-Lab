
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "google/flan-t5-base"
DOCUMENT_PATH = "document.txt"

if not os.path.exists(DOCUMENT_PATH):
    with open(DOCUMENT_PATH, "w", encoding="utf-8") as file:
        file.write(
            "Artificial Intelligence is a branch of computer science. "
            "Machine Learning is a subset of Artificial Intelligence. "
            "Deep Learning uses neural networks to learn complex patterns. "
            "Natural Language Processing helps computers understand human language."
        )

with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
    document = file.read()

chunks = [
    document[i:i + 500]
    for i in range(0, len(document), 500)
]

vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(chunks)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def retrieve_context(question):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    top_indices = similarities.argsort()[-3:][::-1]

    context = " ".join(
        chunks[i] for i in top_indices if similarities[i] > 0
    )

    return context

def chatbot(question):
    context = retrieve_context(question)

    if not context:
        return "The answer is not available in the provided document."

    prompt = (
        "Answer the question using only the given context. "
        "If the answer is not present, say you do not know. "
        "\nContext: " + context +
        "\nQuestion: " + question +
        "\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer

print("Document-Based AI Chatbot")
print("Type 'exit' to stop the chatbot.")

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = chatbot(question)
    print("Chatbot:", response)
