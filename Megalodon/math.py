import os
from datasets import load_dataset
import openai

api_key = os.getenv("OPENAI_API_KEY")

dataset = load_dataset("flax-sentence-embeddings/stackexchange_math_jsonl")



def preprocess_sample(sample):
    text = ''.join(str(sample[key]) for key in sample.keys())
    return text 


#generate an explanation for votes
def explain_votes(model, text):
    prompt = f"The text: {text} was upvoted/downvoted. Explain why thoroughly"
    response = model(prompt=prompt, max_tokens=60)
    return response.choice[0].text.strip()


#iterate for sample dataset:
for sample in dataset:
    sample_text = preprocess_sample(sample)
    explanation = explain_votes(openai.GPT, sample_text)
    print(f"Explanation: {explanation}")
