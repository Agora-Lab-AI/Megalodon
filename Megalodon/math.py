import os
from datasets import load_dataset
import openai

api_key = os.getenv("OPENAI_API_KEY")

dataset = load_dataset("flax-sentence-embeddings/stackexchange_math_jsonl")

def prep_sample(sample):
    question = sample["question"]
    multiple_choice_answer = sample["multiple_choice_answer"]
    answers = sample["answers"]
    image_id = sample["image_id"]
    answer_type = sample["answer_type"]
    question_id = sample["question_id"]
    image = sample["image"]

    text = f"Question: {question} Multiple Choice Answer: {multiple_choice_answer} Answers: {answers} Answer Type: {answer_type} Question ID: {question_id} Image ID: {image_id}"
    
    return {
        "image": image,
        "target_text": text
    }


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
