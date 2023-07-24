import os
from datasets import load_dataset
import openai
from datasets import load_dataset, DatasetDict, Dataset

from Megalodon.models import OpenAILanguageModel, HuggingFaceLLM

class Megalodon:
    def __init__(self, model_id: str, api_key: str = None, dataset: str = None, prompt: str = None):
        self.dataset = load_dataset(dataset) if dataset else None
        self.prompt = prompt if prompt else "This text: {text} was upvoted/downvoted. Explain why in a simple, comprehensive manner."
        
        if "gpt" in model_id.lower():
            self.model = OpenAILanguageModel(api_key, api_model=model_id)
        else:
            self.model = HuggingFaceLLM(model_id)

    def preprocess_sample(self, sample):
        text = ' '.join(str(sample[key]) for key in sample.keys())
        return text 

    def explain_votes(self, text):
        prompt = self.prompt.format(text=text)
        explanation = self.model.generate_text(prompt, 1)[0]
        return explanation

    def run(self):
        if not self.dataset:
            raise ValueError("Dataset not provided.")
        explanations = []
        for sample in self.dataset:
            sample_text = self.preprocess_sample(sample)
            explanation = self.explain_votes(sample_text)
            explanations.append(explanation)
            print(f"Explanation: {explanation}")
        return explanations

    def save_to_huggingface(self, explanations, output_dir):
        data = Dataset.from_dict({'explanations': explanations})
        data.save_to_disk(output_dir)



    


# TODO: Make it modular to plug in and play with any model openai or huggingface
# TODO: Save to huggingface after each iteration is labelled
# TODO: Potentially use parquet for optimized storage
# TODO: Add in polymorphic or shape shifting preprocesing logi 
# # Using OpenAI model
# megalodon = Megalodon(model_id="gpt-3", api_key="your-api-key", dataset="flax-sentence-embeddings/stackexchange_math_jsonl")
# explanations = megalodon.run()
# megalodon.save_to_huggingface(explanations, 'hf_output_dir')

# # Using Hugging Face model
# megalodon = Megalodon(model_id="gpt2", dataset="flax-sentence-embeddings/stackexchange_math_jsonl")
# explanations = megalodon.run()
# megalodon.save_to_huggingface(explanations, 'hf_output_dir')
