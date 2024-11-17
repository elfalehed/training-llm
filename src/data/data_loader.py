from datasets import load_dataset
from transformers import AutoTokenizer

class DataLoader:
    def __init__(self, model_name="gpt2", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def load_dataset(self, dataset_name, split="train"):
        dataset = load_dataset(dataset_name)[split]
        return dataset
    
    def preprocess(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
