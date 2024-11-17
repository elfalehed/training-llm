import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import wandb

class LLMTrainer:
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        
    def train(self, train_dataset, eval_dataset=None, output_dir="./results"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            save_steps=500,
            report_to="wandb"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
        
    def save_model(self, path):
        self.model.save_pretrained(path)
