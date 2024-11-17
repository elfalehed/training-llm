import wandb
from src.data.data_loader import DataLoader
from src.model.trainer import LLMTrainer
from src.utils.metrics import calculate_perplexity
from src.config.config import TrainingConfig

def main():
    # Initialize config
    config = TrainingConfig()
    
    # Initialize wandb
    wandb.init(project="llm-training", config=vars(config))
    
    # Initialize data loader
    data_loader = DataLoader(config.model_name, config.max_length)
    
    # Load and preprocess dataset
    train_dataset = data_loader.load_dataset(config.dataset_name, split="train")
    eval_dataset = data_loader.load_dataset(config.dataset_name, split="validation")
    
    # Preprocess datasets
    train_encodings = data_loader.preprocess(train_dataset)
    eval_encodings = data_loader.preprocess(eval_dataset)
    
    # Initialize trainer
    trainer = LLMTrainer(config.model_name)
    
    # Train model
    trainer.train(train_encodings, eval_encodings, config.output_dir)
    
    # Calculate and log final perplexity
    perplexity = calculate_perplexity(trainer.model, eval_encodings, trainer.device)
    wandb.log({"final_perplexity": perplexity})
    
    # Save the model
    trainer.save_model(f"{config.output_dir}/final_model")
    
if __name__ == "__main__":
    main()
