from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int = 128
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "./results"
    logging_dir: str = "./logs"
