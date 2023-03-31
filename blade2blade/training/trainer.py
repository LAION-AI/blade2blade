import os

import hydra
from training.custom_datasets.prosocial import ProSocialDataset,ProSocialCollator
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from training.utils import get_model, get_tokenizer
from transformers import Trainer 


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.save_folder):
        os.mkdir(cfg.save_folder)

    model = get_model(cfg.model)
    tokenizer = get_tokenizer(cfg)

    training_args = instantiate(cfg.trainer)

    train_dataset = ProSocialDataset(
        cfg.dataset.name, split=OmegaConf.to_object(cfg.dataset.train), tokenizer=tokenizer
    )
    validation_dataset = ProSocialDataset(
        cfg.dataset.name, split=OmegaConf.to_object(cfg.dataset.validation), tokenizer=tokenizer
    )
    datacollator = ProSocialCollator(tokenizer=tokenizer, padding="max_length",
                                       max_length=cfg.max_length)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=datacollator,
    )

    # Training
    trainer.train()

    trainer.save_model(os.path.join(cfg.save_folder, f"{cfg.model_name}-model"))


if __name__ == "__main__":
    train()
