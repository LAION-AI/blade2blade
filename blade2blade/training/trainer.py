import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from blade2blade.training.utils import get_model, get_tokenizer
from blade2blade.training.custom_datasets.utils import get_dataset
from blade2blade.training.custom_datasets.prosocial import ProSocialCollator
from transformers import Trainer


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    if not os.path.exists(cfg.log_dir):
        os.mkdir(cfg.log_dir)

    if not cfg.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    if cfg.log_wandb:
        import wandb

        wandb.init(
            project="blade2blade",
            entity=cfg.wandb_entity,
            name=f"{cfg.model}-{cfg.log_dir}-rm",
            config=cfg,
        )

    model = get_model(cfg.model)
    tokenizer = get_tokenizer(cfg)

    training_args = instantiate(
        cfg.trainer, report_to="wandb" if cfg.log_wandb else None
    )
    train_dataset = get_dataset(cfg.train_dataset, tokenizer)
    validation_dataset = get_dataset(cfg.test_dataset, tokenizer)
    datacollator = ProSocialCollator(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=cfg.max_length,
        evil=cfg.evil,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=datacollator,
    )

    # training
    trainer.train()

    trainer.save_model(os.path.join(cfg.log_dir, f"{cfg.model.split('/')[-1]}-model"))
    tokenizer.save_pretrained(cfg.log_dir)


if __name__ == "__main__":
    train()
