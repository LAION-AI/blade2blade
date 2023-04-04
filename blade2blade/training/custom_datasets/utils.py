from typing import List
from omegaconf import OmegaConf
import torch

from blade2blade.training.custom_datasets.prosocial import ProSocialDataset


def convert_to_list(item):

    if not isinstance(item, List):
        item = [item]

    return item


def get_dataset(config, tokenizer):

    dataset_names = OmegaConf.to_object(config.name)
    dataset_names = convert_to_list(dataset_names)
    splits = OmegaConf.to_object(config.splits)
    splits = convert_to_list(splits)

    all_datasets = []
    for dataset, split in zip(dataset_names, splits):
        all_datasets.append(
            ProSocialDataset(
                dataset, tokenizer, split=split, confidence=config.confidence
            )
        )

    return torch.utils.data.ConcatDataset(all_datasets)
