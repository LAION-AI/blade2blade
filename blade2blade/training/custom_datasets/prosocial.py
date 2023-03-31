from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from typing import Union,List
import itertools


class ProSocialDataset(Dataset):

    def __init__(self,path:str, split:Union[List[str],str]="train"):

        super().__init__()
        
        dataset = load_dataset(path)
        if isinstance(split, List):
            self.split = "-".join(split)
            self.dataset = concatenate_datasets([dataset[sp] for sp in split])
        else:
            self.split = split
            self.dataset = dataset[split]

        

    def __len__(self):
            return len(self.dataset)
        
        
    def __getitem__(self, idx):

            idx_start = idx
            end = self.dataset[max(0, idx_start - 1)]["episode_done"]
            while (not end) and (idx_start > 0):
                end = self.dataset[max(0, idx_start - 2)]["episode_done"]
                idx_start -= 1
            idx_start = max(0, idx_start)

            history = [
                (self.dataset[i]["context"],self.dataset[i]['response']) for i in range(idx_start, idx)
            ]
            history = list(itertools.chain(*history))
            history.append(self.dataset[idx]["context"])
            rots = self.dataset[idx]["rots"]
            label = self.dataset[idx]["safety_label"]

            return history,rots,label
    












