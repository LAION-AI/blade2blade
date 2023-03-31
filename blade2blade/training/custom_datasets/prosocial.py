from attr import dataclass
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from typing import Union,List, Optional
import itertools
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from training.custom_datasets.utils import format_history



class ProSocialDataset(Dataset):
    """
    Dataset class to load dataset in alleai/prosocial format
    """

    def __init__(self,path:str, tokenizer:PreTrainedTokenizerBase, 
                 split:Union[List[str],str]="train",):

        super().__init__()
        
        dataset = load_dataset(path)
        if isinstance(split, List):
            self.split = "-".join(split)
            self.dataset = concatenate_datasets([dataset[sp] for sp in split])
        else:
            self.split = split
            self.dataset = dataset[split]

        self.tokenizer = tokenizer

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
            history = "".join(format_history(history,eos_token=self.tokenizer.eos_token))
            output =  self.dataset[idx]["safety_label"] + self.tokenizer.sep_token +\
                         self.tokenizer.sep_token.join(self.dataset[idx]["rots"]) +\
                         self.tokenizer.eos_token

            return history,output
    



@dataclass
class ProSocialCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None 
    truncation: Optional[bool] = True

    def __call__(self, examples):
        

        inputs = self.tokenizer([example[0] for example in examples],
                        max_length=self.max_length,
                        padding=self.padding,
                        pad_to_multiple_of = self.pad_to_multiple_of,
                        add_special_tokens=False,
                        truncation = self.truncation,
                        return_tensors="pt")
        
        output = self.tokenizer([example[1] for example in examples], add_special_tokens=False)

        return {"input":inputs,"output":output}




