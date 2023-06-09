#https://stackoverflow.com/questions/60993677/how-can-i-save-pytorchs-dataloader-instance

import torch
import tqdm
from lightning.fabric.utilities.types import _Stateful

class ResumableRandomSampler(torch.utils.data.Sampler, torch.nn.Module):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1]

    def __len__(self):
        return self.num_samples
    
    def state_dict(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def load_state_dict(self, state, strict=True):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])



# https://github.com/tqdm/tqdm/issues/73
class CustomStartProgressBar :
    '''
    TQDM progressbar, that allows to define initial state and description
    '''
    def __init__(self, total_blocks, init_blocks=0, description='') :
        self.total_blocks = total_blocks
        self._first = True
        self.init_blocks = init_blocks
        self._progress = tqdm.tqdm(total = total_blocks, desc=description)

    def __call__(self) :
        if self._first:
            self._progress.update(self.init_blocks)
            self._first = False
        else:
            self._progress.update()
