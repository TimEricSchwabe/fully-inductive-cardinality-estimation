
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Optional, Union
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import random



class MultigraphLoader():
    # TODO: typing von object 
    def __init__(self, data: Union[object, List[object]], queries: Union[object, List[object]], batch_size: Union[int, List[int]] = 1, fair_chance: bool = True, shuffle: bool = False, **kwargs):
        if not isinstance(data, List):
            data = [data]
        if not isinstance(queries, List):
            queries = [queries]
        if not isinstance(batch_size, List):
            batch_size = [batch_size] * len(data)
        assert len(data) == len(queries) == len(batch_size), "All input lists must have the same length."

        self.data = data
        self.queries = queries
        self.batch_size = batch_size
        self.fair_chance = fair_chance
        self.shuffle = shuffle

        self.queries_loaders = []
        for i in range(len(self.data)):
            loader = DataLoader(
                dataset=self.queries[i],
                batch_size=self.batch_size[i],
                shuffle=self.shuffle,
            )
            self.queries_loaders.append(loader)

        self.queries_iters = []
        self.empty = []
        self.length = sum([len(ql) for ql in self.queries_loaders])

    def _get_random_loader(self):
        # use torch.multinomial with uniform weights to choose a loader index
        if not self.fair_chance: 
            weights = torch.ones(len(self.queries_iters), dtype=torch.float32)
            idx = int(torch.multinomial(weights, 1).item())
        else:
            idx = random.randint(0, len(self.queries_iters) - 1)

        if self.empty[idx]:
            idx = self._get_random_loader()

        return idx


    def __next__(self):
        # if no loaders available, stop
        if all(self.empty):
            raise StopIteration

        idx = self._get_random_loader()

        try:
            batch = next(self.queries_iters[idx])
        except StopIteration:
            # remove exhausted loader and continue
            self.empty[idx] = True
            if len(self.queries_iters) == 0:
                raise StopIteration
            else:
                return self.__next__()

        return self.data[idx], batch
    

    def __iter__(self):
        self.queries_iters = [iter(loader) for loader in self.queries_loaders]
        self.empty = [False] * len(self.queries_loaders)
        return self
    
    def __len__(self):
        return self.length