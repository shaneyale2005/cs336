import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Type, Any

class SharedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.all_params = list(params)
        self.param_to_global_idx = {id(p): i for i, p in enumerate(self.all_params)}

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size
        else:
            self.rank = 0
            self.world_size = 1
        
        self.param_shard = self.all_params[self.rank::self.world_size]
        self.local_optimizer = optimizer_cls(self.param_shard, **kwargs)
        super().__init__(self.local_optimizer.param_groups, self.local_optimizer.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.local_optimizer.step(closure)
        if self.world_size > 1:
            for param in self.all_params:
                param_idx = self.param_to_global_idx[id(param)]
                owner_rank = param_idx % self.world_size
                dist.broadcast(param.data, src=owner_rank)
        return loss

