import torch
from typing import Dict, List, Tuple, Optional

class MR_volume:
    def __init__(self, data : torch.Tensor):
        self.data_chunks : List[torch.Tensor] = [data]
        self.scale_factors : List[float] = [1.0]
        self.corners : List[Tuple[int]] = [(0,0)]
        self.n_channels : int = data.shape[0]
        self.n_dims : int = len(data.shape[1:])
    
    def __str__(self) -> str:
        return "{ }" 
    
def MR_SR(input_mr_data : MR_volume) -> torch.Tensor:
    # to be implemented
    output_uniform_data = torch.zeros([1])
    return output_uniform_data

