import torch
from mamba2 import Mamba2Simple

from timeit import default_timer as timer

import numpy as np

device = torch.device("cuda")


'''

Idea is to split image into chunks and then 
pass them to Mamba Model, and then it will
generate more tokens which will represents
estimated map.  

Input image will have size of 224 x 224.

'''


class MapMemorizer(torch.nn.Module):
    
    def __init__(self,N:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        layers:list[Mamba2Simple] = []
        
        for n in range(N):
            
            layer =  Mamba2Simple(
                    d_model=8*8,
                    d_state=32,
                    d_conv=4,
                    expand=2,
                    use_mem_eff_path=False
                )
            
            layers.append(layer)
            
        self._layers = torch.nn.ModuleList(layers)
        
        
    def forward(self,x):
        
        with torch.no_grad():
            for layer in self._layers[:-1]:
                x = layer.forward(x)
        
        return self._layers[-1].forward(x)
    
    def fit(self,epoches:int,x,y):
        
        self.train(True)
                
        optimizer = torch.optim.AdamW(self.parameters(),lr=0.001)
        
        loss_fn = torch.nn.MSELoss()
        
        mean_error = 0
        
        batch_count:int = len(x)
                
        for i in range(epoches):
            
            mean_error = 0
            
            batch_size = 0
            
            with torch.set_grad_enabled(True):
        
                for batch_id in range(batch_count):
                    _x = x[batch_id]
                    _y = y[batch_id]
                    
                    batch_size = _x.shape[0]
                                        
                    output = self.forward(_x)
                    
                    loss = loss_fn(output,_y)
                    # loss.requires_grad = True
                    # loss.retain_grad()
                    
                    # for param in self.parameters():
                        # param.requires_grad = True
                        # param.retain_grad()
                        # print(param.is_leaf)
                    
                    optimizer.zero_grad()
                                    
                    loss.backward()
                    
                    mean_error += loss.item()
                    
                    optimizer.step()

            mean_error /= batch_size
        
            print(f"Epoch: {i+1} loss: {mean_error}")
            
        self.train(False)

            

def main():
    
    net = MapMemorizer(30)
    
    net = net.to(device=device)
    
    with torch.no_grad():
    
        _input = torch.rand((1,1,8*8)).to(device=device)
        _input1 = torch.rand((1,1,8*8)).to(device=device)
        
        start = timer()
        
        out = net(_input)
        
        end = timer()
        
        print(out.shape)
        print(f"Time in: {end-start} s")
        
        start = timer()
        
        out = net(_input1)
        
        end = timer()
        
        print(out.shape)
        print(f"Time in: {end-start} s")
        
        
    x = [torch.rand((32,16,8*8)).to(device=device),torch.rand((32,16,8*8)).to(device=device)] 
    
    y = [torch.rand((32,16,8*8)).to(device=device),torch.rand((32,16,8*8)).to(device=device)*10.0]
    
    # Test train
    
    print("Test training")
    
    net.fit(100,x,y)
        
    input()


if __name__ == "__main__":
    main()