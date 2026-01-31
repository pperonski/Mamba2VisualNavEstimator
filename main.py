import torch
from fastkan import FastKANLayer
from mamba2 import Mamba2Simple

from timeit import default_timer as timer

import numpy as np
import os
import cv2

import open3d as o3d

device = torch.device("cuda")


'''

Idea is to split image into chunks and then 
pass them to Mamba Model, and then it will
generate more tokens which will represents
estimated map.  

Input image will have size of 224 x 224.

'''


class DatasetMemorizer:
    def __init__(self,image_dir:str,batch_size:int=32):
        super().__init__()
        
        self.image_dir = os.path.join(image_dir,"img")
        
        files = os.listdir(self.image_dir)
        
        # Split files into batches
        self._batch_size = batch_size
        self.batches = [
            files[i:i+batch_size] for i in range(0,len(files),batch_size)
        ]
                
        # load map with all points to memorize
        #
        # points format: x,y,z, point_class
        map_to_memorize = np.load(os.path.join(image_dir,"map.npy"))
        
        map_to_memorize = torch.tensor(map_to_memorize,dtype=torch.float32).flatten()
        
        numbers_count = int(map_to_memorize.shape[0])
        
        numbers_count_missing = numbers_count % 1024
        
        # append dummy values to make it divisible by 1024
        if numbers_count_missing != 0:
            padding_count = 1024 - numbers_count_missing
            
            padding = torch.zeros((padding_count,),dtype=torch.float32)
            
            map_to_memorize = torch.cat([map_to_memorize,padding],dim=0)
        
        self.map_to_memorize = map_to_memorize
        
        self.map_to_memorize = self.map_to_memorize.reshape((-1,1024))
        
    def batch_size(self):
        return self._batch_size
                
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self,idx:int):
        
        file_batch = self.batches[idx]
                
        images = []
        
        for file in file_batch:
            image = np.load(f"{self.image_dir}/{file}").reshape((-1,8*8))
            images.append(image)
                        
        image_batch = torch.tensor(images,dtype=torch.float32)
                
        return (image_batch,self.map_to_memorize)
    

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
            
        self.projection_point_layer = FastKANLayer(64,1024)
            
        self._layers = torch.nn.ModuleList([*layers,self.projection_point_layer])
        
        
    def _forward(self,x):
        
        with torch.no_grad():
            for layer in self._layers[:-2]:
                x = layer.forward(x)
                
        return self._layers[-2].forward(x)
    
    def forward(self,x):
        '''
        Input is a numpy array 
        of normalized image 224 x 224
        split into 8x8 chunks
        '''
        
        return self._layers[-1].forward(self._forward(x))
    
    def fit(self,epoches:int,dataset:DatasetMemorizer):
        
        self.train(True)
                
        optimizer = torch.optim.AdamW(self.parameters(),lr=0.001)
        
        loss_fn = torch.nn.MSELoss()
        
        mean_error = 0
                        
        for i in range(epoches):
            
            mean_error = 0
                        
            with torch.set_grad_enabled(True):
        
                for x,y in dataset:
                    
                    _x = x.to(device=device)
                    _y = y.to(device=device)
                                                            
                    output = self._forward(_x)
                    
                    while output.shape[1] < _y.shape[0]:
                        
                        __x = torch.cat([_x,output],dim=1)
                        
                        _out = self._forward(__x)
                        
                        output = torch.cat([output,_out],dim=1)
                        
                    output = self._layers[-1].forward(output)
                    
                    print(output.shape,_y.shape)
                    
                    loss = loss_fn(output,_y)
                    
                    optimizer.zero_grad()
                                    
                    loss.backward()
                    
                    mean_error += loss.item()
                    
                    optimizer.step()

            mean_error /= dataset.batch_size()
        
            print(f"Epoch: {i+1} loss: {mean_error}")
            
        self.train(False)

def convert_point_cloud_into_numpy_points_set(cloud:o3d.geometry.PointCloud):
    """
    Docstring for convert_point_cloud_into_numpy_points_set
    
    """
    
    points_set = []
    
    for point in cloud.points:
        points_set.append(
            (point[0],point[1],point[2],1.0)
        )
        
    points_set.append((0.0,0.0,0.0,255.0))
    
    points_set = np.array(points_set,dtype=np.float32)
    
    return points_set

def read_and_parse_images(image_dir:str,target_dir:str):
    
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    for i,file in enumerate(os.listdir(image_dir)):
        img = cv2.imread(f"{image_dir}/{file}")
        
        img_resize = cv2.resize(img,(224,224))
        image_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        
        image = image_rgb.reshape((-1,8,8))
        
        np.save(f"{target_dir}/img_{i}.npy",image)
        


def main():
    
    # test split points
    
    dataset = DatasetMemorizer("dataset")
    
    batch = dataset[0]
    
    print(batch[0].shape)
    print(batch[1].shape)
                
    net = MapMemorizer(10)
    
    net = net.to(device=device)
    
    net.fit(1,dataset)
    
    exit()
    
    with torch.no_grad():
        
        print("Test inference")
        
        out = net._forward(batch[0].to(device=device))
        
        _x = torch.cat([batch[0].to(device=device),out],dim=1)
        
        print(net._forward(_x).shape)
        
        exit()
    
        _input = torch.rand((1,784,8*8)).to(device=device)
        _input1 = torch.rand((1,784,8*8)).to(device=device)
        
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
    
    input()
    exit()
        
        
    x = [] 
    y = []
    
    for i in range(32):
        x.append(torch.rand((32,16,8*8)).to(device=device))
        y.append(torch.rand((32,16,8*8)).to(device=device))
    
    # Test train
    
    print("Test training")
    
    net.fit(1000,x,y)
        
    input()


if __name__ == "__main__":
    main()