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
            for layer in self._layers[:-1]:
                x = layer.forward(x)
                
        return self._layers[-1].forward(x)
    
    def forward(self,x):
        '''
        Input is a numpy array 
        of normalized image 224 x 224
        split into 8x8 chunks
        '''
        
        return self._forward(x)
    
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
                
    if not os.path.exists("./dataset"):
        os.mkdir('./dataset')
        
    read_and_parse_images('/home/projectrobal/data/colosseo0_kitti/camera_left/data','./dataset/img')
        
    exit()
    
    # print(pcd.get_axis_aligned_bounding_box())
    
    # # We are intrested in spliting it into cubes of 1x1x1 meters 
    # # so the resolutions is 0.125 meter per point
    # grid_width = 256
    # grid_resolution = 0.125
    
    # cube_size = o3d.geometry.AxisAlignedBoundingBox()
    # cube_size.max_bound = np.ones(3)*(grid_width/2)
    # cube_size.min_bound = np.ones(3)*(-grid_width/2)
    
    # # cube_chunk = pcd.crop(cube_size)
    
    # grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    #     pcd,
    #     grid_resolution
    #     )
    
    # o3d.visualization.draw_geometries([grid])
    
    # exit()
    
    # split_point_cloud_into_chunks(grid,256.0,0.125)
    
    # exit()
    
    
    net = MapMemorizer(10)
    
    net = net.to(device=device)
    
    with torch.no_grad():
    
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