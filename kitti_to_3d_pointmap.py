import sys
import numpy as np
import os
import pickle
import time

import open3d as o3d

from threading import Thread,Lock

class Position:
    def __init__(self):
        self._position = np.zeros(3)
        self._orientation = np.zeros(4)
        self._timestamp = 0.0
        
    def __str__(self):
        return f"postion: {self._position[0]}, {self._position[1]}, {self._position[2]} \
            quaterion: {self._orientation[0]}, {self._orientation[1]}, {self._orientation[2]}, {self._orientation[3]}"
        
    def read_from_line(self,line:str):
        
        data = line.split()
        
        t = float(data[0])
                
        x = float(data[1])
        y = float(data[2])
        z = float(data[3])
        
        q1 = float(data[4])
        q2 = float(data[5])
        q3 = float(data[6])
        q4 = float(data[7])
        
        self._timestamp = t
        self._position[0] = x
        self._position[1] = y
        self._position[2] = z
        
        self._orientation[0] = q4
        self._orientation[1] = q1
        self._orientation[2] = q2
        self._orientation[3] = q3

def rotate_point_numpy(point, quaternion):
    """
    Rotates a 3D point by a unit quaternion.
    quaternion: [w, x, y, z]
    point: [x, y, z]
    """
    
    # Extract scalar (w) and vector (vec_q) parts
    w = quaternion[0]
    vec_q = quaternion[1:]
    
    # Calculate the rotation using the optimized formula:
    # v' = v + 2w(vec_q x v) + 2(vec_q x (vec_q x v))
    t = 2.0 * np.cross(vec_q, point)
    v_rotated = point + w * t + np.cross(vec_q, t)
    
    return v_rotated

def quaternion_to_matrix(q):
    w, x, y, z = q
    # Pre-calculate products
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(y2 + z2),     2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(x2 + z2),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx), 1 - 2*(x2 + y2)]
    ])

def main():
    # timestamp 
    
    recordings_path = sys.argv[1]
        
    # timestamp file
    
    timestamp_file_path = sys.argv[2]
    
    # offset 
    
    if len(sys.argv) == 4:
        offset = int(sys.argv[3])
    else:
        offset = 0
    
    positions = []
    
    # get lidar files timestamps
    
    lidar_path = recordings_path+"/ouster_points/data"
    
    entries = os.listdir(lidar_path)

    # Filter to get only files
    cloud_files = [f for f in entries if os.path.isfile(os.path.join(lidar_path, f))]
    
    cloud_files = sorted(cloud_files)
    cloud_converer_file = os.path.join(lidar_path, cloud_files.pop(0))
    
    with open(cloud_converer_file,"rb") as file:
        cloud_converer_type = pickle.load(file)
        
    amount_of_clouds = len(cloud_files)
    
    print(f"Readed {len(cloud_files)} of points data")
    
    # read timestamps
    
    with open(timestamp_file_path) as file:
        lines = file.readlines()[offset:offset+amount_of_clouds+1]
        
        for line in lines:
            if line[0] == '#':
                continue
            pos = Position()
            pos.read_from_line(line)
            positions.append(pos)
                
    print(f"Readed {len(positions)} of positions.")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=800, width=600)
    
    BORDER_SIZE = 100 #2**32-1
    
    
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array(border_points))
        
    # pcd = o3d.io.read_point_cloud("point_cloud.ply",format="ply",print_progress=True,remove_nan_points=True,remove_infinite_points=True)
    # # pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pcd.points),3)))
    
    # bbox = o3d.geometry.AxisAlignedBoundingBox()
    
    # bbox.max_bound = np.array([1e16,1e16,1e16])
    # bbox.min_bound = np.array([-1e16,-1e16,-1e16])
    
    # pcd = pcd.crop(bbox)
    
   
    
    # o3d.visualization.draw_geometries([pcd])
    
    limits = o3d.geometry.AxisAlignedBoundingBox()
    limits.max_bound = np.ones(3)*100
    limits.min_bound = np.ones(3)*-100
    
    try:
        # render cloud map
        for i,(points_file,position) in enumerate(zip(cloud_files,positions)):
            cloud_numpy = np.fromfile(os.path.join(lidar_path,points_file), dtype=cloud_converer_type)
            # it is structured numpy array 
            cloud_points = o3d.geometry.PointCloud()
            points = []
                        
            print(f"Cloud: {i}")
            for point in cloud_numpy:
                
                if point['intensity'] <= 5 or point['range'] >= 1000000:
                    continue
                                
                p = np.array([point['x'],point['y'],point['z']],dtype=np.float32)
                
                # if abs(point['x']) < 0.1 and abs(point['y']) < 0.1 and abs(point['z']) >= 0.1:
                #     continue
                
                # if abs(point['z']) < 0.1 and abs(point['y']) < 0.1 and abs(point['x']) >= 0.1:
                #     continue
                
                # if abs(point['x']) < 0.1 and abs(point['z']) < 0.1 and abs(point['y']) >= 0.1:
                #     continue
                                
                if not np.isnan(np.sum(p)) and not np.isinf(np.sum(p)):
                    points.append(p)
            
            cloud_points.points = o3d.utility.Vector3dVector(np.array(points))
            rot_matrix = quaternion_to_matrix(position._orientation)
            cloud_points = cloud_points.rotate(R=rot_matrix,center=np.zeros(3))
            cloud_points = cloud_points.translate(position._position)
            
            pcd.points.extend(cloud_points.points)
            
            pcd = pcd.crop(limits)
            
            # if i % 10 == 0:
            #     pcd = pcd.remove_duplicated_points()
                            
    except KeyboardInterrupt:
        pass
        
    vis.add_geometry(pcd)
    
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
        
    while True:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)
    
    points = np.array(points,dtype=np.float32)
    
    print(points.shape)
    print(points)
    
    np.save("points",points)
    
    maxes = np.max(points,axis=0)
    mins = np.min(points,axis=0)
    
    print(f"Max x: {maxes[0]}, y: {maxes[1]}, z: {maxes[2]}")
    print(f"Min x: {mins[0]}, y: {mins[1]}, z: {mins[2]}")
    
    print(points.shape)
    
    
            
            
            
    
    
    
    

if __name__ == "__main__":
    main()