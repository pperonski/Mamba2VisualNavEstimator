import sys
import numpy as np
import os
import pickle

from scipy.spatial import cKDTree


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
        
        self._orientation[0] = q1
        self._orientation[1] = q2
        self._orientation[2] = q3
        self._orientation[3] = q4

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
            
    points = []
    
    try:
        # render cloud map
        for i,(points_file,position) in enumerate(zip(cloud_files,positions)):
            cloud_numpy = np.fromfile(os.path.join(lidar_path,points_file), dtype=cloud_converer_type)
            # it is structured numpy array 
            print(f"Cloud: {i}")
            for point in cloud_numpy:
                
                p = np.array([point['x'],point['y'],point['z']],dtype=np.float32)
                
                p = rotate_point_numpy(p,position._orientation)
                p += position._position
                                
                if not np.isnan(np.sum(p)) and not np.isinf(np.sum(p)):
                    if len(points) == 0:
                        points.append(p)
                    else:
                        tree = cKDTree(points)
                        
                        dist, index = tree.query(p)
                        
                        if dist >=  0.001:
                            points.append(p)
                            
    except KeyboardInterrupt:
        pass
    
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