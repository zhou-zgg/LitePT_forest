import os
import numpy as np
import laspy
import open3d as o3d

def pcd_to_las(pcd_path, las_path, label_value=0):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.label = np.full(len(points), label_value, dtype=np.float64)

    las.write(las_path)
    print(f"Converted: {pcd_path} -> {las_path} ({len(points)} points)")

def convert_inference_dir():
    pcd_dir = "/home/caozhou/workshop/deep_learning/LitePT/data/forest/Inference"
    las_dir = pcd_dir

    for f in os.listdir(pcd_dir):
        if f.endswith(".pcd"):
            pcd_path = os.path.join(pcd_dir, f)
            las_name = f.replace(".pcd", ".las")
            las_path = os.path.join(las_dir, las_name)
            pcd_to_las(pcd_path, las_path)
            return las_dir

    print(f"No .pcd files found in {pcd_dir}")
    return None

if __name__ == "__main__":
    convert_inference_dir()