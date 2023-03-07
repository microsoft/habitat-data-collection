"""
    Given depth image and camera (matrix), generate point cloud data measurments.
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from math import pi

# todo: watch out cx cy in intrinsic matrix
def main():

    container_dir = "/datadrive/azure_storage/pactdata/habitat-data"
    data_dir = f"{container_dir}/collected-data/hm3d/train/0"
    rgb_path = f"{data_dir}/48-observations-rgb.png"
    depth_path = f"{data_dir}/48-observations-depth.npy"

    color = o3d.io.read_image(rgb_path)
    # depth = o3d.io.read_image(depth_path)
    depth = o3d.geometry.Image(
        np.load(depth_path) * 10
    )  # depth_scale=10.0 refs: https://github.com/facebookresearch/habitat-lab/issues/752#issuecomment-947225722

    # hfov = pi / 2  # default hfov is 90 in habitat
    # the chosen camera coordinate space is [-1,1], so W==2 ref:https://github.com/facebookresearch/habitat-sim/issues/1677
    # why it is 4d here?
    K = np.array(
        [
            [128.0, 0.0, 128.0],
            [0.0, 128.0, 128.0],
            [0.0, 0.0, 1],
        ]
    )  # depending on the image resolution
    cam = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    cam.intrinsic_matrix = K
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    plt.subplot(1, 2, 1)
    plt.title(" grayscale image")
    plt.imshow(rgbd.color)
    plt.subplot(1, 2, 2)
    plt.title("depth image")
    plt.imshow(rgbd.depth)
    plt.show()

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
