import sys
import math
import io
from typing import Dict, Any
import habitat
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from torchlightning_utils import cli
from habitat_simulation.agents import ShortestPathFollowerAgent
from habitat_simulation.collector import DatasetCollector
from habitat_simulation.envs import SimpleRLEnv
import matplotlib.pyplot as plt
from PIL import Image

cv2 = try_cv2_import()

# todo: implement metrics
# todo: pose and maps transform


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], output_size)


def map_centered_by_agent(info) -> np.ndarray:
    """Center the map wrt to agent's pose.

    Args:
        info (dict): _description_
    """
    env_map = info["top_down_map"]["map"]
    agent_map_pose = np.array(
        [
            info["top_down_map"]["agent_map_coord"][0],
            info["top_down_map"]["agent_map_coord"][1],
            info["top_down_map"]["agent_angle"],
        ]
    )
    cx = agent_map_pose[0]
    cy = agent_map_pose[1]
    map_size = min(info["top_down_map"]["map"].shape)
    rect = (
        (cx, cy),
        (map_size, map_size),
        90 - math.degrees(agent_map_pose[2]),
    )
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped_map = cv2.warpPerspective(env_map, M, (width, height))
    return warped_map


def get_2dpose(env: habitat.Env):
    """Return agent's pose in both simulation env and also wrt map.

    Args:
        env (habitat.Env): current environment

    Returns:
        dict:
            pose (np.ndarray): agent's pose wrt simulation env, 4d with 1-3d representing position, last entry represents yaw
    """
    agent_state = env._sim.get_agent_state()
    rotation_angles = Rotation.from_quat(
        quaternion.as_float_array(agent_state.rotation)
    ).as_euler("ZYX", degrees=False)
    pose = np.concatenate(
        [
            agent_state.position,
            [rotation_angles[1]],
        ]
    )

    return pose


def pltimg(img: np.ndarray) -> np.ndarray:
    """Given an image (not rgb), use plt to transform it image rgb image

    Args:
        img (np.ndarray): image with leading two dims should be width and height.

    Returns:
        np.ndarray: rgb repr of img with size (original w, original h, 3)
    """
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])  # remove margins
    plt.imshow(img)
    buf = io.BytesIO()
    plt.axis("off")
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB").resize((img.shape[0], img.shape[1]))
    buf.close()
    np_img = np.asarray(pil_img)
    return np_img


def make_image(observations, info):
    rgb_im = observations["rgb"]
    depth_im = pltimg(np.squeeze(observations["depth"], axis=2))
    semantic_im = pltimg(np.squeeze(observations["semantic"], axis=2))
    top_down_map = draw_top_down_map(info, rgb_im.shape[0])
    output_im = np.concatenate((rgb_im, depth_im, semantic_im, top_down_map), axis=1)
    return output_im


def make_offline_dataset(offline_dataset_config: dict, online_env_config: dict):
    image_feature = {
        "suffix": "png",
        "write_method": ["imageio", "imwrite"],  # module name, method name
        "read_method": ["imageio", "imread"],
        "path_first": True,
    }
    array_feature = {
        "suffix": "npy",
        "write_method": ["numpy", "save"],
        "read_method": ["numpy", "load"],
        "path_first": True,
    }
    heavy_features = {
        "rgb": image_feature,
        "depth": array_feature,
        "semantic": array_feature,
        "map": array_feature,
        "fog_of_war_mask": array_feature,
        "agent_map": array_feature,
    }
    offline_dataset_config["heavy_features"] = heavy_features
    offline_dataset = DatasetCollector(offline_dataset_config, online_env_config)
    return offline_dataset


# run episode
config_path = (
    "/home/azureuser/AutonomousSystemsResearch/habitat-simulation/configs/collect.yaml"
)
with open(config_path, mode="r") as f:
    from omegaconf import OmegaConf

    config = OmegaConf.create(f.read())
    config = OmegaConf.to_container(config, resolve=True)

offline_dataset_config, online_env_config = (
    config["offline_dataset"],
    config["online_env"],
)


dataset_collector = make_offline_dataset(offline_dataset_config, online_env_config)
env_config = habitat.get_config(config_paths=online_env_config["env_config_path"])

with habitat.config.read_write(env_config):
    env_config.habitat.dataset.split = online_env_config["env_split"]
    env_config.habitat.dataset.data_path = online_env_config["env_data_path"]
    env_config.habitat.dataset.scenes_dir = online_env_config["env_scene_dir"]
    env_config.habitat.simulator.scene_dataset = online_env_config[
        "env_scene_dataset_config"
    ]

env = habitat.Env(config=env_config)

for episode in range(20):
    observations = env.reset()
    while not env.episode_over:
        action = "move_forward"
        next_observations = env.step(action)
        print(next_observations["semantic"])
        break
env.close()
