"""Perform point navigation task in habitat(-lab)

refs: /examples/shortest_path_follower_example.py
"""

import sys
import math
import io
from tqdm import tqdm
from typing import Dict, Any, List, Tuple 
import habitat
import numpy as np
import itertools
import quaternion
from scipy.spatial.transform import Rotation
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from torchlightning_utils import cli
from agents import ShortestPathFollowerAgent
from collector import DatasetCollector
from envs import SimpleEnv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

cv2 = try_cv2_import()


def draw_top_down_map(info, output_size):
    # top_down_map here is a dictionary that contains maps, agent coordinates, fog_mask. We can use fog_mask and map to colorize the top_down_map
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


def display_obs(obs: np.ndarray, obs_type: str) -> np.ndarray:
    """Given an image (not rgb), use plt to transform it image rgb image

    Args:
        obs (np.ndarray): observations from sensors with leading two dims should be width and height.

    Returns:
        np.ndarray: rgb repr of img with size (original w, original h, 3)
    """
    if obs_type == "semantic":
        obs_img = Image.new("P", (obs.shape[1], obs.shape[0]))
        obs_img.putpalette(d3_40_colors_rgb.flatten())
        obs_img.putdata((obs.flatten() % 40).astype(np.uint8))
        obs_img = obs_img.convert("RGBA")
    elif obs_type == "depth":
        obs_img = Image.fromarray((obs / 10 * 255).astype(np.uint8), mode="L")
    else:
        raise NotImplementedError("observation type not supported for display!")
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])  # remove margins
    plt.imshow(obs_img)
    buf = io.BytesIO()
    plt.axis("off")
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB").resize((obs.shape[1], obs.shape[0]))
    buf.close()
    np_img = np.asarray(pil_img)
    return np_img


def make_image(observations, info):
    images = []
    rgb_im = observations["rgb"]
    depth_im = display_obs(np.squeeze(observations["depth"], axis=2), "depth")
    semantic_im = display_obs(np.squeeze(observations["semantic"], axis=2), "semantic")
    equirect_rgb_im = observations["equirect_rgb"]
    equirect_depth_im = display_obs(np.squeeze(observations["equirect_depth"], axis=2), "depth")
    equirect_semantic_im = display_obs(np.squeeze(observations["equirect_semantic"], axis=2), "semantic")
    top_down_rgb = observations["top_down_rgb"]
    top_down_semantic = display_obs(np.squeeze(observations["top_down_semantic"], axis=2), "semantic")
    # top_down_map = draw_top_down_map(observations, rgb_im.shape[0])
    # output_im = np.concatenate((rgb_im, depth_im, semantic_im, top_down_map), axis=1)
    top_down_map = cv2.resize(
        draw_top_down_map(info, rgb_im.shape[1]),
        (rgb_im.shape[1], rgb_im.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    fig = plt.figure(figsize=(16., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                 axes_pad=0.0,  # pad between axes in inch.
                 )
    images = [rgb_im, depth_im, semantic_im,
            equirect_rgb_im,equirect_depth_im ,equirect_semantic_im, 
            top_down_rgb, top_down_semantic, top_down_map]
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.axis("off")
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB").resize((rgb_im.shape[1]*4, rgb_im.shape[0]*4))
    buf.close()
    output_img = np.asarray(pil_img)
    # output_img = np.concatenate((rgb_im, depth_im, semantic_im,equirect_rgb_im,equirect_depth_im ,equirect_semantic_im, top_down_semantic, top_down_map), axis=1)
    return output_img


def run_pointnav_episodes(
    offline_dataset_config: Dict[str, Any],
    online_env_config: Dict[str, Any],
    agent_config: Dict[str, Any],
):
    dataset_collector = DatasetCollector(offline_dataset_config, online_env_config)
    env_config = habitat.get_config(config_paths=online_env_config["env_config_path"])
    with habitat.config.read_write(env_config):
        env_config.habitat.dataset.split = online_env_config["env_split"]
        env_config.habitat.dataset.data_path = online_env_config["env_data_path"]
        env_config.habitat.dataset.scenes_dir = online_env_config["env_scene_dir"]
        env_config.habitat.simulator.scene_dataset = online_env_config[
            "env_scene_dataset_config"
        ]

    # start simulation
    with SimpleEnv(config=env_config) as env:
        # override episodes iterator in env to skip episodes in dataset
        # simple agent from native habitat
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = env_config.habitat.simulator.forward_step_size
        # agent = PACTPointNavAgent(
        #     raw_action_space=env.habitat_env.action_space, agent_config=agent_config
        # )
        agent = ShortestPathFollowerAgent(env.sim, goal_radius)
        print("Environment creation successful") 

        pbar = tqdm(range(0, len(env.episodes)), desc="Episodes") 
        for episode in pbar:
            observations = env.reset()
            dataset_collector.episode_start(env.current_episode.__getstate__())

            # RLEnv.reset() only returns observations: https://aihabitat.org/docs/habitat-lab/habitat.core.env.RLEnv.html
            step = 0
            while not env.episode_over:
                # action = agent.act(observations)
                action = agent.act(env.current_episode.goals[0].position)
                # print("shortest path action: ", action)
                if action is None:
                    break

                # next_observations, reward, done, info = env.step(action)
                
                next_observations = env.step(action)
                
                info = env.get_metrics()
                # print("info after first action: ", info)
                # print("env is over? ", env.episode_over)
                output_image = make_image(observations, info) if offline_dataset_config["record_video"] else None

                # compute pose
                pose = get_2dpose(env)
                observations["top_down_map"] = info["top_down_map"]["map"]
                observations["agent_centered_top_down_map"] = map_centered_by_agent(
                    info
                )
                # record step
                dataset_collector.record_step(
                    {
                        "observations": observations,
                        "action": action,
                        "pose": pose,
                    },
                    output_image,
                )
                observations = next_observations

                step += 1
            # uncomment me for pact agent
            # agent.clear_buffer()
            dataset_collector.episode_end()
            if step == 1:
                print("abnormal episodes again!")
                print(env.current_episode.__getstate__())
                exit()
        # save data set to seqrecord format
        # dataset_collector.to_seqrecord(
        #     recorddir=offline_dataset_config["seqrecord_dir"],
        #     features=offline_dataset_config["seqrecord_features"],
        #     dataset_transform_module=offline_dataset_config["dataset_transform_module"],
        # )
        dataset_collector.to_json()

def add_sensors_to_default():
    import habitat.config.default as default
    default._C.habitat.simulator.top_down_rgb = default.simulator_sensor.clone()
    default._C.habitat.simulator.top_down_rgb.sensor_subtype = "ORTHOGRAPHIC"
    default._C.habitat.simulator.top_down_rgb.type = "HabitatSimRGBSensor"
    default._C.habitat.simulator.top_down_rgb.uuid = "top_down_rgb"
    default._C.habitat.simulator.top_down_semantic = default.simulator_sensor.clone()
    default._C.habitat.simulator.top_down_semantic.sensor_subtype = "ORTHOGRAPHIC"
    default._C.habitat.simulator.top_down_semantic.type = "HabitatSimSemanticSensor"
    default._C.habitat.simulator.top_down_semantic.uuid = "top_down_semantic"

def pre_setup():
    """Runnning a few setups that update the behavior of habiatat-lab, without changing the habitat-lab repo
    """

    # add new sensors
    add_sensors_to_default()

if __name__ == "__main__":

    from time import perf_counter
    start_time = perf_counter()
    pre_setup()
    # skip the program name in sys.argv
    config = cli.parse(sys.argv[1:])
    run_pointnav_episodes(config["offline_dataset"], config["online_env"], config["agent_config"])
    end_time = perf_counter()
    print(f"Program takes {(end_time-start_time)/60.0} mins")
