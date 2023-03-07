"""Generage configuration files for the habitat dataset (need for simulation)
"""
import os
import math
import random
import json
import shutil
import gzip
from tqdm import tqdm
import magnum as mn
import habitat_sim
from habitat_sim.utils import common as habita_sim_comon_utils

# todo: 4. save sanpshot for dataset card
# todo: 6: lift up agent, does not seem to be easy to setup

defaut_config = {
    "name": "test",
    "dataset": "hm3d",
    "seed_everything": 40,
    "scene_dir": "/datadrive/azure_storage/pactdata/habitat-data/habitat-dataset",
    "scene_dataset_config": "hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    "num_episodes_per_scene": 100,
    "goal_radius": 0.2,
    "minimum_distance_apart": 5.0,
    "task_dir": "/datadrive/azure_storage/pactdata/habitat-data/habitat-dataset/hm3d/pointnav_sem_v0",
}

# default_episode_config = {"episode_id": "0", "scene_id": "hm3d/val/00859-3t8DB4Uzvkt/3t8DB4Uzvkt.basis.glb", "start_position":[], "start_rotation": [], "info": {"geodesic_distance": 0}, "goals": [{"position": [7.360230445861816, 2.7564451694488525, 2.521066188812256], "radius": 0.2}], "start_room": None, "shortest_paths": None}


def make_simple_cfg(scene_id, seed):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    if seed is not None:
        sim_cfg.random_seed = seed
    sim_cfg.scene_id = scene_id

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def scene_has_semantic_annot(dir: str) -> bool:
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    return True if any("semantic" in file for file in files) else False


def test_scene_has_semantic_annot() -> None:
    root_dir = (
        "/datadriver/azure_storage/pactdata/habitat-data/habitat-dataset/hm3d/minival/"
    )
    dir = "00800-TEEsavR23oF"
    assert test_scene_has_semantic_annot(os.path.join(root_dir, dir))
    dir = "00801-HaxA7YrQdEC"
    assert not test_scene_has_semantic_annot(os.path.join(root_dir, dir))
    print("Semantic scene finder tested!")
    return


def get_start_goal_positions(sim, minimum_distance_apart: float):
    """Return a tuple of positions that satisfies:
    1. geodestic distance is greater than
    2. There is a viable path between two positions
    """
    path = habitat_sim.ShortestPath()
    while True:
        # todo: impose distance constraint
        start = sim.pathfinder.get_random_navigable_point()
        path.requested_start = start
        succ = False
        for _ in range(10):
            # for each start point try 10 times (10 is a magic number for now)
            goal = sim.pathfinder.get_random_navigable_point()
            path.requested_end = goal
            found_path = sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            # sometimes, geodesic_distance is `Infinity`, which means there is not viable path?
            if (
                found_path
                and path.points is not None
                and len(path.points) > 0
                and geodesic_distance != float("inf")
                and geodesic_distance > minimum_distance_apart
            ):
                succ = True
                break
        if succ:
            break
    return start, goal, geodesic_distance


def main(config: dict) -> dict:
    """generate dataset of configuration files, which will be used by habitat-lab to generate episode data."""
    dataset_config = {}
    if os.path.exists(config["task_dir"]):
        ans = input(
            "The target directory for storing datset configuration file exists. Do you want to delete it? [y/n] "
        )
        if ans == "y":
            print(f"Removing existing directory {config['task_dir']}")
            shutil.rmtree(config["task_dir"])
        else:
            # abort otherwise
            print(
                "Please specify a new directory for storing dataset config files. Aborted!"
            )
            exit()

    # setup seed
    if config.get("seed_everything", None) is not None:
        random.seed(config["seed_everything"])
    splits = ["train", "val"]
    for split in splits:

        scene_episodes_config = {"episodes": []}
        # writes an empty json file as the pointnav dataset does
        scene_episodes_split_dir = os.path.join(config["task_dir"], split)
        os.makedirs(scene_episodes_split_dir, exist_ok=True)
        os.makedirs(os.path.join(scene_episodes_split_dir, "content"), exist_ok=True)
        with gzip.open(
            os.path.join(scene_episodes_split_dir, f"{split}.json.gz"), "wb"
        ) as f:
            scene_episodes_config_json = json.dumps(scene_episodes_config)
            f.write(scene_episodes_config_json.encode())
        # step 1, get the whole set of scenes with semantic annotation
        scene_dataset_dir = os.path.join(config["scene_dir"], config["dataset"], split)
        scenes = [
            scene_dir
            for scene_dir in os.listdir(scene_dataset_dir)
            if not os.path.isfile(os.path.join(scene_dataset_dir, scene_dir))
        ]
        semantic_scenes = [
            scene_dir
            for scene_dir in scenes
            if scene_has_semantic_annot(os.path.join(scene_dataset_dir, scene_dir))
        ]
        dataset_config[f"{split}-num_scenes"] = len(semantic_scenes)
        dataset_config[f"{split}-num_episodes"] = (
            len(semantic_scenes) * config["num_episodes_per_scene"]
        )
        for i in tqdm(range(len(semantic_scenes)), desc=split):
            scene_name = semantic_scenes[i]
            scene_subdir = os.path.join(config["dataset"], split, scene_name)
            scene_id = os.path.join(
                scene_subdir, f"{scene_name.split('-')[1]}.basis.glb"
            )

            # step 2, for each scene: random query start and end position
            scene_cfg = make_simple_cfg(
                os.path.join(config["scene_dir"], scene_id),
                config.get("seed_everything", None),
            )
            try:  # Needed to handle out of order cell run in Colab
                sim.close()
            except NameError:
                pass
            sim = habitat_sim.Simulator(scene_cfg)

            for j in range(config["num_episodes_per_scene"]):
                # seed = 4  # @param {type:"integer"}
                # sim.pathfinder.seed(seed)

                # fmt off
                # @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
                # fmt on
                # ref for generation start and goal states: https://colab.research.google.com/github/facebookresearch/habitat-sim/blob/main/examples/tutorials/colabs/ECCV_2020_Navigation.ipynb#scrollTo=CPKbqbrHxH6m
                # step 3-1: get valid start and goal positions
                start, goal, geodesic_distance = get_start_goal_positions(
                    sim, config["minimum_distance_apart"]
                )

                # step 3-2: get random start-orientation
                start_orientation = random.random() * math.pi * 2.0
                start_rotation = habita_sim_comon_utils.quat_from_magnum(
                    mn.Quaternion.rotation(
                        -mn.Rad(start_orientation), mn.Vector3(0, 1.0, 0)
                    )
                )
                # ref for the format of rotation: https://github.com/facebookresearch/habitat-sim/issues/131
                # this seems to be consistent with pointnav dataset config
                start_rotation = habitat_sim.utils.common.quat_to_coeffs(start_rotation)
                # step 4, save the episode config into json files
                episode_config = {
                    "episode_id": str(j),
                    "scene_id": scene_id,
                    "scene_dataset_config": config["scene_dataset_config"],
                    "start_position": start.tolist(),
                    "start_rotation": start_rotation.tolist(),
                    "info": {"geodesic_distance": geodesic_distance},
                    "goals": [
                        {"position": goal.tolist(), "radius": config["goal_radius"]}
                    ],
                    "start_room": None,
                    "shortest_paths": None,
                }
                scene_episodes_config["episodes"].append(episode_config)

            # step 5: write episodes config into json file

            with gzip.open(
                os.path.join(
                    scene_episodes_split_dir,
                    "content",
                    scene_name.split("-")[1] + ".json.gz",
                ),
                "wb",
            ) as f:
                scene_episodes_config_json = json.dumps(scene_episodes_config)
                f.write(scene_episodes_config_json.encode())

            # for inspection only, delete me once things are working
            with open(
                os.path.join(
                    scene_episodes_split_dir,
                    "content",
                    scene_name.split("-")[1] + ".json",
                ),
                "w",
            ) as f:
                f.write(scene_episodes_config_json)

            scene_episodes_config = {"episodes": []}
    return dataset_config


if __name__ == "__main__":
    dataset_config = main(defaut_config)

    # saving dataset card
    defaut_config["date"] = 1
    import git

    repo = git.Repo(search_parent_directories=True)
    defaut_config["git_repo"] = repo.remotes.origin.url
    defaut_config["git_branch"] = str(repo.active_branch)
    defaut_config["git_commit"] = repo.head.object.hexsha
    for key in dataset_config:
        defaut_config[key] = dataset_config[key]
    with open(
        os.path.join(defaut_config["task_dir"], "dataset_config_card.yaml"), "w"
    ) as f:
        import yaml

        f.write(yaml.safe_dump(defaut_config))
