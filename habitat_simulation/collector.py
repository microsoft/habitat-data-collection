"""Collect data inside habitat env for later training or analysis."""

import os
from typing import Any, Callable, Dict, Generator, List, Optional
import json
import shutil
import collections
import numpy as np
from seqrecorder.seqrecord import SeqRecord
from habitat.utils.visualizations.utils import images_to_video


class DatasetCollector:
    def __init__(
        self,
        offline_dataset_config: dict,
        online_env_config: dict,
        episode_start_index: int = 0,
    ) -> None:
        self.description = offline_dataset_config["description"]
        # get absolute path. The expanduser is to take care of `~`
        # ref: https://stackoverflow.com/questions/51520/how-to-get-an-absolute-file-path-in-python
        self._dataset_dir = os.path.abspath(
            os.path.expanduser(offline_dataset_config["dataset_dir"])
        )
        self.record_video = offline_dataset_config["record_video"]

        self.online_env_config = online_env_config
        # episode collection
        self.episode_start_index = episode_start_index
        self.current_episode: Optional[EpisodeCollector] = None
        self.episodes: List[EpisodeCollector] = []

        # info on seqrecord if to_seqrecord() is called
        self.seqrecord_features = {}
        self.seqrecord_dir = ""
        self.seqrecord_pretransform_module = ""

    def episode_start(self, episode_config):
        offline_episode_config = {
            "episode_config": episode_config,
            "dataset_dir": self._dataset_dir,
            "nth_episode": len(self.episodes) + self.episode_start_index,
            "record_video": self.record_video,
        }
        self.current_episode = EpisodeCollector(
            offline_episode_config=offline_episode_config
        )

    def episode_end(self) -> None:
        self.current_episode.episode_end()
        self.episodes.append(self.current_episode)
        self.current_episode = None

    def record_step(self, step_info: dict, image: Optional[np.ndarray]) -> None:
        self.current_episode.record_step(step_info, image)

    def to_json(self, rank: int = 0):
        class OfflineDatasetJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()

                return (
                    obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__
                )

        with open(
            os.path.join(self._dataset_dir, f"offlinedataset_{rank}.json"), mode="w"
        ) as f:
            f.write(OfflineDatasetJSONEncoder().encode(self))

        return

    @classmethod
    def merge_jsons(self, dataset_dir, world_size):
        sub_configs = []
        for rank in range(world_size):
            with open(
                os.path.join(dataset_dir, f"offlinedataset_{rank}.json"),
                encoding="utf-8",
            ) as f:
                sub_config = json.load(f)
            sub_configs.append(sub_config)
        # merge sub_configs
        config = sub_configs[0]
        # merge episodes and save them into a single json file
        for i in range(1, len(sub_configs)):
            config["episodes"] = config["episodes"] + sub_configs[i]["episodes"]

        # write full configs to json
        with open(os.path.join(dataset_dir, "offlinedataset_all.json"), mode="w") as f:
            json.dump(config, f)

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, encoding="utf-8") as f:
            config = json.load(f)
        offline_dataset_config = {
            "description": config["description"],
            "dataset_dir": config["dataset_dir"],
            "record_video": config["record_video"],
        }
        online_env_config = config["online_env_config"]
        dataset_collector = cls(offline_dataset_config, online_env_config)

        # load some useful attributes that are saved outside of DatasetConfig__init__()
        dataset_collector.seqrecord_dir = config.get("seqrecord_dir", "")
        dataset_collector.seqrecord_features = config.get("seqrecord_features", {})
        dataset_collector.seqrecord_pretransform_module = config.get(
            "seqrecord_pretransform_module", ""
        )
        for episode_dict in config["episodes"]:
            offline_episode_config = {
                "episode_config": episode_dict["episode_config"],
                "dataset_dir": dataset_collector._dataset_dir,
                "nth_episode": episode_dict["nth_episode"],
                "record_video": dataset_collector.record_video,
            }
            episode = EpisodeCollector(offline_episode_config=offline_episode_config)
            episode.steps = episode_dict["_steps"]
            dataset_collector.episodes.append(episode)
        return dataset_collector

    @classmethod
    def to_seqrecord(
        self,
        recorddir: str,
        features: Dict[str, str],
        dataset_pretransform_module: str,
        dataset_dir: str,
        num_episodes: int = 0,
    ):
        """_summary_

        Args:
            recorddir (str): the directory that stores seqrecord data
            features (Dict[str, str]): features to encode, should be a subset of data stored in each episode
            input_pre_transform (str): pretransform that applies to data to be saved by seqrecord (todo, put this into registry in future)
            dataset_dir (str): dataset directory that stores the saved episodes data.
            num_episodes (int): number of episdes to encode, if it is 0, seqrecord will try to encode the whole dataset under dataset_dir
        """
        print("Saving data into seqrecord format")
        seqrecord_dir = os.path.abspath(os.path.expanduser(recorddir))
        record = SeqRecord(seqrecord_dir, features, dataset_pretransform_module)
        episode_ids = (
            range(num_episodes)
            if num_episodes > 0
            else [
                d
                for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()
            ]
        )
        for episode_idx in episode_ids:
            episode_data = np.load(
                os.path.join(dataset_dir, str(episode_idx), "observations.npz")
            )
            # pick an arbitrary feature stored in episode
            feature_placehoder = episode_data.files[0]
            for i in range(episode_data[feature_placehoder].shape[0]):
                # episode.files[0].shape[0] -> length of the episode (assume no missing modalities during episode)
                item = {}
                for feature in features:
                    item[feature] = episode_data[feature][i]
                record.write_item(item, (i == 0))
        record.close_recordfile()
        record.dump()
        return

    @property
    def dataset_dir(self) -> str:
        return self._dataset_dir

    def update_dataset_dir(self, dataset_dir: str):
        self._dataset_dir = dataset_dir


class EpisodeCollector:
    def __init__(self, offline_episode_config: Dict[str, Any]) -> None:
        self.episode_config = offline_episode_config["episode_config"]
        self._dataset_dir = os.path.abspath(
            os.path.expanduser(offline_episode_config["dataset_dir"])
        )
        self.nth_episode = offline_episode_config["nth_episode"]
        self.record_video = offline_episode_config["record_video"]
        self.episode_dir = os.path.join(self._dataset_dir, str(self.nth_episode))
        # if os.path.exists(self.episode_dir):
        #     shutil.rmtree(self.episode_dir)
        os.makedirs(self.episode_dir, exist_ok=True)
        self.data = collections.defaultdict(list)
        self.num_steps = 0
        self.images = []

    def record_step(self, observation, image: Optional[np.ndarray] = None):
        # There are two types of data in info
        # 1. small data like action, pose that can be stored in memory
        # 2. big data (rgb image, depth, etc) that we need to store it in disk and keep a record of its relative path
        #    The name of those big data that we need to store path, are in heavy_features
        #    where the method for save and read them are in corresponding `value`
        def parse_observation(
            obs: Dict[str, Any], past_keys: List[str]
        ) -> Dict[str, Any]:
            for key, value in obs.items():
                if isinstance(value, dict):
                    parse_observation(value, past_keys + [key])
                else:
                    # save the value directly or save path
                    feature_name = "-".join(past_keys + [key])
                    if isinstance(value, np.ndarray):
                        self.data[feature_name].append(value)
                    else:
                        self.data[feature_name].append(np.asanyarray(value))
            return

        if self.record_video and image is not None:
            self.images.append(image)
        parse_observation(observation, [])
        self.num_steps += 1

    def episode_end(self):
        if self.record_video and len(self.images) > 0:
            images_to_video(self.images, self.episode_dir, "trajectory")
        # write data to files in disk

        # collate features in dictionary
        res = {}
        for key, value in self.data.items():
            res[key] = np.stack(value, axis=0)
        path = os.path.join(
            self._dataset_dir, str(self.nth_episode), "observations.npz"
        )
        np.savez(path, **res)

        # delete buffer since we do not need them in memory
        self.images = []
        self.data = []

    def __getstate__(self):
        """Overriding this method to specify attributes to saved when in json."""
        obj_dict = self.__dict__.copy()
        del obj_dict["images"]
        del obj_dict["data"]
        del obj_dict["_dataset_dir"]
        return obj_dict
