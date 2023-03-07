"""Implement agents to be deployed in Habitat.

Refs:
    https://aihabitat.org/docs/habitat-lab/habitat.Agent.html#act agent needs to have act and reset methods
"""

from typing import Tuple, Union, Dict, Any
import collections
import numpy as np
import torch
from torch import package
import habitat
import habitat.core as core
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.core.utils import try_cv2_import

cv2 = try_cv2_import()

# todo: cuda inference
# todo: registry
class RandomAgent(habitat.Agent):
    def __init__(self, action_space) -> None:
        super().__init__()
        self.action_space = action_space

    def reset(self) -> None:
        pass

    def act(self, obs: core.simulator.Observations) -> dict:
        action = np.random.choice(self.action_space)
        return {"action": action}


class ShortestPathFollowerAgent(habitat.Agent):
    def __init__(self, sim: habitat.Simulator, goal_radius: float) -> None:
        super().__init__()
        self._agent = ShortestPathFollower(sim, goal_radius, False)

    def reset(self) -> None:
        pass

    def act(self, obs: Union[np.ndarray, core.simulator.Observations]) -> dict:
        # for shortest path follower, the observation is agent's current position
        action = self._agent.get_next_action(obs)
        return {"action": action}


class ModelPointNavAgent(habitat.Agent):
    """A learning-model based agent wandering in Habitat.

    Args:
        habitat (Agent):
    """

    def __init__(self, raw_action_space: np.ndarray, agent_config: dict) -> None:
        super().__init__()

        # setup pact agent
        imp = package.package_importer(agent_config["package_path"])
        ckpt = agent_config["ckpt_path"]
        self.model = imp.load_pickle(
            agent_config["package_name"], agent_config["resource_name"]
        )
        self.input_seq_len = ckpt["hyper_parameters"]["gpt_config"]["seq_len"]

        # setup features that will be sent to pact modal in batch
        self.action_features = agent_config["action_features"]
        self.state_features = agent_config["state_features"]
        # inputs we need in deployment is a subset of inputs used to train the ckpt
        inputs_in_batch = {
            feature: ckpt["datamodule_hyper_parameters"]["dataset_config"]["inputs"][
                feature
            ]
            for feature in self.state_features + [self.action_features]
        }
        self.input_transform = InputTransform(inputs_in_batch)
        # action setup
        self.raw_action_space = raw_action_space
        self.rawact2action = {action: i for i, action in enumerate(raw_action_space)}
        self.action2rawact = {
            action: rawact for rawact, action in self.rawact2action.items()
        }
        # used when action buffer is not full
        self.padding_action = self.rawact2action[np.random.choice(raw_action_space)]

        # special care for stop action, which should not be a real action
        self.stop_action = self.rawact2action["STOP"]

        # buffer to hold data sequence up to input_seq_len
        self.buffer = collections.deque(maxlen=self.input_seq_len - 1)

    def reset(self) -> None:
        self.clear_buffer()

    def clear_buffer(self) -> None:
        self.buffer.clear()

    def act(self, observations: Dict[str, Any]) -> dict:
        """Act given aggreated info including state, reward, and map information etc.

        Args:
            observations (Dict[str, Any]): _description_

        Returns:
            dict: _description_
        """
        item = {}
        for state_feature in self.state_features:
            value = observations
            for key in state_feature.split("->")[1:]:
                value = value[key]
            item[state_feature] = value
        # a fake action input
        item[self.action_features] = np.asarray(self.padding_action)

        # pre-process input for pact
        item = self.input_transform.pre_transform(item)
        # ! only state and action are being sent to model. This transform
        # cannot handle pose
        item = self.input_transform.transform("val", item)
        self.buffer.append(item)
        batch, pred_action_idx = self.collate_buffer()

        # send data to pact model for predication
        pred_action = self.pact_model.predicate_action(batch)[0][
            pred_action_idx
        ].detach()

        ## output_action
        # not considering 'STOP' for now as we let env decide whether we should stop
        pred_action[self.stop_action] = float("-inf")
        # choose 'greedy' action
        action = torch.argmax(pred_action).item()

        # update action buffer with the choose action
        self.buffer[-1][self.action_features] = np.asarray(action)

        return {"action": self.action2rawact[action], "action_value": action}

    def collate_buffer(self) -> Tuple[dict, int]:
        """_summary_

        Returns:
            Tuple[dict, int]: _description_
        """
        # the index of action we need from predicted action sequence
        pred_action_idx = len(self.buffer) - 1
        if len(self.buffer) == self.input_seq_len:
            buffer_placeholder = self.buffer
        else:
            buffer_placeholder = [
                self.buffer[i] if i < len(self.buffer) else self.buffer[-1]
                for i in range(self.input_seq_len)
            ]

        # collate items in buffer_placeholder
        batch = {}
        for key in buffer_placeholder[-1]:
            batch[key] = torch.stack(
                [buffer_placeholder[i][key] for i in range(self.input_seq_len)], dim=0
            )
            # add a batch dim
            batch[key] = batch[key].unsqueeze(dim=0)
        return batch, pred_action_idx


class InputTransform:
    # todo: need to implement this when deploy model-based agent in habitat
    # placeholder of input transform for model based agent
    def __init__(self) -> None:
        pass
