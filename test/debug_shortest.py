import habitat
from habitat import Agent
from tqdm import tqdm
from habitat import Env
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


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
    """Runnning a few setups that update the behavior of habiatat-lab, without changing the habitat-lab repo"""

    # add new sensors
    add_sensors_to_default()


pre_setup()


class ShortestPathFollowerAgent(Agent):
    def __init__(self, sim, goal_radius: float) -> None:
        super().__init__()
        self._agent = ShortestPathFollower(sim, goal_radius, False)

    def reset(self) -> None:
        pass

    def act(self, obs) -> dict:
        # for shortest path follower, the observation is agent's current position
        action = self._agent.get_next_action(obs)
        return {"action": action}


class SimpleEnv(Env):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0


# reproduce results of habitat lab
env_config = habitat.get_config(
    config_paths="../configs/pointnav_config/pointnav_hm3d.yaml"
)
with habitat.config.read_write(env_config):
    env_config.habitat.dataset.split = "train"
    env_config.habitat.dataset.data_path = "/datadrive/azure_storage/pactdata/habitat-data/habitat-dataset/hm3d/test/train/train.json.gz"
    env_config.habitat.dataset.scenes_dir = (
        "/datadrive/azure_storage/pactdata/habitat-data/habitat-dataset"
    )
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
            observations = next_observations

            step += 1
        print(step)
