__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Tuple

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class HopperEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        heter: np.array = np.array([0.5, 0.5, 0.5]),
        frame_skip: int = 4,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            heter,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        # ---------------------------------
        assert (heter > 0).all() and (heter < 1).all(), "heter scope violate!"
        self.heter = heter
        self.xml_file = self.generate_heter_config()
        # ---------------------------------

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        MujocoEnv.__init__(
            self,
            self.xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size - exclude_current_positions_from_observation
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        observation = self._get_obs()
        reward = rewards - costs
        terminated = self.terminated
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def generate_heter_config(self):
        """根据heter参数, 动态生成xml配置文件"""
        import time
        import tempfile
        from pathlib import Path
        import xml.etree.ElementTree as ET

        # 解析XML文件
        tree = ET.parse(Path(__file__).parent / "config.xml")
        root = tree.getroot()
        # 找到需要修改的元素
        # 重力加速度
        option = root.find("option")
        option.set("gravity", f"0 0 {-15+10*self.heter[0]}")
        # 脚的长宽
        foot_geom = root.find("worldbody/body/body/body/body/geom")
        s1, s2 = foot_geom.get("size").split(" ")
        s1 = 0.03 + 0.06 * self.heter[1]
        s1 = 0.095 + 0.20 * self.heter[2]
        foot_geom.set("size", f"{s1} {s2}")

        tmp_folder = Path(tempfile.gettempdir())
        config_path = tmp_folder / f"{time.perf_counter_ns()}.xml"
        tree.write(config_path)

        return str(config_path)
