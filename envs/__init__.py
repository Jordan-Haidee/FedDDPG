"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from gymnasium.envs.registration import register

register(
    id="Hopper-heter",
    entry_point="envs.hopper:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Pendulum-heter",
    entry_point="envs.pendulum:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="HalfCheetah-heter",
    entry_point="envs.half_cheetah:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
