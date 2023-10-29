"""支持两种训练：
1. 有embedding
2. 无embedding
"""
import copy
from datetime import datetime
from pathlib import Path
import numpy as np
import pytoml
import argparse
from addict import Dict
from model import FedDDPG

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.toml")
parser.add_argument("--exp-num", type=int, default=1)
args = parser.parse_args()

with open(Path(__file__).parent / args.config, encoding="utf-8") as f:
    config = Dict(pytoml.load(f))

exp_start_time = datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")
np.random.seed(config.seed)

# 进行args.exp_num次实验
for e in range(0, args.exp_num):
    save_dir = Path(__file__).parent / f"{config.save_dir}-{exp_start_time}" / f"exp-{e}"
    save_dir.mkdir(parents=True, exist_ok=True)

    heter_set = np.random.uniform(0, 1, (config.env_num, config.heter_num))
    backup_config = {**copy.deepcopy(config), "heter_set": heter_set.tolist()}
    if config.embedding_dim > 0:
        embedding_set = np.random.uniform(-1, 1, (config.env_num, config.embedding_dim))
        backup_config.update({"embedding_set": embedding_set.tolist()})
    with open(save_dir / "config_backup.toml", "w") as f:
        pytoml.dump(backup_config, f)

    env_config_list = [
        {
            "env": config.env,
            "heter": heter_set[i],
            "env_index": i,
            "hidden_dim": config.hidden_dim,
            "lr": config.lr,
            "gamma": config.gamma,
            "sigma": config.sigma,
            "embedding_init": embedding_set[i] if config.embedding_dim > 0 else None,
            "buffer_capicity": config.buffer_capicity,
            "buffer_init_ratio": config.buffer_init_ratio,
            "batch_size": config.batch_size,
            "train_batchs": config.merge_num * config.merge_interval,
            "save_dir": save_dir / f"point-{i}",
            "device": config.device,
        }
        for i in range(config.env_num)
    ]

    model = FedDDPG(
        point_configs=env_config_list,
        merge_interval=config.merge_interval,
        merge_num=config.merge_num,
        episode_num_eval=config.episode_num_eval,
        save_dir=save_dir,
        device=config.device,
        merge_target=config.merge_target,
    )
    model.train()
