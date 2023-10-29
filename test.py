from deprecated.ddpg import DDPG

model = DDPG.load(r"D:\edge_cache\episode-2700-reward-1705.73.pt")
# model = DDPG.load(r"result/DDPG-Pendulum-v1-2023-10-26-18-49-22/episode-145-reward--3.77.pt")
total_reward = model.test(truncated_exit=True)
print(total_reward)

