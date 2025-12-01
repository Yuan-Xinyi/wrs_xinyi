import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ppo import Agent, RobotsEnv, Args    # 直接引用你的训练文件


def evaluate_and_visualize(model_path, episodes=3, save_fig=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load Args ----
    args = Args()
    args.num_envs = 1    # evaluate 单环境

    # ---- Create Env ----
    env = RobotsEnv(num_envs=1, device=device)

    # ---- Create Agent ----
    agent = Agent(env).to(device)

    # ---- Load weights ----
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    print(f"[OK] Loaded model: {model_path}")

    # --------------------------
    # Evaluate Episodes
    # --------------------------
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        done = False
        total_reward = 0

        tcp_traj = []          # 记录tcp轨迹
        goal = env.goal_pos[0].cpu().numpy()

        while not done:
            # record tcp
            tcp_traj.append(env.tcp_pos[0].cpu().numpy())

            # deterministic action
            with torch.no_grad():
                mean = agent.actor_mean(obs)
                action = mean

            action_np = action.cpu().numpy()
            next_obs_np, reward, term, trunc, info = env.step(action_np)

            total_reward += reward[0]
            done = bool(term[0] or trunc[0])

            obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)

        tcp_traj = np.array(tcp_traj)

        print(f"Episode {ep+1} return={total_reward:.3f}, steps={tcp_traj.shape[0]}")

        # ----------------------------
        # Visualize TCP trajectory
        # ----------------------------
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(tcp_traj[:,0], tcp_traj[:,1], tcp_traj[:,2], '-r', label='TCP Trajectory')
        ax.scatter(goal[0], goal[1], goal[2], c='g', s=100, label='Goal')
        ax.scatter(tcp_traj[0,0], tcp_traj[0,1], tcp_traj[0,2], c='b', s=60, label='Start')

        ax.set_title(f"Episode {ep+1} TCP Trajectory")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # 设置固定比例
        max_range = np.array([tcp_traj[:,0].max()-tcp_traj[:,0].min(),
                              tcp_traj[:,1].max()-tcp_traj[:,1].min(),
                              tcp_traj[:,2].max()-tcp_traj[:,2].min()]).max()
        Xb = 0.5*max_range * np.array([-1,1]) + np.mean(tcp_traj[:,0])
        Yb = 0.5*max_range * np.array([-1,1]) + np.mean(tcp_traj[:,1])
        Zb = 0.5*max_range * np.array([-1,1]) + np.mean(tcp_traj[:,2])
        ax.set_xlim(Xb[0], Xb[1])
        ax.set_ylim(Yb[0], Yb[1])
        ax.set_zlim(Zb[0], Zb[1])

        plt.tight_layout()

        if save_fig:
            plt.savefig(f"trajectory_ep{ep+1}.png", dpi=150)
            print(f"[Saved] trajectory_ep{ep+1}.png")

        plt.show()


if __name__ == "__main__":
    model_path = "runs/ppo_xarm_1764581123.5161152/checkpoints/agent_iter15_step1966080.pth"
    evaluate_and_visualize(model_path, episodes=3)
