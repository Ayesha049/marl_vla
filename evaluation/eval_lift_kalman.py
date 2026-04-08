"""
Evaluate lift policy robustness with action noise across multiple predefined noise levels.

Tests noise standard deviations: [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]

Results are saved in both JSON and CSV formats for easy analysis.

Examples:
python eval_robustness_lift.py --n_rollouts 100
python eval_robustness_lift.py --epoch 500 --seed 42
"""

import os
import sys
import glob
import subprocess
import argparse
import json
import csv
from datetime import datetime

class KalmanFilter:
    def __init__(self, dim, Q=1e-3, R=1e-2):
        import numpy as np
        self.dim = dim
        self.x = np.zeros(dim)
        self.P = np.eye(dim)
        self.Q = Q * np.eye(dim)
        self.R = R * np.eye(dim)

    def update(self, z):
        import numpy as np

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Gain
        K = P_pred @ np.linalg.inv(P_pred + self.R)

        # Update
        self.x = x_pred + K @ (z - x_pred)
        self.P = (np.eye(self.dim) - K) @ P_pred

        return self.x

# =========================
# PATH SETUP
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_lift/bc_rnn_lift')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# MUJOCO SETUP
# =========================
def setup_mujoco():
    paths = [
        os.path.expanduser('~/.mujoco/mujoco210/bin'),
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/x86_64-linux-gnu/nvidia',
    ]

    current = os.environ.get('LD_LIBRARY_PATH', '')

    new_paths = [p for p in paths if os.path.exists(p)]
    updated = ":".join(new_paths + [current])

    os.environ['LD_LIBRARY_PATH'] = updated

    print("🔧 LD_LIBRARY_PATH set to:")
    print(updated)


# =========================
# CHECKPOINT FINDER
# =========================
def find_latest_checkpoint(epoch=600):
    pattern = os.path.join(CHECKPOINT_DIR, f'*/models/model_epoch_{epoch}.pth')
    matches = sorted(glob.glob(pattern))

    if matches:
        return matches[-1]

    print("❌ Checkpoint not found.")
    return None


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=600)
    parser.add_argument('--n_rollouts', type=int, default=50)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    # Define noise levels to test
    noise_levels = [0.1, 0.2, 0.5, 0.75]

    setup_mujoco()

    agent_path = find_latest_checkpoint(args.epoch)
    if not agent_path:
        return 1

    print(f"\n📦 Using checkpoint: {agent_path}")
    print(f"🌪 Testing noise stds: {noise_levels}")

    # =========================
    # LOAD ROBOMIMIC SCRIPT
    # =========================
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy once
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=agent_path,
        device=device,
        verbose=False
    )

    print("✅ Policy loaded")

    # Create environment once
    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    print("✅ Environment created")

    # =========================
    # LOOP OVER NOISE LEVELS
    # =========================
    all_results = []
    try:
        action_dim = env.action_spec[0].shape[0]
    except:
        action_dim = len(policy(env.reset()))
    print(f"🔍 Action dimension: {action_dim}")
    for noise_std in noise_levels:
        print(f"\n🔄 Testing noise_std = {noise_std}")
        
        
        # =========================
        # EVALUATION LOOP
        # =========================
        rewards = []
        successes = []

        for ep in range(args.n_rollouts):
            kf = KalmanFilter(dim=action_dim)
            obs = env.reset()
            policy.start_episode()

            ep_reward = 0.0
            success = False

            for step in range(args.horizon):

                action = policy(obs)

                # 🔥 ADD NOISE
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, size=action.shape)
                    noisy_action = action + noise
                else:
                    noisy_action = action

                # 🔥 KALMAN FILTER
                filtered_action = kf.update(noisy_action)

                # clip
                action = np.clip(filtered_action, -1.0, 1.0)

                obs, reward, done, _ = env.step(action)
                ep_reward += reward

                if env.is_success()["task"]:
                    success = True
                    break

                if done:
                    break

            rewards.append(ep_reward)
            successes.append(int(success))

        # =========================
        # RESULTS FOR THIS NOISE LEVEL
        # =========================
        mean_reward = float(np.mean(rewards))
        success_rate = float(np.mean(successes))

        print(f"  📊 Mean Reward:    {mean_reward:.4f}")
        print(f"  📊 Success Rate:   {success_rate * 100:.2f}%")

        # =========================
        # SAVE RESULTS
        # =========================
        result = {
            "noise_std": noise_std,
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "n_rollouts": args.n_rollouts,
            "seed": args.seed,
        }
        
        all_results.append(result)

    # =========================
    # SUMMARY
    # =========================
    print("\n" + "=" * 60)
    print("📊 SUMMARY RESULTS")
    print("=" * 60)
    for result in all_results:
        print(f"Noise {result['noise_std']:4.2f}: Reward={result['mean_reward']:7.4f}, Success={result['success_rate']*100:5.2f}%")
    print("=" * 60)

    # =========================
    # SAVE ALL RESULTS
    # =========================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_filename = f"robustness_eval_{timestamp}.json"
    json_path = os.path.join(RESULTS_DIR, json_filename)
    
    with open(json_path, "w") as f:
        json.dump({
            "results": all_results,
            "config": {
                "epoch": args.epoch,
                "n_rollouts": args.n_rollouts,
                "horizon": args.horizon,
                "seed": args.seed,
                "noise_levels": noise_levels
            }
        }, f, indent=4)

    print(f"💾 Saved JSON results to: {json_path}")
    
    # Save CSV
    csv_filename = f"robustness_eval_lift_{timestamp}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['noise_std', 'mean_reward', 'success_rate', 'n_rollouts', 'seed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for result in all_results:
            writer.writerow({
                'noise_std': result['noise_std'],
                'mean_reward': f"{result['mean_reward']:.4f}",
                'success_rate': f"{result['success_rate']:.4f}",
                'n_rollouts': result['n_rollouts'],
                'seed': result['seed']
            })

    print(f"💾 Saved CSV results to: {csv_path}")


if __name__ == "__main__":
    sys.exit(main())