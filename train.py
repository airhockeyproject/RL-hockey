import myenv
import gymnasium as gym
import cv2
import os
from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from tqdm import tqdm

def make_env():
    return gym.make("AirHockey-v0")

def render(images):
    for img in images:
        # 画像をuint8のBGRに変換（OpenCV用）
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Video', img_bgr)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def train(model_path=None, train=True):
    # 保存用のディレクトリを作成
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 並列数を指定（CPUコア数に応じて調整）
    num_envs = 8
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecMonitor(env, "logs/monitor")
    
    # チェックポイントコールバックの設定
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # 50000ステップごとに保存（環境数も考慮）
        save_path="./models/",
        name_prefix="sac_airhockey",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # モデルの作成と学習
    if model_path is None:
        model = SAC("MlpPolicy", env, verbose=True, device="cuda")
    else:
        model = SAC.load(model_path, env, Verbose=True, device="cuda")

    if train:
        model.learn(total_timesteps=100000, callback=checkpoint_callback)
        model.save("./models/sac_airhockey_final")
    
    
    # 評価用に単一環境を作成（並列環境では render できない）
    eval_env = make_env()
    obs, _ = eval_env.reset()
    images = []
    observations = []
    for _ in tqdm(range(1000)):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        observations.append(obs)
        images.append(eval_env.render())
        if done:
            obs, _ = eval_env.reset()
    render(images)

if __name__=="__main__":
    train("./models/sac_airhockey_final", False)