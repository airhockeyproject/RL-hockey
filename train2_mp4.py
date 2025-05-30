import myenv2            # ← myenv ではなく myenv2 を読み込む
import gymnasium as gym
import cv2
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from tqdm import tqdm

def make_env():
    # myenv2 側で "AirHockey2-v0" として環境登録している前提
    return gym.make("AirHockey2-v0")

def save_video(frames, filename="playback.mp4", fps=30):
    """
    frames: list of RGB images (numpy arrays)
    filename: 出力ファイル名
    fps: フレームレート
    """
    if not frames:
        print("No frames to save.")
        return

    # OpenCV VideoWriter 初期化には (width, height)
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    for img in frames:
        # RGB → BGR に変換して書き込む
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    print(f"Saved video to {filename}")

def train(model_path=None, train=True):
    # ディレクトリ作成
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs2/monitor2", exist_ok=True)

    # マルチプロセス環境の生成
    num_envs = 8
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecMonitor(env, "logs2/monitor2")

    # チェックポイント用コールバック
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="sac_airhockey2",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # モデル生成またはロード
    if model_path is None:
        model = SAC("MlpPolicy", env, verbose=1, device="cuda")
    else:
        model = SAC.load(model_path, env=env, verbose=1, device="cuda")

    # 学習実行
    if train:
        model.learn(total_timesteps=2000000, callback=checkpoint_callback)
        model.save("./models/sac_airhockey2_final")

    # 単一環境での評価＋フレーム収集
    eval_env = make_env()
    obs, _ = eval_env.reset()
    frames = []

    for _ in tqdm(range(1000), desc="Evaluating"):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        frame = eval_env.render()  # RGB ndarray
        frames.append(frame)

        if terminated or truncated:
            obs, _ = eval_env.reset()

    # 動画として保存
    save_video(frames, filename="airhockey_playback.mp4", fps=30)

if __name__ == "__main__":
    # 新規学習＋動画保存
    train(model_path=None, train=True)

    # モデル読み込み＋評価のみで動画保存したいときはコメントを外して使ってください
    # train(model_path="./models/sac_airhockey2_final.zip", train=False)
