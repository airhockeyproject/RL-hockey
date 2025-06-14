import myenv # 環境を登録するためにインポート
import gymnasium as gym
import cv2
import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC, PPO # SACが連続値には向いている
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise # 探索促進のためのノイズ
from typing import Callable

# train.py 内
def make_env():
    """新しいホッケー環境を作成する関数"""
    env_id_to_make = "AirHockey-v0" # 使用するIDを確認
    env_instance = gym.make(env_id_to_make)
    # ★★★ 以下の2行のデバッグプリントが非常に重要です ★★★
    return env_instance

def render(images):
    """画像リストをOpenCVでレンダリング"""
    if not images:
        print("レンダリングする画像がありません。")
        return
        
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (640, 480))
        cv2.imshow('Video', img_bgr)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    線形に学習率を減衰させるための関数を返す。

    :param initial_value: 初期の学習率 (例: 3e-4)
    :return: 現在の進捗(1.0から0.0)を入力とし、現在の学習率を返す関数
    """
    def func(progress_remaining: float) -> float:
        """
        進捗に応じて現在の学習率を計算する。
        progress_remaining は、学習の残り割合を示し、1.0 (開始時) から 0.0 (終了時) まで減少する。
        """
        return progress_remaining * initial_value

    return func

def train_rl_agent(model_path=None, train=True, total_timesteps=500000):
    """RLエージェントの学習または評価"""
    print("RLエージェントのセットアップ...")
    os.makedirs("models_hockey", exist_ok=True)
    os.makedirs("logs_hockey", exist_ok=True)

    # train.py の train_rl_agent 関数内
    # ...
    num_envs = 32
    env = DummyVecEnv([make_env for _ in tqdm(range(num_envs))])
    env = VecMonitor(env, "logs_hockey/monitor")


    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50000 // num_envs),
        save_path="./models_hockey/",  # ★★★ 保存パスを正しく指定 ★★★
        name_prefix="sac_hockey",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    # ...
    if model_path is None:
        print("新しいSACモデルを作成します。")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda", # 必要に応じて
            learning_starts=10000,
            buffer_size=200000,
            gamma=0.98,
            tau=0.01,
            learning_rate=linear_schedule(1e-3), # 線形スケジュールを使用
            action_noise=NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=0.1 * np.ones(env.action_space.shape[-1])), # action_noiseも設定
            policy_kwargs=dict(net_arch=[256, 256])
        )
    else:
    # ...
        print(f"{model_path} からモデルを読み込みます。")
        model = SAC.load(model_path, env, verbose=1, device="cuda", learning_rate=linear_schedule(1e-3))
        # model = SAC("MlpPolicy", env, verbose=1, device="cuda")
        # model.policy.load_state_dict(
        #     old_model.policy.state_dict()
        # )
    if train:
        print(f"{total_timesteps} ステップの学習を開始します...")
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, log_interval=10)
        model.save("./models_hockey/sac_hockey_final")
        print("学習が完了しました。")

    # --- 評価 ---
    print("評価を開始します...")
    eval_env = make_env() # 評価用に単一環境を作成
    obs, _ = eval_env.reset()
    images = []
    total_reward = 0
    num_hits = 0

    os.makedirs("videos", exist_ok=True)
    out_path = "videos/evaluation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    frame_size = (640, 480)
    writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)








    for i in tqdm(range(1000)): # 評価ステップ数
        action, _state = model.predict(obs, deterministic=True) # 決定論的に行動を選択
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward

        frame = eval_env.render()  # RGB ndarray
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame_bgr, frame_size)
        images.append(frame)
        writer.write(frame_resized)




        if info.get("hit_puck", False):
            num_hits += 1
            
        images.append(eval_env.render())
        if terminated or truncated:
            print(f"評価エピソード終了。報酬: {total_reward:.2f}, ヒット数: {num_hits}")
            obs, _ = eval_env.reset()
            total_reward = 0
            num_hits = 0
            
    eval_env.close()
    writer.release()
    print("評価のレンダリング...")
    render(images)
    env.close()


if __name__ == "__main__":
    # --- モードを選択 ---

    # Option 1: 新規に学習を開始
    #train_rl_agent(train=True, total_timesteps=1000000) # ステップ数を増やす

    # Option 2: 既存モデルをロードして学習を再開
    # train_rl_agent(model_path="./models_hockey/sac_hockey_final.zip", train=True, total_timesteps=2000000)

    # Option 3: 既存モデルをロードして評価のみ実行
    train_rl_agent(model_path="./models_hockey/sac_hockey_final.zip", train=False)
