from myenv import env # 環境を登録するためにインポート
import gymnasium as gym
import cv2
import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC, PPO # SACが連続値には向いている
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise # 探索促進のためのノイズ

# 環境登録の実行
env.register_env()

# train.py 内
def make_env():
    """新しいホッケー環境を作成する関数"""
    env_id_to_make = "RobotHockey-v0" # 使用するIDを確認
    print(f"DEBUG make_env: gym.make('{env_id_to_make}') を呼び出します...")
    env_instance = gym.make(env_id_to_make, render_mode="rgb_array")
    # ★★★ 以下の2行のデバッグプリントが非常に重要です ★★★
    print(f"DEBUG make_env: 作成されたインスタンスの型: {type(env_instance)}")
    print(f"DEBUG make_env: 作成されたインスタンスの action_space: {env_instance.action_space}, shape: {env_instance.action_space.shape}")
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

def train_rl_agent(model_path=None, train=True, total_timesteps=500000):
    """RLエージェントの学習または評価"""
    print("RLエージェントのセットアップ...")
    os.makedirs("models_hockey", exist_ok=True)
    os.makedirs("logs_hockey", exist_ok=True)

    # train.py の train_rl_agent 関数内
    # ...
    num_envs = 16
    env = DummyVecEnv([make_env for _ in range(num_envs)])
    env = VecMonitor(env, "logs_hockey/monitor")

    # --- ★★★ ここにデバッグプリントを追加 ★★★ ---
    print(f"DEBUG TRAIN.PY: env.action_space: {env.action_space}")
    if hasattr(env, 'single_action_space'):
        print(f"DEBUG TRAIN.PY: env.single_action_space: {env.single_action_space}")
    # -----------------------------------------

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
            learning_rate=3e-4,
            action_noise=NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=0.1 * np.ones(env.action_space.shape[-1])), # action_noiseも設定
            policy_kwargs=dict(net_arch=[256, 256])
        )
    else:
    # ...
        print(f"{model_path} からモデルを読み込みます。")
        model = SAC.load(model_path, env, verbose=1, device="cuda")

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

    for i in tqdm(range(2000)): # 評価ステップ数
        action, _state = model.predict(obs, deterministic=True) # 決定論的に行動を選択
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if info.get("hit_puck", False):
            num_hits += 1
            
        images.append(eval_env.render())
        if terminated or truncated:
            print(f"評価エピソード終了。報酬: {total_reward:.2f}, ヒット数: {num_hits}")
            obs, _ = eval_env.reset()
            total_reward = 0
            num_hits = 0
            
    eval_env.close()
    print("評価のレンダリング...")
    render(images)
    env.close()


if __name__ == "__main__":
    # --- モードを選択 ---

    # Option 1: 新規に学習を開始
    train_rl_agent(train=True, total_timesteps=1) # ステップ数を増やす

    # Option 2: 既存モデルをロードして学習を再開
    #train_rl_agent(model_path="./models_hockey/sac_hockey_XXXXX.zip", train=True)

    # Option 3: 既存モデルをロードして評価のみ実行
    # train_rl_agent(model_path="./models_hockey/sac_hockey_final.zip", train=False)