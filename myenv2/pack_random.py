import mujoco
import mujoco.viewer
import os
import time
import random
import math

# === assetsディレクトリをカレントに ===
os.chdir("/workspace/ros2_ws/src/airhockey2025/ka/assets")

# === MuJoCoモデル読み込み（patched_table.xml を直接使う） ===
model = mujoco.MjModel.from_xml_path("patched_table.xml")
data = mujoco.MjData(model)

# テーブルの範囲・パラメータ
x_range = (-0.8, 0.8)
y_range = (-0.5, 0.5)
speed = 3.0

x_limit = 1.1
y_limit = 0.6
min_speed = 0.01  # 停止判定の閾値

# === GUIビューア起動 ===
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 毎回状態リセット
        mujoco.mj_resetData(model, data)

        # 少し待つ（任意）
        time.sleep(0.1)

        # ランダムな出現位置と方向
        x0 = random.uniform(*x_range)
        y0 = random.uniform(*y_range)
        theta = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(theta)
        vy = speed * math.sin(theta)

        # 初期位置・速度設定（qpos[0:3] が x,y,z）
        data.qpos[0:3] = [x0, y0, 0.0]
        data.qvel[0:3] = [vx, vy, 0.0]

        print(f"🚀 発射: 位置=({x0:.2f}, {y0:.2f})、速度=({vx:.2f}, {vy:.2f})")

        # シミュレーションループ
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            x, y = data.qpos[0], data.qpos[1]
            vx, vy = data.qvel[0], data.qvel[1]
            speed_now = math.hypot(vx, vy)

            # 場外判定
            if abs(x) > x_limit or abs(y) > y_limit:
                print(f"🛑 場外: x={x:.2f}, y={y:.2f} → リセット")
                break

            # 停止判定
            if speed_now < min_speed:
                print(f"🛑 停止検知: 速度={speed_now:.4f} → リセット")
                break
