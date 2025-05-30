import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
import math

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class AirHockeyEnv(MujocoEnv):
    def __init__(
        self,
        xml_path: str = "/workspace/ros2_ws/src/airhockey2025/ka/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        step_cnt_threshold: int = 200,
        **kwargs,
    ):
        # モデル読み込み＆観測次元取得
        model = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = model.nq + model.nv
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # 親クラス初期化
        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode="rgb_array",
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # action_space を後から上書き
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=model.actuator_actnum.shape,
            dtype=np.float32,
        )

        # エピソード切り替え上限
        self.step_cnt_threshold = step_cnt_threshold
        # カウンター初期化
        self.step_cnt = 0

        # ゲーム設定
        self.speed = 1.5
        self.min_speed = 0.01
        self.x_limit = 1.1
        self.y_limit = 0.6
        self.hit_reward = 1.0
        self.lose_penalty = -5.0
        self.win_reward = 5.0
        self.collision_penalty = -3.0
        self.prox_limit = 0.5
        self.prox_coeff = 0.5
        self.table_height_thresh = 0.1

        # ジオメトリ&サイトIDキャッシュ
        self.puck_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "puck")
        self.paddle_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "paddle")
        self.table_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.paddle_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "paddle_site")

    def step(self, action):
        self.step_cnt += 1
        scaled = action * self.model.actuator_ctrlrange[:,1]
        self.do_simulation(scaled, self.frame_skip)
        obs = self._get_obs()
        reward, done, info = self._compute_reward_and_done()
        truncated = self.step_cnt >= self.step_cnt_threshold

        # ログ出力（50ステップごと）
        if self.step_cnt % 50 == 0 or done:
            outcome = info.get('outcome', 'ongoing')
            #print(f"Step {self.step_cnt}: reward={reward:.3f}, outcome={outcome}, dist_to_puck={info.get('dist_to_puck', 0.0):.3f}")

        return obs, reward, done, truncated, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _compute_reward_and_done(self):
        reward = 0.0
        done = False
        info = {'outcome': 'ongoing'}

        # ヒット検出
        hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == self.puck_geom and c.geom2 == self.paddle_geom) or \
               (c.geom2 == self.puck_geom and c.geom1 == self.paddle_geom):
                hit = True
                break
        if hit:
            self.has_hit = True
            reward += self.hit_reward

        # パック位置取得
        px, py = self.data.qpos[0], self.data.qpos[1]
        puck_pos = np.array([px, py, 0.0])

        # パドル先端位置取得
        paddle_pos = self.data.site_xpos[self.paddle_site]
        dist_to_puck = np.linalg.norm(paddle_pos[:2] - puck_pos[:2])
        info['dist_to_puck'] = dist_to_puck

        # 近接報酬
        if dist_to_puck < self.prox_limit:
            reward += self.prox_coeff * (1 - dist_to_puck / self.prox_limit)

        # アーム高さ報酬
        if paddle_pos[2] < self.table_height_thresh:
            reward += 0.1

        # 自陣からの場外
        if px < -self.x_limit:
            reward += self.lose_penalty
            done = True
            info['outcome'] = 'lose'
        # 相手陣から場外
        elif px > self.x_limit:
            if getattr(self, 'has_hit', False):
                reward += self.win_reward
                info['outcome'] = 'win'
            else:
                reward += self.lose_penalty
                info['outcome'] = 'no_hit_opp'
            done = True
        # サイドアウト
        if abs(py) > self.y_limit:
            done = True
            info['outcome'] = 'side'

        # 衝突ペナルティ
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == self.table_geom or c.geom2 == self.table_geom) and \
               not (c.geom1 == self.puck_geom or c.geom2 == self.puck_geom):
                reward += self.collision_penalty
                done = True
                info['collision'] = True
                break

        return reward, done, info

    def reset_model(self):
        self.step_cnt = 0
        self.has_hit = False

        # ロボット初期状態ランダム化
        qpos = self.init_qpos + np.random.uniform(-0.1, 0.1, self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-0.1, 0.1, self.model.nv)
        self.set_state(qpos, qvel)

        # パック初期位置・速度設定
        x0 = np.random.uniform(-self.x_limit, 0)
        y0 = np.random.uniform(-self.y_limit, self.y_limit)
        theta = np.random.uniform(-math.pi/4, math.pi/4)
        vx = self.speed * math.cos(theta)
        vy = self.speed * math.sin(theta)
        self.data.qpos[0:3] = [x0, y0, 0.0]
        self.data.qvel[0:3] = [vx, vy, 0.0]

        return self._get_obs()
