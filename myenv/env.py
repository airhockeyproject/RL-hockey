import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box
from typing import Dict, Union



DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class MyRobotEnv(MujocoEnv):
    def __init__(
        self, 
        xml_path = "/workspace/ros2_ws/src/airhockey2025/ka/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        # 一度モデルを読み込んで観測次元を取得
        model = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = model.nq + model.nv
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        # 行動空間：joint velocities [-1, 1] 正規化
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=model.actuator_actnum.shape, dtype=np.float32
        )
        
        # 親クラスの初期化
        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode="rgb_array",
            default_camera_config=default_camera_config,
            **kwargs,
        )
        
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.step_cnt_threshold = 400
        self.configure_puck()

    def step(self, action):
        self.step_cnt += 1
        # 正規化されたactionをスケーリング
        scaled_action = action * self.model.actuator_ctrlrange[:, 1]
        self.do_simulation(scaled_action, self.frame_skip)

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._is_done()
        truncated = True if self.step_cnt > self.step_cnt_threshold else False
        info = {}

        return obs, reward, done, truncated, info
    
    # # パックが入ったら終了
    # def _is_done(self):
    #     x = self.data.qpos[self.puck_x_id]
    #     if not self.puck_x_range.min() < x < self.puck_x_range.max():
    #         return True
    #     return False
    
    def _is_done(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        end_effector_pos = self.data.site_xpos[site_id] 
        flag = end_effector_pos[-1] < 0.4
        # if flag: print(end_effector_pos[-1])
        return flag


    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _compute_reward(self, obs, action):
        # エンドエフェクタの位置を使用した報酬例
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        end_effector_pos = self.data.site_xpos[site_id]
        return 0.7 - end_effector_pos[-1]

    def reset_model(self):
        self.step_cnt = 0

        # ランダム初期化
        qpos = self.init_qpos + np.random.uniform(-0.5, 0.5, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-0.5, 0.5, size=self.model.nv)

        # パック初期化
        theta = np.random.uniform(0, 2 * np.pi)
        qpos[self.puck_x_id] = np.random.uniform(*self.puck_x_range)
        qpos[self.puck_y_id] = np.random.uniform(*self.puck_y_range)
        qvel[self.puck_x_id] = self.puck_speed * np.cos(theta)
        qvel[self.puck_y_id] = self.puck_speed * np.sin(theta)

        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def configure_puck(self):
        self.puck_speed = 10.0
        self.puck_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_x")
        self.puck_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_y")

        self.table_surface_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table_surface")
        geom_indices = [j for j in range(self.model.ngeom) if self.model.geom_bodyid[j] == self.table_surface_id]
        assert len(geom_indices) == 1
        x, y, _ = self.model.geom_size[geom_indices[0]]
        self.puck_x_range = np.array([-x, x]) * 0.8
        self.puck_y_range = np.array([-y, y]) * 0.8