import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box
from typing import Dict, Union, Optional
import gymnasium as gym


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class MyRobotHockeyEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_path="/workspace/ros2_ws/src/RL-hockey/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        kp: float = 300.0,
        kd: float = 10.0,
        lambda_ik: float = 0.01,
        step_cnt_threshold: int = 500,
        puck_speed: float = 5.0,
        render_mode: Optional[str] = "rgb_array",
        max_ee_vel: float = 2.0,
        **kwargs,
    ):
        _model = mujoco.MjModel.from_xml_path(xml_path) # action_space定義のために一時ロード

        # --- 他のID設定など (puck_body_id, ee_site_id, geom_ids, joint_ids) ---
        self.puck_body_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "puck")
        self.ee_site_name = "ee_site"
        self.ee_site_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        self.racket_geom_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_GEOM, "racket_geom")
        self.puck_geom_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_GEOM, "puck_geom")
        if self.racket_geom_id == -1 or self.puck_geom_id == -1:
            print("警告: ラケットまたはパックのGeom IDが見つかりません。XMLで 'racket_geom' と 'puck_geom' を定義してください。")
            self.racket_geom_id = -1
            self.puck_geom_id = -1
        self.puck_x_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "puck_x")
        self.puck_y_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "puck_y")
        self.puck_z_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "puck_z")
        if self.puck_x_id == -1 or self.puck_y_id == -1:
             raise ValueError("パックのJoint ID (puck_x, puck_y) が見つかりません。")
        # ----------------------------------------------------------------------

        # --- 観測空間 ---
        # _model は MujocoEnv の super().__init__ で self.model としてロードされるので、
        # ここでは obs_dim の計算にだけ使う
        obs_dim = _model.nq + _model.nv + 3 + 3
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        # obs_space は super().__init__ に渡す必要がある
        # -----------------

        # ★★★ 親クラスの初期化を先に実行 ★★★
        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space, # observation_space はここで渡す
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        # super().__init__ が呼ばれた後、self.model と self.data が利用可能になる

        # ★★★ 行動空間 (目標EE位置 + 目標EE速度) を super().__init__ の後に設定 ★★★
        low_pos = np.array([0.0, -0.6, 0.1])
        high_pos = np.array([1.6, 0.6, 0.6])
        low_vel = np.array([-max_ee_vel] * 3)
        high_vel = np.array([max_ee_vel] * 3)
        action_low = np.concatenate([low_pos, low_vel])
        action_high = np.concatenate([high_pos, high_vel])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(6,), dtype=np.float32)
        print(f"DEBUG Env __init__: Action Space Defined (after super, explicitly set): {self.action_space}")
        # ----------------------------------------------------------------------------

        # --- 残りの初期化 (self.model を使うものは super の後) ---
        self.Kp = kp
        self.Kd = kd
        self.lambda_ik = lambda_ik
        self.jnt_range = self.model.jnt_range.copy() * 0.95
        
        self.arm_joint_ids = np.arange(3, 12)
        if len(self.arm_joint_ids) != self.model.nu: # self.model.nu は super() 後にアクセス
            print(f"警告: arm_joint_ids ({len(self.arm_joint_ids)}) と nu ({self.model.nu}) が一致しません！ XML構成を確認してください。")
        print(f"DEBUG: nu={self.model.nu}, nv={self.model.nv}, arm_ids={self.arm_joint_ids}")

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt)) # self.dt も super() 後
        self.step_cnt_threshold = step_cnt_threshold
        self.step_cnt = 0
        self.puck_speed = puck_speed
        self.current_target_pos = None
        self.current_target_vel = None
        self.hit_puck_this_step = False

        self._configure_table_bounds(self.model)

    # ... ( _configure_table_bounds, _get_puck_state, _get_obs は変更なし) ...
    def _configure_table_bounds(self, model):
        """テーブルのサイズからパックの活動範囲を設定"""
        self.table_surface_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_surface")
        geom_indices = [j for j in range(model.ngeom) if model.geom_bodyid[j] == self.table_surface_id]
        if geom_indices:
            x, y, _ = model.geom_size[geom_indices[0]]
            self.puck_x_range = np.array([-x, x]) * 0.95 # 少し内側に
            self.puck_y_range = np.array([-y, y]) * 0.95
        else:
            print("警告: table_surface が見つかりません。パック範囲をデフォルトに設定。")
            self.puck_x_range = np.array([-0.8, 0.8])
            self.puck_y_range = np.array([-0.6, 0.6])

    def _get_puck_state(self):
        """パックの位置と速度を取得"""
        if self.puck_body_id != -1:
            puck_pos = self.data.body(self.puck_body_id).xpos.copy()
            puck_vel = self.data.body(self.puck_body_id).cvel.copy()[:3]
        else:
            # Body IDがない場合、Joint IDから推測 (精度が低い可能性)
            puck_qpos_ids = [self.puck_x_id, self.puck_y_id]
            if self.puck_z_id != -1: puck_qpos_ids.append(self.puck_z_id)
            puck_pos = self.data.qpos[puck_qpos_ids].copy()
            puck_vel = self.data.qvel[puck_qpos_ids].copy()
        
        while len(puck_pos) < 3: puck_pos = np.append(puck_pos, 0.0)
        while len(puck_vel) < 3: puck_vel = np.append(puck_vel, 0.0)

        return puck_pos[:3], puck_vel[:3]

    def _get_obs(self):
        """観測データを取得"""
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        puck_pos, puck_vel = self._get_puck_state()
        return np.concatenate([qpos, qvel, puck_pos, puck_vel])
    # --------------------------------------------------------------------------

    def _calculate_ik_control(self, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)

        ee_pos = self.data.site_xpos[self.ee_site_id]
        J_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, None, self.ee_site_id)
        ee_vel = J_pos @ self.data.qvel

        # --- ★★★ ここで target_vel の形状をチェック ★★★ ---
        if target_vel.shape != (3,):
             raise ValueError(f"IK Error: target_vel の形状が (3,) ではありません。実際の形状: {target_vel.shape}")
        # --------------------------------------------------

        command_vel = target_vel + self.Kp * (target_pos - ee_pos)
        vel_error = command_vel - ee_vel

        JJt = J_pos @ J_pos.T
        damped_inv = np.linalg.inv(JJt + self.lambda_ik**2 * np.eye(3))
        qvel_des = J_pos.T @ (damped_inv @ (command_vel + self.Kd * vel_error))

        # --- ★★★ arm_joint_ids を使って ctrl_vel を取得 ★★★ ---
        ctrl_vel = qvel_des[self.arm_joint_ids]
        # --------------------------------------------------

        qpos_arm = self.data.qpos[self.arm_joint_ids]
        for j in range(self.model.nu):
            if qpos_arm[j] < self.jnt_range[j, 0] + 0.1 and ctrl_vel[j] < 0:
                ctrl_vel[j] *= 0.1
            elif qpos_arm[j] > self.jnt_range[j, 1] - 0.1 and ctrl_vel[j] > 0:
                ctrl_vel[j] *= 0.1

        ctrl_range = self.model.actuator_ctrlrange.copy()
        ctrl_limited = np.clip(ctrl_vel, ctrl_range[:, 0], ctrl_range[:, 1])

        return ctrl_limited

    # ... ( _check_puck_hit, _compute_reward, _is_done は変更なし) ...
    def _check_puck_hit(self):
        """ラケットとパックの接触をチェック"""
        self.hit_puck_this_step = False
        if self.racket_geom_id == -1 or self.puck_geom_id == -1:
            return

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            is_hit = (geom1 == self.racket_geom_id and geom2 == self.puck_geom_id) or \
                     (geom1 == self.puck_geom_id and geom2 == self.racket_geom_id)

            if is_hit:
                self.hit_puck_this_step = True
                return

    def _compute_reward(self):
        """報酬を計算"""
        reward = 0.0
        
        if self.hit_puck_this_step:
            reward += 10.0 
            # print("ヒット！ 🏒") 

        ee_pos = self.data.site_xpos[self.ee_site_id]
        puck_pos, _ = self._get_puck_state()
        dist_to_puck = np.linalg.norm(ee_pos - puck_pos)
        reward += (1.0 - np.tanh(dist_to_puck * 5.0)) * 0.5 

        if ee_pos[2] < 0.12:
            reward -= 1.0

        return reward

    def _is_done(self):
        """終了条件をチェック"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        puck_pos, _ = self._get_puck_state()

        if ee_pos[2] < 0.1:
            return True

        if puck_pos[0] < self.puck_x_range[0] + 0.05:
            # print("失点... 🥅")
            return True

        if puck_pos[0] > self.puck_x_range[1] - 0.05:
            # print("ゴール！ 🎉")
            return True

        return False
    # ----------------------------------------------------------------------

    def step(self, action: np.ndarray):
        # --- ★★★ ここで action の形状をチェック ★★★ ---
        # print(f"DEBUG: Step received action shape: {action.shape}")
        if action.shape != (6,):
             # 複数の環境を動かしている場合、(N, 6) になることがあるので、ここではチェックを緩めるか削除
             # DummyVecEnvなら(6,)のはずだが、念のため。
             if len(action.shape) != 1 or action.shape[0] != 6:
                print(f"警告: 受け取った action の形状が (6,) ではありません！ 形状: {action.shape}")
        # --------------------------------------------------

        self.step_cnt += 1
        
        self.current_target_pos = action[:3]
        self.current_target_vel = action[3:]

        # --- ★★★ target_vel の形状をここで確認 ★★★ ---
        # print(f"DEBUG: target_pos={self.current_target_pos.shape}, target_vel={self.current_target_vel.shape}")
        # ----------------------------------------------

        ctrl = self._calculate_ik_control(self.current_target_pos, self.current_target_vel)
        self.do_simulation(ctrl, self.frame_skip)
        self._check_puck_hit()
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_done()
        truncated = self.step_cnt >= self.step_cnt_threshold
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ... ( _get_info, reset_model は変更なし) ...
    def _get_info(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        puck_pos, puck_vel = self._get_puck_state()
        return {
            "ee_pos": ee_pos,
            "puck_pos": puck_pos,
            "puck_vel": puck_vel,
            "target_pos": self.current_target_pos,
            "target_vel": self.current_target_vel,
            "hit_puck": self.hit_puck_this_step,
        }

    def reset_model(self):
        self.step_cnt = 0
        self.hit_puck_this_step = False

        qpos = self.init_qpos + np.random.uniform(-0.1, 0.1, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-0.1, 0.1, size=self.model.nv)

        theta = np.random.uniform(np.pi/4, np.pi*3/4) + np.random.choice([0, np.pi])
        qpos[self.puck_x_id] = np.random.uniform(-0.1, 0.1)
        qpos[self.puck_y_id] = np.random.uniform(*self.puck_y_range * 0.5)
        qvel[self.puck_x_id] = self.puck_speed * np.cos(theta)
        qvel[self.puck_y_id] = self.puck_speed * np.sin(theta)
        
        if self.puck_z_id != -1:
            qpos[self.puck_z_id] = 0.11
            qvel[self.puck_z_id] = 0.0

        self.set_state(qpos, qvel)
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        self.current_target_pos = ee_pos
        self.current_target_vel = np.zeros(3)

        return self._get_obs()
    # ------------------------------------------------------------------

# env.py または my_robot_hockey_env.py の末尾
def register_env():
    try:
        import gymnasium as gym

        new_env_id = 'RobotHockey-v0' # ★★★ 新しいID ★★★
        if new_env_id in gym.envs.registry:
             print(f"{new_env_id} は既に登録されています。新しい定義で上書きを試みます。")

        gym.register(
            id=new_env_id, # ★★★ 新しいID ★★★
            entry_point='myenv.env:MyRobotHockeyEnv', # あなたのファイル名に合わせてください
            max_episode_steps=500
        )
        print(f"Registered {new_env_id}")
    except Exception as e:
        print(f"環境を登録できませんでした: {e}")

register_env()