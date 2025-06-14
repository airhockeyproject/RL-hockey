import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box
from typing import Dict, Union


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid":-1,
    "lookat": np.array([0.0, 0.0, 0.0]), # テーブルの中心(0,0,0)を見る
    "distance": 2.5,                      # 上からの距離 (ズームレベル、小さいほど近い)
    "azimuth": 90.0,                      # 水平方向の回転 (90度にするとテーブルが横長に表示される)
    "elevation": -90.0  
}


def get_joint_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def get_actuator_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def get_site_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def get_body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


class PuckModel:
    def __init__(self, model, data, x_name, y_name):
        self.model = model
        self.data = data
        self.joint_ids = [get_joint_id(model, name) for name in [x_name, y_name]]

    def get_pos(self):
        qpos = self.data.qpos
        return qpos[self.joint_ids]

    def get_vel(self):
        qvel = self.data.qvel
        return qvel[self.joint_ids]

    def set_pos(self, pos):
        qpos = np.zeros_like(self.data.qpos)
        qpos[self.joint_ids] = pos
        return qpos

    def set_vel(self, vel):
        qvel = np.zeros_like(self.data.qvel)
        qvel[self.joint_ids] = vel
        return qvel


class ArmModel:
    def __init__(self, model, data, joint_names, actuator_names, site_name):
        self.model = model
        self.data = data
        self.joint_names = joint_names
        self.actuator_names = actuator_names
        self.site_name = site_name
        self.joint_ids = [get_joint_id(model, name) for name in joint_names]
        self.actuator_ids = [get_actuator_id(model, name) for name in actuator_names]
        self.site_id = get_site_id(self.model, site_name)

    def get_pos(self):
        qpos = self.data.qpos
        return qpos[self.joint_ids]

    def get_vel(self):
        qvel = self.data.qvel
        return qvel[self.joint_ids]

    def set_pos(self, pos):
        qpos = np.zeros_like(self.data.qpos)
        qpos[self.joint_ids] = pos
        return qpos

    def set_vel(self, vel):
        qvel = np.zeros_like(self.data.qvel)
        qvel[self.joint_ids] = vel
        return qvel

    def get_site_pos(self):
        return self.data.site_xpos[self.site_id]
    
    def get_site_vel(self):
        # サイトの位置に関するヤコビアンを取得 (3 x nv の行列)
        J_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, None, self.site_id)
        
        # 線形速度 = ヤコビアン と 関節速度ベクトルの積
        site_linear_velocity = J_pos @ self.data.qvel
        
        return site_linear_velocity

    def set_ctrl(self, ctrl):
        result = np.zeros_like(self.data.ctrl)
        result[self.actuator_ids] = ctrl
        return ctrl


class MyRobotEnv(MujocoEnv):
    def __init__(
        self,
        xml_path="/workspace/ros2_ws/src/RL-hockey/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        joint_names=[
            # CRANE-X7 本体のアーム関節
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
            # 追加したマレットの関節 (Universal Joint)
            "striker_joint_1",
            "striker_joint_2",
        ],
        actuator_names=[
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
        ],
        site_name="ee_site", 
        **kwargs,
    ):
        # 一度モデルを読み込んで観測次元を取得
        model = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = model.nq-1 + model.nv-1
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # 2. PD制御ゲインを設定（これらの値は調整が必要です）
        self.kp_pos = 10.0  # タスク空間(EE)での位置制御Pゲイン
        self.kp_joint = 5.0   # 関節空間でのPD制御Pゲイン (目標速度への追従性)
        self.kd_joint = 1   # 関節空間でのPD制御Dゲイン (動きの滑らかさ、ダンピング)
        self.lambda_ik = 0.01 # IK計算の特異点回避のための減衰係数
        
        # 3. action_space を再定義
        # 目標位置(x,y,z)の範囲 (ロボットのワークスペースに合わせる)
        pos_limits = np.array([1.0, 1.0, 0.001]) 
        # 目標速度(vx,vy,vz)の範囲
        vel_limits = np.array([1.5, 1.5, 0.001])
        
        action_low = np.concatenate([-pos_limits, -vel_limits])
        action_low[2] = 0.0
        action_high = np.concatenate([pos_limits, vel_limits])
        action_high[0] = -0.7
        
        
        # 親クラスの初期化
        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode="rgb_array",
            default_camera_config=default_camera_config,
            **kwargs,
        )
        # 新しいアクション空間 (目標位置3次元 + 目標速度3次元)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

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
        self.arm = ArmModel(
            self.model, self.data, joint_names, actuator_names, site_name
        )
        self.puck = PuckModel(self.model, self.data, "puck_x", "puck_y")
        self.step_cnt = 0
        self.hit_puck_this_step = False
        self.racket_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "striker_mallet")
        self.puck_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "puck")
    def _calculate_ik_control(self, target_ee_pos: np.ndarray, target_ee_vel: np.ndarray) -> np.ndarray:
        """
        ArmModelを使い、EEの目標位置・速度から逆運動学(IK)で正規化トルクを計算する。
        姿勢制御は行わない。
        
        引数:
            target_ee_pos (np.ndarray): EEの目標ワールド座標 [x, y, z]
            target_ee_vel (np.ndarray): EEの目標ワールド線形速度 [vx, vy, vz]
            
        戻り値:
            np.ndarray: -1から1に正規化されたアクチュエータへのトルク指令値
        """
        # 1. タスク空間での目標速度を決定 (P制御 + フィードフォワード)
        current_ee_pos = self.arm.get_site_pos() # ArmModelを使用
        #print(current_ee_pos[-1])
        pos_error = target_ee_pos - current_ee_pos
        command_linear_vel = self.kp_pos * pos_error + target_ee_vel

        # 2. 逆運動学 (IK): EE目標速度 -> 関節目標速度
        J_pos = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, None, self.arm.site_id) # ArmModelからsite_idを取得
        
        # ArmModelで定義されたアクチュエータに対応するヤコビアン列のみを抽出
        # 注意: ArmModelのactuator_namesとIKで制御したい関節を一致させる必要があります
        # ここでは ArmModel の joint_ids を使いますが、制御対象アクチュエータの関節に合わせます
        J_arm = J_pos[:, self.arm.joint_ids[:6]]

        # DLS法でヤコビアンの逆行列を計算
        try:
            A = J_arm @ J_arm.T + self.lambda_ik**2 * np.eye(3)
            x = np.linalg.solve(A, command_linear_vel)
            target_joint_vel = J_arm.T @ x
        except np.linalg.LinAlgError:
            print("警告: IK計算で数値エラーが発生しました。")
            target_joint_vel = np.zeros(len(self.arm.joint_ids))

        # 3. 関節空間でのトルク計算 (PD制御)
        current_joint_vel = self.arm.get_vel() # ArmModelを使用
        vel_error = target_joint_vel - current_joint_vel[:6]
        torques = self.kp_joint * vel_error - self.kd_joint * current_joint_vel[:6]
        # 4. トルクの正規化 (-1から1の範囲へ)
        max_torques = self.model.actuator_ctrlrange[self.arm.actuator_ids, 1]
        normalized_torques = torques / max_torques
        action = np.clip(normalized_torques, -1.0, 1.0)
        
        return action
    
    def step(self, action):
        self.step_cnt += 1

        target_ee_pos = action[:3]
        feedforward_ee_vel = action[3:]

        # print(target_ee_pos[-1])
        # print(action)

        #target_ee_pos = np.array([-0.6,-0,0])
        #feedforward_ee_vel = (target_ee_pos-self.arm.get_site_pos())/1.0e05

        # IKで計算した正規化トルクを取得
        ik_action = self._calculate_ik_control(target_ee_pos, feedforward_ee_vel)
        # do_simulationにはスケーリング前のトルクを渡すのが一般的
        # ここでは ArmModel の set_ctrl を使って制御入力を設定する例を示す
        final_torques = self.arm.set_ctrl(ik_action * self.model.actuator_ctrlrange[self.arm.actuator_ids, 1])
        # print(self.puck.get_pos())
        # print(final_torques)
        self.do_simulation(final_torques, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._is_done()
        if done:
            reward -=1000
        if self.step_cnt%100 == 0:
            reward += self.step_cnt * 3
        truncated = True if self.step_cnt > self.step_cnt_threshold else False
        
        info = {}

        return obs, reward, done, truncated, info

    def _is_done(self):
        ee_pos = self.arm.get_site_pos()
        flag = ee_pos[-1] >0.02
        return flag

    def _get_obs(self):
        puck_pos = self.puck.get_pos()
        puck_vel = self.puck.get_vel()
        arm_pos = self.arm.get_pos()
        arm_vel = self.arm.get_vel()
        return np.concatenate([puck_pos, arm_pos, puck_vel, arm_vel])

    def _check_puck_hit(self):
        """ラケットとパックの接触をチェック"""
        # 毎ステップ、フラグをリセット
        self.hit_puck_this_step = False
        if self.racket_geom_id == -1 or self.puck_geom_id == -1:
            return

        # 全ての接触情報をループで確認
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # 接触しているgeomのペアがラケットとパックかどうかを判定
            is_hit = (geom1 == self.racket_geom_id and geom2 == self.puck_geom_id) or \
                     (geom1 == self.puck_geom_id and geom2 == self.racket_geom_id)

            if is_hit:
                self.hit_puck_this_step = True
                # ヒットを検知したら、それ以上ループを回す必要はない
                return
            
    def _compute_reward(self, obs, action):
        reward = 0.0

        # 1. 現在のパック位置・速度を取得
        puck_pos = self.puck.get_pos()  # [x, y]
        puck_vel = self.puck.get_vel()  # [vx, vy]
    
        # 2. アームの初期X座標を取得（予測地点）
        predicted_x = self.init_site_pos[0]
    
        # 3. パックがX軸上でアームの初期位置を通過する時のY座標を予測（R）
        if abs(puck_vel[0]) > 1e-5:
            t = (predicted_x - puck_pos[0]) / puck_vel[0]  # いつ通過するか
            predicted_y = puck_pos[1] + puck_vel[1] * t    # その時のY位置（R）
            valid_prediction = t > 0
        else:
            predicted_y = puck_pos[1]
            valid_prediction = False
    
        # 4. エンドエフェクタの現在位置
        ee_pos = self.arm.get_site_pos()  # [x, y, z]
        ee_y = ee_pos[1]
    
        # 5. 予測位置 R に近づくほど報酬を高く（ただし未来予測が有効な場合のみ）
        if valid_prediction:
            dist = abs(ee_y - predicted_y)
            reward += np.tanh((0.2 - dist) * 5.0) * 50  # 高速で近づくと最大50の報酬
        else:
            reward += 0  # 速度ゼロなどで予測不能時、なにもしない


        
        if self.hit_puck_this_step:
            reward += 10000.0 
            print("ヒット！ 🏒") 

        if self.puck.get_vel()[0] < 0:
            ee_pos = self.arm.get_site_pos()[:2]
            puck_pos, _ = self.puck.get_pos()[:2]
            dist_to_puck = np.linalg.norm(ee_pos - puck_pos)
            reward += np.tanh((0.1-dist_to_puck) * 5.0) * 30 
        
        if self.puck.get_vel()[0] > 0:
            vel = np.linalg.norm(self.arm.get_site_vel())
            reward -= 5*vel

        # reward -= 5000*self.arm.get_site_pos()[-1] # z軸の高さが高いほどペナルティ

        return reward


    def reset_model(self):
        self.step_cnt = 0
        # qvel = self.init_qvel

        # アームの初期化
        qpos = self.arm.set_pos(
            [0,-0.859,0,-1.07,0,-0.51,0.657,0]
        )
        qvel = self.arm.set_vel(
            np.random.uniform(-0.5, 0.5, size=len(self.arm.joint_ids))
        )

        # パック初期化
        theta = np.random.uniform(np.pi/2+(np.pi/12) , 3*np.pi/2-(np.pi/12)) # ランダムな角度を選択
        # theta = - np.pi
        puck_speed = np.random.uniform(2.0, 4.5)
        qpos += self.puck.set_pos(
            [
                np.random.uniform(*self.puck_x_range),
                np.random.uniform(*self.puck_y_range),
                
            ]
        )
        qvel += self.puck.set_vel(
            [
                puck_speed * np.cos(theta),
                puck_speed * np.sin(theta),
            ]
        )

        self.set_state(qpos, qvel)
        return self._get_obs()

    def configure_puck(self):
        self.table_surface_id = get_body_id(self.model, "table_surface")
        geom_indices = [
            j
            for j in range(self.model.ngeom)
            if self.model.geom_bodyid[j] == self.table_surface_id
        ]
        assert len(geom_indices) == 1
        x, y, _ = self.model.geom_size[geom_indices[0]]
        self.puck_x_range = np.array([-0.6, x*0.8])
        self.puck_y_range = np.array([-y,y]) * 0.8
