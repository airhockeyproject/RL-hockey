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
    
def get_geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


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


class BaseEnv(MujocoEnv):
    def __init__(
        self,
        xml_path="/workspace//RL-hockey/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=None,
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
        joint_names=[
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
            "striker_joint_1",
            "striker_joint_2",
        ]
        actuator_names=[
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
        ]
        site_name="ee_site"

        # 2. PD制御ゲインを設定（これらの値は調整が必要です）
        self.kp_pos = 10.0  # タスク空間(EE)での位置制御Pゲイン
        self.kp_joint = 5.0   # 関節空間でのPD制御Pゲイン (目標速度への追従性)
        self.kd_joint = 1   # 関節空間でのPD制御Dゲイン (動きの滑らかさ、ダンピング)
        self.lambda_ik = 0.01 # IK計算の特異点回避のための減衰係数
        
        # 3. action_space を再定義
        pos_limits = np.array([1.0, 1.0, 0.001]) 
        vel_limits = np.array([1.5, 1.5, 0.001])
        action_low = np.concatenate([-pos_limits, -vel_limits])
        action_high = np.concatenate([pos_limits, vel_limits])
        action_low[2] = 0.0
        action_high[0] = -0.7
        
        
        DIM_PUCK_POS = 2
        DIM_PUCK_VEL = 2
        DIM_ARM_POS = 8
        DIM_ARM_VEL = 8
        OBS_DIM = DIM_PUCK_POS + DIM_PUCK_VEL + DIM_ARM_POS + DIM_ARM_VEL
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self.arm = ArmModel(self.model, self.data, joint_names, actuator_names, site_name)
        self.puck = PuckModel(self.model, self.data, "puck_x", "puck_y")
        self.racket_geom_id = get_geom_id(self.model, "striker_mallet")
        self.puck_geom_id = get_geom_id(self.model, "puck")
        self.table_surface_id = get_body_id(self.model, "table_surface")
        self.configure_puck()
        
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
        current_ee_pos = self.arm.get_site_pos()
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
        max_torques = self.model.actuator_ctrlrange[self.arm.actuator_ids, 1]
        normalized_torques = torques / max_torques
        action = np.clip(normalized_torques, -1.0, 1.0)
        return action
    
    def step(self, action):
        target_ee_pos = action[:3]
        feedforward_ee_vel = action[3:]
        torques = self._calculate_ik_control(target_ee_pos, feedforward_ee_vel)
        ctrl = self.arm.set_ctrl(torques * self.model.actuator_ctrlrange[self.arm.actuator_ids, 1])
        return self.get_output_for_step(action, ctrl)

    def get_output_for_step(self, action, ctrl):
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._is_done()
        return obs, reward, done, False, {}

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
        puck_and_racket = [self.puck_geom_id, self.racket_geom_id]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 in puck_and_racket and geom2 in puck_and_racket:
                return True
        return False
            
    def _compute_reward(self, obs, action):
        raise NotImplementedError


    def reset_model(self):
        """環境の状態をリセット"""
        # アームの初期化
        qpos = self.arm.set_pos(
            [0,-0.859,0,-1.07,0,-0.51,0.657,0]
        )
        qvel = self.arm.set_vel(
            np.random.uniform(-0.5, 0.5, size=len(self.arm.joint_ids))
        )

        # パック初期化
        theta = np.random.uniform(np.pi/2+(np.pi/12) , 3*np.pi/2-(np.pi/12)) # ランダムな角度を選択
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
        geom_indices = [
            j for j in range(self.model.ngeom)
            if self.model.geom_bodyid[j] == self.table_surface_id
        ]
        assert len(geom_indices) == 1
        x, y, _ = self.model.geom_size[geom_indices[0]]
        self.puck_x_range = np.array([-0.6, x*0.8])
        self.puck_y_range = np.array([-y,y]) * 0.8

if __name__=="__main__":
    BaseEnv()