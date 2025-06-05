import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box
from typing import Dict, Union


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
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

    def set_ctrl(self, ctrl):
        result = np.zeros_like(self.data.ctrl)
        result[self.actuator_ids] = ctrl
        return ctrl


class MyRobotEnv(MujocoEnv):
    def __init__(
        self,
        xml_path="/workspace/RL-hockey/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        joint_names=[
            "crane_x7_gripper_finger_a_joint",
            "crane_x7_gripper_finger_b_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_wrist_joint",
        ],
        actuator_names=[
            "crane_x7_gripper_finger_a_joint",
            "crane_x7_gripper_finger_b_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_wrist_joint",
        ],
        site_name="ee_site",
        **kwargs,
    ):
        # 一度モデルを読み込んで観測次元を取得
        model = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = model.nq + model.nv
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

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
        self.arm = ArmModel(
            self.model, self.data, joint_names, actuator_names, site_name
        )
        self.puck = PuckModel(self.model, self.data, "puck_x", "puck_y")
        self.step_cnt = 0

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

    def _is_done(self):
        ee_pos = self.arm.get_site_pos()
        flag = ee_pos[-1] < 0.4
        return flag

    def _get_obs(self):
        puck_pos = self.puck.get_pos()
        puck_vel = self.puck.get_vel()
        arm_pos = self.arm.get_pos()
        arm_vel = self.arm.get_vel()
        return np.concatenate([puck_pos, arm_pos, puck_vel, arm_vel])

    def _compute_reward(self, obs, action):
        # エンドエフェクタの位置を使用した報酬例
        ee_pos = self.arm.get_site_pos()
        return 0.7 - ee_pos[-1]

    def reset_model(self):
        self.step_cnt = 0
        qpos = self.init_qpos
        qvel = self.init_qvel

        # アームの初期化
        qpos += self.arm.set_pos(
            np.random.uniform(-0.5, 0.5, size=len(self.arm.joint_ids))
        )
        qvel += self.arm.set_vel(
            np.random.uniform(-0.5, 0.5, size=len(self.arm.joint_ids))
        )

        # パック初期化
        theta = np.random.uniform(0, 2 * np.pi)
        qpos += self.puck.set_pos(
            [
                np.random.uniform(*self.puck_x_range),
                np.random.uniform(*self.puck_y_range),
            ]
        )
        qvel += self.puck.set_pos(
            [
                self.puck_speed * np.cos(theta),
                self.puck_speed * np.sin(theta),
            ]
        )

        self.set_state(qpos, qvel)
        return self._get_obs()

    def configure_puck(self):
        self.puck_speed = 10.0
        self.table_surface_id = get_body_id(self.model, "table_surface")
        geom_indices = [
            j
            for j in range(self.model.ngeom)
            if self.model.geom_bodyid[j] == self.table_surface_id
        ]
        assert len(geom_indices) == 1
        x, y, _ = self.model.geom_size[geom_indices[0]]
        self.puck_x_range = np.array([-x, x]) * 0.8
        self.puck_y_range = np.array([-y, y]) * 0.8
