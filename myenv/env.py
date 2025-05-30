import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box
from typing import Dict, Union, Optional
import gymnasium as gym
import scipy


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

    # env.py (または my_robot_hockey_env.py) の MyRobotHockeyEnv クラス

    def __init__(
        self,
        xml_path="/workspace/ros2_ws/src/RL-hockey/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        kp_pos: float = 300.0,  # ★★★ 位置制御用のKp ★★★
        kd_pos: float = 10.0,   # ★★★ 位置制御用のKd (線形速度誤差に使う) ★★★
        kp_ori: float = 10.0,  # ★★★ 姿勢制御用のKp ★★★
        kd_ori: float = 10.0,    # ★★★ 姿勢制御用のKd (角速度誤差に使う) ★★★
        lambda_ik: float = 0.1,
        step_cnt_threshold: int = 500,
        puck_speed: float = 10.0,
        render_mode: Optional[str] = "rgb_array",
        max_ee_vel: float = 2.0,
        **kwargs,
    ):
        _model = mujoco.MjModel.from_xml_path(xml_path)

        # --- 他のID設定など ---
        self.puck_body_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "puck")
        self.ee_site_name = "ee_site"
        self.ee_site_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{self.ee_site_name}' not found in model.")
        self.ee_body_id = _model.site_bodyid[self.ee_site_id] # ★★★ EEサイトが属するボディのIDを取得 ★★★

        self.racket_geom_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_GEOM, "racket_head")
        self.puck_geom_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_GEOM, "puck")
        if self.racket_geom_id == -1 or self.puck_geom_id == -1:
            print("警告: ラケットまたはパックのGeom IDが見つかりません。XMLで 'racket_geom' と 'puck_geom' を定義してください。")
            self.racket_geom_id = -1
            self.puck_geom_id = -1
        self.puck_x_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "puck_x")
        self.puck_y_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "puck_y")
        self.puck_z_id = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "puck_z")
        if self.puck_x_id == -1 or self.puck_y_id == -1:
             raise ValueError("パックのJoint ID (puck_x, puck_y) が見つかりません。")

        obs_dim = _model.nq + _model.nv + 3 + 3
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        low_pos = np.array([0.0, -0.6, 0.1])
        high_pos = np.array([1.6, 0.6, 0.6])
        low_vel = np.array([-max_ee_vel] * 3)
        high_vel = np.array([max_ee_vel] * 3)
        action_low = np.concatenate([low_pos, low_vel])
        action_high = np.concatenate([high_pos, high_vel])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(6,), dtype=np.float32)
        print(f"DEBUG Env __init__: Action Space Defined (after super, explicitly set): {self.action_space}")

        self.Kp_pos = kp_pos # ★★★ 位置制御ゲイン ★★★
        self.Kd_pos = kd_pos # ★★★ 位置制御用ダンピング (線形速度誤差に適用) ★★★
        self.Kp_ori = kp_ori # ★★★ 姿勢制御ゲイン ★★★
        self.Kd_ori = kd_ori # ★★★ 姿勢制御用ダンピング (角速度誤差に適用) ★★★
        self.lambda_ik = lambda_ik
        self.jnt_range = self.model.jnt_range.copy() * 0.95
        
        self.arm_joint_ids = np.arange(3, 12)
        if len(self.arm_joint_ids) != self.model.nu:
            print(f"警告: arm_joint_ids ({len(self.arm_joint_ids)}) と nu ({self.model.nu}) が一致しません！")
        print(f"DEBUG: nu={self.model.nu}, nv={self.model.nv}, arm_ids={self.arm_joint_ids}")

        self.target_orientation_matrix = np.array([
            [-1.0,  0.0,  0.0],
            [0.0, 1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])
        # (もしマレットの初期姿勢が「垂直」で、それを維持したい場合は、
        # reset_model などで初期姿勢を取得してここに設定することもできます)

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))
        self.step_cnt_threshold = step_cnt_threshold
        # ... (残りの初期化)
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

    def _calculate_ik_control(self, target_pos: np.ndarray, target_linear_vel: np.ndarray) -> np.ndarray:
        """目標位置、目標線形速度、固定目標姿勢からIKで関節速度を計算"""
        mujoco.mj_forward(self.model, self.data) # 現在の状態を更新

        # --- 現在のエンドエフェクタの状態 ---
        current_ee_pos = self.data.site_xpos[self.ee_site_id]
        current_ee_orientation_matrix = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        
        # ヤコビアンの取得 (サイト位置ヤコビアンとボディ回転ヤコビアン)
        J_pos_site = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos_site, None, self.ee_site_id)
        
        J_rot_body = np.zeros((3, self.model.nv))
        # EEサイトがアタッチされているボディの回転ヤコビアンを使用
        mujoco.mj_jacBody(self.model, self.data, None, J_rot_body, self.ee_body_id)
        
        # 6xNv ヤコビアンの作成
        J_full = np.vstack([J_pos_site, J_rot_body])

        # 現在のEE速度 (線形速度と角速度)
        current_ee_linear_vel = J_pos_site @ self.data.qvel
        current_ee_angular_vel = J_rot_body @ self.data.qvel

        # --- 目標値と誤差の計算 ---
        # 1. 位置制御
        pos_error = target_pos - current_ee_pos
        # 指令する線形速度 = 目標線形速度(RLから) + Kp_pos * 位置誤差 - Kd_pos * 現在の線形速度 (ダンピング)
        command_linear_vel = target_linear_vel + self.Kp_pos * pos_error - self.Kd_pos * current_ee_linear_vel

        # 2. 姿勢制御
        # 目標姿勢は self.target_orientation_matrix (固定値、単位行列)
        # 目標角速度は 0 (静止)
        # 姿勢誤差 (回転ベクトルとして近似)
        # R_err = R_target * R_current^T
        error_rot_matrix = self.target_orientation_matrix @ current_ee_orientation_matrix.T
        # 回転ベクトル (軸-角度表現の近似) omega = [R_21-R_12, R_02-R_20, R_10-R_01] / 2
        # (より正確には対数写像 log(R_err) を使うか、クォータニオンで計算)
        # ここでは簡略化した誤差トルクに似たものを計算
        # error_orientation_vec = np.zeros(3)
        # error_orientation_vec[0] = error_rot_matrix[2, 1] - error_rot_matrix[1, 2] # rx
        # error_orientation_vec[1] = error_rot_matrix[0, 2] - error_rot_matrix[2, 0] # ry
        # error_orientation_vec[2] = error_rot_matrix[1, 0] - error_rot_matrix[0, 1] # rz
        # error_orientation_vec *= 0.5 # 角度に近づける

        # クォータニオンベースの姿勢誤差計算 (より安定しやすい)
        target_quat_arr = np.zeros(4) # 結果を格納する配列を事前に用意
        mujoco.mju_mat2Quat(target_quat_arr, self.target_orientation_matrix.flatten())
        target_quat = target_quat_arr # 配列を参照

        current_quat_arr = np.zeros(4) # 結果を格納する配列を事前に用意
        mujoco.mju_mat2Quat(current_quat_arr, current_ee_orientation_matrix.flatten())
        current_quat = current_quat_arr # 配列を参照
        
        # 差分クォータニオン: q_error = q_target * conjugate(q_current)
        # conjugate(q) = [q_w, -q_x, -q_y, -q_z]
        conj_current_quat = current_quat.copy()
        conj_current_quat[1:] *= -1
        error_quat = np.zeros(4)
        mujoco.mju_mulQuat(error_quat, target_quat, conj_current_quat)

        # 差分クォータニオンを軸角度誤差ベクトルに (3次元)
        # error_quat が [w, x, y, z] の場合、2 * [x, y, z] がおおよその回転ベクトル (wが1に近い場合)
        # または、mju_quat2Vel を使う (q_target, q_current から直接角速度誤差を出す方が良いかも)
        # ここでは、mju_subQuat のようなものは直接ないため、
        # ターゲットへの「回転速度」指令としてerror_quatのベクトル部を使う
        # error_orientation_vec = 2.0 * error_quat[1:]
        # if error_quat[0] < 0: # wが負なら回転方向を反転 (常に最短経路)
        #     error_orientation_vec *= -1.0
        
        # MuJoCoの差分関数を利用: target_orientation_matrix と current_orientation_matrix の間の「差」
        # を軸角度ベクトルとして得る
        # mju_subQuat はないので、mju_errorVelocity を使うか、
        # 2つの姿勢から、目標に到達するための「角速度」を計算する
        # ワールドフレームでの姿勢誤差(軸角度)
        # 実際には、mju_mat2euler, mju_subRvec などを使うか、
        # delta_axis_angle = mat_to_axis_angle(target @ current.T)
        # 以下の方法は、目標姿勢への「最短経路の角速度」を擬似的に計算する
        # 3つの軸ベクトル (x,y,z) がそれぞれ目標とどれだけズレているか
        delta_orientation = np.zeros(3)
        for i in range(3): # x, y, z軸それぞれについて
            # current_ee_orientation_matrix の i列目 (ローカルi軸のワールド表現)
            # target_orientation_matrix の i列目 (目標ローカルi軸のワールド表現)
            # この2つのベクトルの外積が、回転軸を与える
            # 内積が角度を与える
            axis_target = self.target_orientation_matrix[:, i]
            axis_current = current_ee_orientation_matrix[:, i]
            # 軸iを目標の向きに回転させるための回転ベクトル
            # delta_orientation += np.cross(axis_current, axis_target)
        # より標準的な姿勢誤差（軸角度ベクトル）
        # (target_orientation_matrix.T @ current_ee_orientation_matrix) の対数写像
        # または、(current_ee_orientation_matrix.T @ target_orientation_matrix) の対数写像
        # R_error = target_orientation_matrix @ current_ee_orientation_matrix.T
        # angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))
        # if np.abs(angle) > 1e-6: # ゼロ割を避ける
        #     axis = (1 / (2 * np.sin(angle))) * np.array([
        #         R_error[2,1] - R_error[1,2],
        #         R_error[0,2] - R_error[2,0],
        #         R_error[1,0] - R_error[0,1]
        #     ])
        #     error_orientation_vec = axis * angle
        # else:
        #     error_orientation_vec = np.zeros(3)
        
        # MuJoCo の mju_errorPose を使ってみる (ただしこれはワールドフレームでの位置・姿勢誤差を出す)
        # mju_errorPose(pos_error, error_orientation_vec, target_pos, target_quat, current_ee_pos, current_quat)
        # これだと target_pos, target_quat がワールド固定になってしまう。
        # ここでは、目標姿勢へのフィードバックを生成したい。
        # 目標角速度は0とする。
        # command_angular_vel = Kp_ori * error_orientation_vec - Kd_ori * current_ee_angular_vel
        # 以下のエラー計算は、scipy.spatial.transform.Rotation を使うのが最も堅実
        # ここでは簡略化のため、目標に「向かう」角速度を生成
        # error_orientation_vec は、目標姿勢に到達するためのワールド座標系での必要な回転
        # (自己流の簡易的な姿勢誤差)
        # R_target * R_current.transpose() から軸角度誤差ベクトルを計算
        # R_target = T, R_current = C
        # R_error = T @ C.T
        # trace = np.trace(R_error)
        # angle = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))
        # if np.isclose(angle, 0):
        #     error_orientation_vec_world = np.zeros(3)
        # else:
        #     axis = np.array([
        #         R_error[2, 1] - R_error[1, 2],
        #         R_error[0, 2] - R_error[2, 0],
        #         R_error[1, 0] - R_error[0, 1]
        #     ]) / (2 * np.sin(angle))
        #     error_orientation_vec_world = axis * angle

        # mju_subQuatがないので、クォータニオンの差から軸角度を自前で計算するか、
        # R_target と R_current の差分回転 R_delta = R_target * R_current_inv を計算し、
        # R_delta から軸角度ベクトルを取り出す。
        # R_current_inv = current_ee_orientation_matrix.T
        # R_delta = self.target_orientation_matrix @ R_current_inv
        # angle = np.arccos( (np.trace(R_delta) - 1) / 2.0 )
        # axis = np.zeros(3)
        # if np.abs(np.sin(angle)) > 1e-6: # ゼロ除算を避ける
        #     axis[0] = (R_delta[2,1] - R_delta[1,2]) / (2 * np.sin(angle))
        #     axis[1] = (R_delta[0,2] - R_delta[2,0]) / (2 * np.sin(angle))
        #     axis[2] = (R_delta[1,0] - R_delta[0,1]) / (2 * np.sin(angle))
        # error_orientation_vec = axis * angle
        # # Handle numerical precision for angle near 0 or pi
        # if np.isclose(angle, 0.0): error_orientation_vec = np.zeros(3)
        # elif np.isclose(angle, np.pi): # 180度回転の場合、軸は不定になることがある
        #     # この場合の処理は複雑なので、ここでは簡易的に対応
        #     # 例えば、R_delta の対角成分から主要な回転軸を推定するなど
        #     # 実際にはこのケースはあまり頻発しないことを期待
        #     pass

        # 姿勢誤差の計算 (より堅牢な方法、ただしscipyが必要)
        try:
            from scipy.spatial.transform import Rotation as R
            r_target = R.from_matrix(self.target_orientation_matrix)
            r_current = R.from_matrix(current_ee_orientation_matrix)
            r_error = r_target * r_current.inv()
            error_orientation_vec = r_error.as_rotvec() # 軸角度ベクトル
        except ImportError:
            print("警告: scipy がインストールされていません。姿勢制御の精度が低下する可能性があります。")
            # scipyがない場合の簡易的な代替 (上記のコメントアウト部分から持ってくるなど)
            # ここでは、エラーを0にして姿勢制御を一時的に無効化するか、簡易実装
            error_orientation_vec = np.zeros(3) # 簡易的に0とする

        command_angular_vel = self.Kp_ori * error_orientation_vec - self.Kd_ori * current_ee_angular_vel

        # --- 統合されたタスクスペース速度 ---
        task_space_desired_vel = np.concatenate([command_linear_vel, command_angular_vel])

        # --- DLS法 (減衰付き最小二乗法) で目標関節速度を計算 ---
        # qvel_des = J_full_transpose @ inv(J_full @ J_full_transpose + damping_factor^2 * I) @ task_space_vel
        try:
            J_t = J_full.T
            A = J_full @ J_t + self.lambda_ik**2 * np.eye(6) # 6x6 matrix
            qvel_des = J_t @ np.linalg.solve(A, task_space_desired_vel)
        except np.linalg.LinAlgError:
            print("IK計算で特異姿勢または数値エラーが発生しました。qvel_desをゼロにします。")
            qvel_des = np.zeros(self.model.nv)


        # アクチュエータへの制御入力
        ctrl_vel = qvel_des[self.arm_joint_ids]

        # 関節リミット付近での減速 (簡易版)
        qpos_arm = self.data.qpos[self.arm_joint_ids]
        for j in range(self.model.nu): # nu はアームの関節数と仮定
            # 関節範囲は self.jnt_range[self.model.jnt_qposadr[self.arm_joint_ids[j]] ... ] などで取得するべきだが
            # self.jnt_range は全関節なので、アームの関節に対応する部分を取得する必要がある。
            # self.arm_joint_ids は qpos/qvel のインデックスなので、
            # jnt_range のインデックスとは異なる場合がある。
            # ここでは arm_joint_ids が jnt_range のインデックスと一致する部分を使っていると仮定 (要確認)
            # 正しくは: arm_qpos_joint_indices = [self.model.jnt_qposadr[joint_id_in_model] for joint_id_in_model in self.arm_actuator_joint_ids_in_model]
            # もし arm_joint_ids が qpos/qvel index なら、
            # model.jnt_bodyid, model.jnt_type などから対応する関節の range を見つける必要がある
            # ここでは、self.jnt_range の最初の model.nu 個がアームの関節に対応し、
            # self.arm_joint_ids が 0 から model.nu-1 の範囲であると仮定する (これは現在の設定と異なる)
            # 現在の arm_joint_ids = np.arange(3,12) (qpos/qvelのインデックス)
            # 対応する model の joint id を見つけ、その jnt_range を使う。
            # ここでは簡略化のため、jnt_range のインデックスと arm_joint_ids のオフセットを合わせる試み（不正確な可能性）
            # joint_idx_in_model = self.arm_joint_ids[j] # これは qvel の idx
            # 実際には actuator_id -> joint_id -> qpos_adr/dof_adr を辿る必要がある
            # ここでは、アームの関節が model.jnt_range のどこに対応するかを事前に知っている必要がある。
            # self.jnt_range は (model.njnt, 2) の形。
            # アームの各関節のID (0からnjnt-1の範囲) を特定し、それを使う。
            # actuator_joint_ids = self.model.actuator_trnid[:,0]
            # arm_actuator_joint_ids = actuator_joint_ids[self.arm_joint_ids_for_actuators] -> これも違う
            # self.arm_joint_ids は qvel のインデックス。
            # qvel インデックスからモデルのジョイントIDを逆引きするのは少し面倒。
            # model.dof_jntid[qvel_idx] でジョイントIDが得られる。
            # なので、joint_model_id = self.model.dof_jntid[self.arm_joint_ids[j]]
            # range_for_this_joint = self.jnt_range[joint_model_id]
            
            joint_model_id = self.model.dof_jntid[self.arm_joint_ids[j]] # qvel idx から model joint idx
            joint_range_for_ctrl_vel_j = self.jnt_range[joint_model_id]

            if qpos_arm[j] < joint_range_for_ctrl_vel_j[0] + 0.1 and ctrl_vel[j] < 0:
                ctrl_vel[j] *= 0.1
            elif qpos_arm[j] > joint_range_for_ctrl_vel_j[1] - 0.1 and ctrl_vel[j] > 0:
                ctrl_vel[j] *= 0.1


        # 制御入力（速度）をアクチュエータの範囲にクリップ
        ctrl_range = self.model.actuator_ctrlrange.copy() # (nu, 2)
        # self.arm_joint_ids は qvel のインデックスなので、actuator のインデックスとは直接対応しない。
        # ctrl_vel は nu 次元なので、そのままクリップできる。
        ctrl_limited = np.clip(ctrl_vel, ctrl_range[:, 0], ctrl_range[:, 1])
        if self.step_cnt == 1:
            print(f"DEBUG IK: Target Ori Mat:\n{self.target_orientation_matrix}")
            print(f"DEBUG IK: Current Ori Mat:\n{current_ee_orientation_matrix}")
            # ... (scipyを使った姿勢誤差計算の後) ...
            print(f"DEBUG IK: Error Ori Vec: {error_orientation_vec}")
            print(f"DEBUG IK: Command Angular Vel: {command_angular_vel}")
            # ... (qvel_des 計算の後) ...
            print(f"DEBUG IK: qvel_des (first 3 arm joints): {qvel_des[self.arm_joint_ids[:3]]}")
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

        # if ee_pos[2] < 0:
        #     reward -= 1.0

        return reward

    def _is_done(self):
        """終了条件をチェック"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        puck_pos, _ = self._get_puck_state()

        # if ee_pos[2] < 0.1:
        #     return True

        if puck_pos[0] < self.puck_x_range[0] + 0.05:
            # print("失点... 🥅")
            return True

        if puck_pos[0] > self.puck_x_range[1] - 0.05:
            # print("ゴール！ 🎉")
            return True

        return False
    # ----------------------------------------------------------------------

    def step(self, action: np.ndarray):
        # action は [target_pos_x,y,z, target_linear_vel_x,y,z] の6次元
        self.step_cnt += 1
        
        self.current_target_pos = action[:3]
        self.current_target_vel = action[3:6] # ★★★ 線形速度のみを取り出す ★★★

        # IKで制御入力を計算 (目標姿勢はIK内部で固定、目標角速度は0)
        ctrl = self._calculate_ik_control(self.current_target_pos, self.current_target_vel)

        self.do_simulation(ctrl, self.frame_skip)
        # ... (残りの処理は同じ) ...
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

        qpos = np.array([0,0,0, 0, -1, 0, -1, 0, -1, 0,0,0])
        qvel = self.init_qvel
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