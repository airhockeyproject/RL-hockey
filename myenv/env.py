from myenv.base_env import BaseEnv
from typing import Dict, Union
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid":-1,
    "lookat": np.array([0.0, 0.0, 0.0]), # テーブルの中心(0,0,0)を見る
    "distance": 2.5,                      # 上からの距離 (ズームレベル、小さいほど近い)
    "azimuth": 90.0,                      # 水平方向の回転 (90度にするとテーブルが横長に表示される)
    "elevation": -90.0  
}

class MyRobotEnv(BaseEnv):
    def __init__(
        self,
        xml_path="/workspace//RL-hockey/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        super().__init__(xml_path, frame_skip, default_camera_config, **kwargs)
    
    def _compute_reward(self, obs, action):
        reward = 0.0
        
        if self._check_puck_hit():
            reward += 10 
            print("ヒット！ 🏒") 

        if self.puck.get_vel()[0] < 0:
            ee_pos = self.arm.get_site_pos()[:2]
            puck_pos, _ = self.puck.get_pos()[:2]
            dist_to_puck = np.linalg.norm(ee_pos - puck_pos)
            reward += np.tanh((0.1-dist_to_puck) * 5.0) * 30 
        
        if self.puck.get_vel()[0] > 0:
            vel = np.linalg.norm(self.arm.get_site_vel())
            reward -= 5*vel
        return reward