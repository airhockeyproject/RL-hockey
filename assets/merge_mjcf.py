from dm_control import mjcf
import numpy as np
import mujoco.viewer  # optional for visualization


def read_model_with_preserved_paths(path):
    with open(path, "r") as f:
        xml = f.read()
    return mjcf.from_xml_string(xml)

def main():
    arm_path = "/workspace/ros2_ws/src/airhockey2025/ka/assets/crane_x7.xml"
    table_path = "/workspace/ros2_ws/src/airhockey2025/ka/assets/patched_table.xml"

    # 2つの MJCF モデルをロード
    # arm_model = mjcf.from_path(arm_path)
    # table_model = mjcf.from_path(table_path)
    arm_model = read_model_with_preserved_paths(arm_path)
    table_model = read_model_with_preserved_paths(table_path)


    # ルートの空の世界（Arena）を作成
    arena = mjcf.RootElement()
    arena.worldbody.add('light', pos=(0, 0, 5), dir=(0, 0, -1))

    arena.attach(arm_model)
    arena.attach(table_model)
    print(table_model.to_xml_string())

    # table_model.attach(arm_model)
    # print(table_model.to_xml_string())
    
if __name__ == "__main__":
    main()
