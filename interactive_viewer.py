import mujoco
import mujoco.viewer
import numpy as np
import time
import glfw

# --- 設定項目 ---
# モデルのXMLファイルのパス
XML_PATH = './assets/main.xml' 

# 座標を監視したいsiteの名前
SITE_NAME = 'ee_site'

# --- グローバル変数 ---
is_paused = False
single_step = False

# --- キーボード入力の処理を行うコールバック関数 ---
def key_callback(keycode):
    """キーが押されたときに呼び出される関数"""
    global is_paused, single_step # 関数内でのglobal宣言は正しい

    if keycode == glfw.KEY_SPACE:
        is_paused = not is_paused
        print("Simulation", "PAUSED" if is_paused else "RUNNING")
    
    elif keycode == glfw.KEY_RIGHT and is_paused:
        single_step = True
        # print("Requesting single step...") # デバッグ用に表示しても良い

# --- メイン処理 ---
try:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"モデルの読み込みに失敗しました: {e}")
    exit()

try:
    site_id = model.site(SITE_NAME).id
except KeyError:
    print(f"エラー: '{SITE_NAME}' という名前のsiteがモデル内に見つかりません。")
    exit()

# パッシブビューアを起動
with mujoco.viewer.la (model, data) as viewer:
    viewer.key_callback = key_callback
    last_print_time = time.time()

    print("--- ビューアの操作方法 ---")
    print("Spaceキー:     シミュレーションの一時停止 / 再生")
    print("右矢印キー:    一時停止中に1ステップ進める")
    print("Ctrl + 左クリック: パーツを掴んで動かす")
    print("--------------------------")
    print("ビューアを起動しました。ターミナルに1秒ごとに ee_site の座標を出力します。")

    while viewer.is_running():
        step_start = time.time()

        # ▼▼▼ 修正点：不要なglobal宣言を削除しました ▼▼▼
        # global single_step 

        # 物理シミュレーションを1ステップ進める
        if not is_paused or single_step:
            mujoco.mj_step(model, data)
            if single_step:
                single_step = False

        # 座標のリアルタイム表示
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            site_position = data.site_xpos[site_id]
            print(f"ee_site position (x, y, z): {np.round(site_position, 4)}")
            last_print_time = current_time
        
        # ビューアの描画更新
        viewer.sync()

        # リアルタイム実行のための待機
        if not is_paused:
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

print("ビューアが閉じられました。")