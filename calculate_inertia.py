import trimesh

# --- 設定項目 ---
# 計測したいSTLファイルのパス
# このパスはアップロードされたファイルのフルパスです
STL_PATH = 'assets/iiwas/assets/EE_mallet_foam.stl'

# --- 計算処理 ---
print(f"メッシュファイルを読み込んでいます: {STL_PATH}")
try:
    mesh = trimesh.load(STL_PATH)
except Exception as e:
    print(f"エラー: ファイルの読み込みに失敗しました。パスが正しいか確認してください。: {e}")
    exit()

# オブジェクトの軸に沿ったバウンディングボックスの寸法 [X, Y, Z] を取得
dimensions = mesh.extents

width = dimensions[0]
depth = dimensions[1]
height = dimensions[2]

print("\n--- EE_mallet_foam.stl の寸法 ---")
# m単位とcm単位の両方で表示
print(f"幅 (X軸):   {width:.4f} m  ({width * 100:.2f} cm)")
print(f"奥行 (Y軸): {depth:.4f} m  ({depth * 100:.2f} cm)")
print(f"高さ (Z軸): {height:.4f} m  ({height * 100:.2f} cm)")

print(f"\n結論として、EE_mallet_foam.stl の高さは約 {height * 100:.2f} cm です。")