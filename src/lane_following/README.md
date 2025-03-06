# レーン検出・追従システム

このディレクトリには、ラズベリーパイとカメラを使用したレーン検出・追従システムの実装が含まれています。既存の障害物検知とモーター制御システムに、自動運転の基本的要素であるレーン追従機能を追加します。

## ファイル構成

1. **lane_detector.py**
   - 基本的なレーン検出機能
   - HSVカラー空間でのレーン色抽出
   - ハフ変換によるライン検出
   - 左右レーンの識別と処理

2. **lane_following_control.py**
   - 検出されたレーンからステアリング角度を計算
   - モーター制御機能（2つのモーターの速度制御）
   - レーンベースの自律走行制御

3. **color_marker_detector.py**
   - 特定の色（赤・緑・青・黄）のマーカー検出
   - 検出に基づいたアクション判定（停止・加速・車線変更）
   - マーカー検出の視覚化

4. **hsv_tuning.py**
   - HSV色空間の閾値調整ツール
   - レーン色検出の最適化のためのインタラクティブツール

5. **integrated_lane_follower.py**
   - すべての機能を統合したメインシステム
   - コマンドライン引数によるモード選択
   - データ可視化とデバッグ機能

## 実行方法

```bash
# 基本的なレーン検出テスト
python3 lane_detector.py

# HSV閾値調整ツール
python3 hsv_tuning.py

# カラーマーカー検出テスト
python3 color_marker_detector.py

# 統合システム（様々なオプションあり）
python3 integrated_lane_follower.py
python3 integrated_lane_follower.py --no-motors
python3 integrated_lane_follower.py --color-markers
python3 integrated_lane_follower.py --record
python3 integrated_lane_follower.py --debug
python3 integrated_lane_follower.py --color-markers --record --debug
```

## コマンドラインオプション

`integrated_lane_follower.py` で使用可能なオプション:

- `--camera <NUM>` - カメラデバイスのインデックス（デフォルト: 0）
- `--no-motors` - モーター制御を無効化（テスト用）
- `--speed <FLOAT>` - 基本速度 0-1（デフォルト: 0.5）
- `--max-steering <FLOAT>` - 最大ステアリング量（デフォルト: 0.5）
- `--steering-sensitivity <FLOAT>` - ステアリング感度（デフォルト: 1.0）
- `--color-markers` - カラーマーカー検出を有効化
- `--record` - 走行映像の録画を有効化
- `--debug` - デバッグ情報の表示

## パラメータ調整

実際の使用環境に合わせて以下のパラメータを調整してください:

### HSV色閾値

```python
# 白レーン検出用
self.white_lower = np.array([0, 0, 200], dtype=np.uint8)
self.white_upper = np.array([180, 30, 255], dtype=np.uint8)

# 黄色レーン検出用
self.yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
self.yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
```

### ハフ変換パラメータ

```python
# ハフ変換パラメータ
self.rho = 1                # 解像度（ピクセル単位）
self.theta = np.pi/180      # 角度解像度（ラジアン単位）
self.min_threshold = 20     # 最小投票数
self.min_line_length = 20   # 最小線分長
self.max_line_gap = 300     # 線分間の最大ギャップ
```

### 制御パラメータ

```python
# 制御パラメータ
self.base_speed = 0.5           # 基本速度
self.max_steering = 0.5         # 最大ステアリング量
self.steering_sensitivity = 1.0 # ステアリング感度
```

## デモ準備

効果的なデモのためのセットアップ:

1. **テストコース作成**
   - 白または黄色のテープでレーンを作成（幅5cm程度）
   - 照明条件を一定に保つ
   - 適切な幅のレーンを確保（約20-30cm）

2. **カメラ位置調整**
   - カメラがレーンを適切に捉えられるよう角度を調整
   - 解像度と処理速度のバランスを確認

3. **カラーマーカーの準備**（オプション機能）
   - 赤・緑・青・黄色のカード（約10cm×10cm）
   - 各色に対応するアクションを確認
     - 赤: 停止
     - 緑: 加速
     - 青: 車線変更
     - 黄: 通常走行に戻る

## トラブルシューティング

- **レーン検出が不安定**: HSV閾値とハフ変換パラメータを調整
- **ステアリングが過敏**: `steering_sensitivity`を下げる
- **モーターの動きがぎこちない**: 基本速度を上げる、ステアリング量を調整
- **誤検知が多い**: 関心領域（ROI）を調整、ノイズフィルタを強化

## 今後の発展

- 曲率計算によるよりスムーズなカーブ走行
- 画像の透視変換による正確なレーン検出
- 機械学習ベースのレーン検出への移行
- マルチセンサー統合（超音波センサーなど）