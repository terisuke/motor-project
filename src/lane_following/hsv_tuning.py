import cv2
import numpy as np
import argparse
from camera_utils import setup_camera

def nothing(x):
    """スライダー用のダミー関数"""
    pass

def hsv_tuning(camera_index=0):
    """
    HSV色空間の閾値を調整するインタラクティブツール
    
    Parameters:
    camera_index (int): カメラデバイスのインデックス
    """
    # カメラの初期化
    cap = setup_camera(camera_index)
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return
    
    # ウィンドウの作成
    cv2.namedWindow('HSV Tuning')
    cv2.namedWindow('Mask')
    
    # トラックバーの作成
    cv2.createTrackbar('H Min', 'HSV Tuning', 0, 179, nothing)
    cv2.createTrackbar('H Max', 'HSV Tuning', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'HSV Tuning', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'HSV Tuning', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'HSV Tuning', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'HSV Tuning', 255, 255, nothing)
    
    # ROIの設定用トラックバー
    cv2.createTrackbar('ROI X', 'HSV Tuning', 0, 100, nothing)
    cv2.createTrackbar('ROI Y', 'HSV Tuning', 50, 100, nothing)
    cv2.createTrackbar('ROI W', 'HSV Tuning', 100, 100, nothing)
    cv2.createTrackbar('ROI H', 'HSV Tuning', 50, 100, nothing)
    
    # 色選択ボタン
    cv2.createTrackbar('Color', 'HSV Tuning', 0, 3, nothing)  # 0:白, 1:黄, 2:赤, 3:青
    
    # プリセット値をロード
    presets = {
        0: {'name': '白', 'values': [0, 179, 0, 30, 200, 255]},  # 白色
        1: {'name': '黄', 'values': [20, 40, 100, 255, 100, 255]},  # 黄色
        2: {'name': '赤', 'values': [0, 10, 100, 255, 100, 255]},  # 赤色
        3: {'name': '青', 'values': [100, 140, 100, 255, 100, 255]}  # 青色
    }
    
    # 現在の色プリセットを設定
    current_preset = 0
    
    # 色プリセットをロード
    for i, key in enumerate(['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max']):
        cv2.setTrackbarPos(key, 'HSV Tuning', presets[current_preset]['values'][i])
        
    # 結果をファイルに保存する関数
    def save_values():
        h_min = cv2.getTrackbarPos('H Min', 'HSV Tuning')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Tuning')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Tuning')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Tuning')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Tuning')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Tuning')
        
        color_name = presets[current_preset]['name']
        
        with open(f"hsv_values_{color_name}.txt", "w") as f:
            f.write(f"# {color_name}色のHSV値\n")
            f.write(f"self.{color_name.lower()}_lower = np.array([{h_min}, {s_min}, {v_min}], dtype=np.uint8)\n")
            f.write(f"self.{color_name.lower()}_upper = np.array([{h_max}, {s_max}, {v_max}], dtype=np.uint8)\n")
        
        print(f"{color_name}色のHSV値を保存しました。")
    
    print("======= HSV調整ツール =======")
    print("* トラックバーでHSV値を調整してください")
    print("* 's'キーを押すと現在の値を保存します")
    print("* 'r'キーを押すと現在のプリセットをリロードします")
    print("* 'q'キーを押すと終了します")
    print("=============================")
    
    try:
        while True:
            # フレームの取得
            ret, frame = cap.read()
            if not ret:
                print("フレームの取得に失敗しました")
                break
            
            # HSVに変換
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # トラックバーの値を取得
            h_min = cv2.getTrackbarPos('H Min', 'HSV Tuning')
            h_max = cv2.getTrackbarPos('H Max', 'HSV Tuning')
            s_min = cv2.getTrackbarPos('S Min', 'HSV Tuning')
            s_max = cv2.getTrackbarPos('S Max', 'HSV Tuning')
            v_min = cv2.getTrackbarPos('V Min', 'HSV Tuning')
            v_max = cv2.getTrackbarPos('V Max', 'HSV Tuning')
            
            # ROIの値を取得
            roi_x = cv2.getTrackbarPos('ROI X', 'HSV Tuning') / 100
            roi_y = cv2.getTrackbarPos('ROI Y', 'HSV Tuning') / 100
            roi_w = cv2.getTrackbarPos('ROI W', 'HSV Tuning') / 100
            roi_h = cv2.getTrackbarPos('ROI H', 'HSV Tuning') / 100
            
            # 選択された色を取得
            new_preset = cv2.getTrackbarPos('Color', 'HSV Tuning')
            if new_preset != current_preset:
                current_preset = new_preset
                for i, key in enumerate(['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max']):
                    cv2.setTrackbarPos(key, 'HSV Tuning', presets[current_preset]['values'][i])
            
            # ROIを計算
            h, w = frame.shape[:2]
            x = int(w * roi_x)
            y = int(h * roi_y)
            width = int(w * roi_w)
            height = int(h * roi_h)
            
            # ROIを描画
            roi_frame = frame.copy()
            cv2.rectangle(roi_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # ROIを切り出し
            if x + width <= w and y + height <= h:
                roi = hsv[y:y+height, x:x+width]
                
                # 赤色は特殊処理（色相が0と180付近の両方をカバー）
                if current_preset == 2:  # 赤色
                    lower1 = np.array([0, s_min, v_min], dtype=np.uint8)
                    upper1 = np.array([h_max, s_max, v_max], dtype=np.uint8)
                    lower2 = np.array([180-h_max, s_min, v_min], dtype=np.uint8)
                    upper2 = np.array([179, s_max, v_max], dtype=np.uint8)
                    
                    mask1 = cv2.inRange(roi, lower1, upper1)
                    mask2 = cv2.inRange(roi, lower2, upper2)
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    # 通常の色
                    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
                    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
                    mask = cv2.inRange(roi, lower, upper)
                
                # マスクされた結果を表示
                res = cv2.bitwise_and(roi, roi, mask=mask)
                
                # 現在の値を表示
                color_name = presets[current_preset]['name']
                
                # 現在の設定を表示
                info_text = f"{color_name}色: H[{h_min}-{h_max}], S[{s_min}-{s_max}], V[{v_min}-{v_max}]"
                cv2.putText(roi_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 画像を表示
                cv2.imshow('HSV Tuning', roi_frame)
                cv2.imshow('Mask', mask)
                
                # キー入力を取得
                key = cv2.waitKey(1) & 0xFF
                
                # 's'キーで保存
                if key == ord('s'):
                    save_values()
                
                # 'r'キーでリロード
                elif key == ord('r'):
                    for i, key in enumerate(['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max']):
                        cv2.setTrackbarPos(key, 'HSV Tuning', presets[current_preset]['values'][i])
                
                # 'q'キーで終了
                elif key == ord('q'):
                    break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("HSV調整ツールを終了しました")

def main():
    parser = argparse.ArgumentParser(description='HSV色空間の閾値調整ツール')
    parser.add_argument('--camera', type=int, default=0,
                        help='カメラデバイスのインデックス')
    args = parser.parse_args()
    
    try:
        hsv_tuning(camera_index=args.camera)
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()