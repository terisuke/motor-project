import cv2
import numpy as np
import subprocess
import time
import os

def setup_camera(camera_index=0, optimize_for='lane_detection'):
    """
    カメラの設定を最適化する拡張版関数
    
    Parameters:
    camera_index (int): カメラデバイスのインデックス
    optimize_for (str): 最適化タイプ ('lane_detection', 'color_marker', 'general')
    
    Returns:
    tuple: (cv2.VideoCapture, dict) - 設定済みのカメラオブジェクトと適用された設定情報
    """
    # カメラの初期化
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("カメラを開けませんでした")
    
    # カメラの基本情報を取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"カメラ初期化: 解像度={frame_width}x{frame_height}, FPS={fps}")
    
    # 設定を保存する辞書
    settings = {
        'resolution': (frame_width, frame_height),
        'fps': fps,
        'optimize_for': optimize_for,
        'applied_settings': {}
    }
    
    # v4l2-ctlコマンドでカメラ設定を最適化（利用可能な場合）
    try:
        if optimize_for == 'lane_detection':
            # レーン検出用の設定（コントラスト重視）
            v4l2_settings = [
                ("white_balance_automatic", "0"),          # 自動ホワイトバランスを無効化
                ("white_balance_temperature", "4500"),     # ホワイトバランスを調整
                ("saturation", "60"),                      # 彩度を上げる
                ("contrast", "20"),                        # コントラストを上げる
                ("sharpness", "3"),                        # シャープネスを最大に
                ("backlight_compensation", "1")            # バックライト補正を有効化
            ]
        elif optimize_for == 'color_marker':
            # 色マーカー検出用の設定（色精度重視）
            v4l2_settings = [
                ("white_balance_automatic", "0"),          # 自動ホワイトバランスを無効化
                ("white_balance_temperature", "5500"),     # 少し暖かい色調に
                ("saturation", "70"),                      # 彩度をさらに上げる
                ("contrast", "15"),                        # コントラストを中程度に
                ("sharpness", "2"),                        # シャープネスを中程度に
                ("backlight_compensation", "2"),           # バックライト補正を最大に
                ("exposure_auto", "1"),                    # 自動露出を制限モードに
                ("exposure_absolute", "500")               # 露出値を設定
            ]
        else:  # 'general'
            # 一般的なバランスの取れた設定
            v4l2_settings = [
                ("white_balance_automatic", "0"),          # 自動ホワイトバランスを無効化
                ("white_balance_temperature", "5000"),     # ニュートラルな色温度
                ("saturation", "55"),                      # 適度な彩度
                ("contrast", "15"),                        # 標準的なコントラスト
                ("sharpness", "2"),                        # 適度なシャープネス
                ("backlight_compensation", "1")            # 標準的なバックライト補正
            ]
            
        # 設定を適用
        for setting, value in v4l2_settings:
            cmd = ["v4l2-ctl", f"--set-ctrl={setting}={value}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            success = result.returncode == 0
            settings['applied_settings'][setting] = {
                'value': value,
                'success': success
            }
            if not success and 'error' in result.stderr.lower():
                print(f"警告: {setting}の設定に失敗しました: {result.stderr.strip()}")
        
        # 成功した設定数を計算
        successful_settings = sum(1 for s in settings['applied_settings'].values() if s['success'])
        print(f"カメラ設定の最適化: {successful_settings}/{len(v4l2_settings)}の設定が適用されました")
        
    except Exception as e:
        print(f"カメラ設定の最適化中にエラーが発生しました: {e}")
        print("OpenCVを使用して基本的な設定を試みます")
    
    # OpenCVでも設定を試みる
    try:
        # 共通の設定
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズを最小に
        
        if optimize_for == 'lane_detection':
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)          # 自動ホワイトバランスを無効化
            cap.set(cv2.CAP_PROP_CONTRAST, 20)        # コントラストを上げる
            cap.set(cv2.CAP_PROP_SATURATION, 60)      # 彩度を上げる
        elif optimize_for == 'color_marker':
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)          # 自動ホワイトバランスを無効化
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)      # 明るさを少し上げる
            cap.set(cv2.CAP_PROP_CONTRAST, 15)        # コントラストを中程度に
            cap.set(cv2.CAP_PROP_SATURATION, 70)      # 彩度を上げる
        else:  # 'general'
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)          # 自動ホワイトバランスを無効化
            cap.set(cv2.CAP_PROP_CONTRAST, 15)        # 標準的なコントラスト
            cap.set(cv2.CAP_PROP_SATURATION, 55)      # 適度な彩度
        
        print("OpenCVによるカメラ設定を適用しました")
        
    except Exception as e:
        print(f"OpenCVカメラ設定の適用中にエラーが発生しました: {e}")
    
    # カメラが安定するまで数フレーム読み飛ばす
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    return cap, settings

def verify_camera_settings(cap, settings=None):
    """
    カメラの現在の設定を検証する
    
    Parameters:
    cap (cv2.VideoCapture): カメラオブジェクト
    settings (dict, optional): setup_cameraから返された設定情報
    
    Returns:
    dict: 現在のカメラ設定情報
    """
    current_settings = {}
    
    # 基本的なカメラパラメータを取得
    props = [
        ('WIDTH', cv2.CAP_PROP_FRAME_WIDTH),
        ('HEIGHT', cv2.CAP_PROP_FRAME_HEIGHT),
        ('FPS', cv2.CAP_PROP_FPS),
        ('BRIGHTNESS', cv2.CAP_PROP_BRIGHTNESS),
        ('CONTRAST', cv2.CAP_PROP_CONTRAST),
        ('SATURATION', cv2.CAP_PROP_SATURATION),
        ('HUE', cv2.CAP_PROP_HUE),
        ('AUTO_WB', cv2.CAP_PROP_AUTO_WB)
    ]
    
    for name, prop in props:
        value = cap.get(prop)
        current_settings[name] = value
    
    # テストフレームを取得して画像分析
    ret, frame = cap.read()
    if ret:
        current_settings['FRAME_AVAILABLE'] = True
        
        # HSV色空間に変換して平均値を計算
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_avg, s_avg, v_avg = cv2.mean(hsv)[:3]
        current_settings['HSV_MEAN'] = (h_avg, s_avg, v_avg)
        
        # 輝度ヒストグラムを計算
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        current_settings['BRIGHTNESS_HISTOGRAM'] = hist
        
        # 明るさの分布を簡易計算
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        current_settings['BRIGHTNESS_STATS'] = {
            'mean': brightness_mean,
            'std': brightness_std
        }
    else:
        current_settings['FRAME_AVAILABLE'] = False
    
    # 設定情報が提供された場合は比較する
    if settings:
        current_settings['MATCHES_REQUESTED'] = (
            int(current_settings['WIDTH']) == settings['resolution'][0] and
            int(current_settings['HEIGHT']) == settings['resolution'][1]
        )
    
    return current_settings

def take_test_shots(cap, num_frames=3, save_dir=None):
    """
    テスト画像を撮影して保存する
    
    Parameters:
    cap (cv2.VideoCapture): カメラオブジェクト
    num_frames (int): 撮影するフレーム数
    save_dir (str, optional): 画像を保存するディレクトリ
    
    Returns:
    list: 撮影された画像のリスト (numpy.ndarray)
    """
    frames = []
    
    # 保存ディレクトリの設定
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # フレームを撮影
    for i in range(num_frames):
        # 数フレーム読み飛ばして安定させる
        for _ in range(3):
            cap.read()
            
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            
            # ディレクトリが指定されていれば保存
            if save_dir:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"test_shot_{timestamp}_{i}.jpg")
                cv2.imwrite(filename, frame)
                print(f"テスト画像を保存しました: {filename}")
        else:
            print(f"フレーム{i+1}の取得に失敗しました")
            
        # 少し待つ
        time.sleep(0.5)
    
    return frames

def test_camera_settings():
    """カメラ設定のテスト関数"""
    try:
        print("=== カメラ設定テスト開始 ===")
        
        # 各最適化モードでテスト
        modes = ['lane_detection', 'color_marker', 'general']
        
        for mode in modes:
            print(f"\n--- {mode}モードのテスト ---")
            
            # カメラの初期化
            cap, settings = setup_camera(camera_index=0, optimize_for=mode)
            
            # 設定の検証
            current_settings = verify_camera_settings(cap, settings)
            
            print(f"解像度: {current_settings['WIDTH']}x{current_settings['HEIGHT']}")
            print(f"FPS: {current_settings['FPS']}")
            print(f"コントラスト: {current_settings['CONTRAST']}")
            print(f"彩度: {current_settings['SATURATION']}")
            
            if current_settings['FRAME_AVAILABLE']:
                h, s, v = current_settings['HSV_MEAN']
                print(f"HSV平均値: H={h:.1f}, S={s:.1f}, V={v:.1f}")
                
                b_mean, b_std = current_settings['BRIGHTNESS_STATS']['mean'], current_settings['BRIGHTNESS_STATS']['std']
                print(f"輝度統計: 平均={b_mean:.1f}, 標準偏差={b_std:.1f}")
            
            # テスト画像の撮影
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images', mode)
            frames = take_test_shots(cap, num_frames=1, save_dir=save_dir)
            
            # カメラの解放
            cap.release()
            
        print("\n=== カメラ設定テスト完了 ===")
        
    except Exception as e:
        print(f"カメラテスト中にエラーが発生しました: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_settings()