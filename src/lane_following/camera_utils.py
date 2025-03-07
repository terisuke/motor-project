import subprocess
import cv2

def setup_camera(camera_index=0):
    """
    カメラの設定を最適化する
    
    Parameters:
    camera_index (int): カメラデバイスのインデックス
    
    Returns:
    cv2.VideoCapture: 設定済みのカメラオブジェクト
    """
    # カメラの初期化
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("カメラを開けませんでした")
    
    # v4l2-ctlコマンドでカメラ設定を最適化
    try:
        # ホワイトバランス自動調整を無効化
        subprocess.run(["v4l2-ctl", "--set-ctrl=white_balance_automatic=0"], check=False)
        
        # ホワイトバランス温度を調整
        subprocess.run(["v4l2-ctl", "--set-ctrl=white_balance_temperature=4500"], check=False)
        
        # 彩度を上げる
        subprocess.run(["v4l2-ctl", "--set-ctrl=saturation=60"], check=False)
        
        # コントラストを上げる
        subprocess.run(["v4l2-ctl", "--set-ctrl=contrast=20"], check=False)
        
        # シャープネスを最大に
        subprocess.run(["v4l2-ctl", "--set-ctrl=sharpness=3"], check=False)
        
        # バックライト補正を最大に
        subprocess.run(["v4l2-ctl", "--set-ctrl=backlight_compensation=2"], check=False)
        
        print("カメラ設定の最適化が完了しました")
    except Exception as e:
        print(f"カメラ設定の最適化中にエラーが発生しました: {e}")
    
    # OpenCVの設定も試みる（カメラによって効果が異なる）
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 20)
    cap.set(cv2.CAP_PROP_SATURATION, 60)
    
    return cap