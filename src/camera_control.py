import cv2
import numpy as np

class CameraController:
    def __init__(self, camera_index=0):
        """
        カメラコントローラーの初期化
        camera_index: カメラデバイスのインデックス（通常は0）
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("カメラを開けませんでした")

    def run(self):
        """
        カメラのライブビューを実行
        'q'キーで終了
        """
        try:
            while True:
                # フレームをキャプチャ
                ret, frame = self.cap.read()
                
                if not ret:
                    print("フレームの取得に失敗しました")
                    break
                    
                # キャプチャした画像を表示
                cv2.imshow('Camera', frame)
                
                # 'q'キーで終了
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.close()

    def close(self):
        """
        カメラリソースの解放
        """
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        camera = CameraController()
        camera.run()
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main() 