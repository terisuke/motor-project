import cv2
import numpy as np

class CameraController:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("カメラを開けませんでした")
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 中心領域の定義（画面中央の50%の領域）
        self.center_region = {
            'x': int(self.frame_width * 0.25),
            'y': int(self.frame_height * 0.25),
            'width': int(self.frame_width * 0.5),
            'height': int(self.frame_height * 0.5)
        }
        
        # 背景差分用のオブジェクトを初期化
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,           # より短いヒストリー
            varThreshold=25,      # より厳密な閾値
            detectShadows=False
        )
        
        # 前フレームの保存用
        self.prev_frame = None
        
        # ベースラインの輝度を保存
        self.baseline_brightness = None
        self.calibration_frames = 30
        self.frame_count = 0
        
        print(f"カメラ解像度: {self.frame_width}x{self.frame_height}")
        print(f"検出領域: x={self.center_region['x']}, y={self.center_region['y']}, "
              f"width={self.center_region['width']}, height={self.center_region['height']}")

    def get_movement_score(self, frame):
        """移動量のスコアを計算（改善版）"""
        x, y = self.center_region['x'], self.center_region['y']
        w, h = self.center_region['width'], self.center_region['height']
        roi = frame[y:y+h, x:x+w]
        
        # 背景差分を適用
        fg_mask = self.bg_subtractor.apply(roi)
        
        # ノイズ除去
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 移動物体の面積を計算
        movement_area = np.count_nonzero(fg_mask)
        max_area = w * h
        
        # より厳密なスケーリング
        movement_score = (movement_area / max_area) * 100
        return min(movement_score * 1.5, 100)  # 2.0から1.5に調整

    def get_proximity_score(self, frame):
        """近接度のスコアを計算（改善版）"""
        x, y = self.center_region['x'], self.center_region['y']
        w, h = self.center_region['width'], self.center_region['height']
        roi = frame[y:y+h, x:x+w]
        
        # グレースケールに変換
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 現在の平均輝度を計算
        current_brightness = np.mean(gray)
        
        # キャリブレーション期間中
        if self.frame_count < self.calibration_frames:
            if self.baseline_brightness is None:
                self.baseline_brightness = current_brightness
            else:
                self.baseline_brightness = (self.baseline_brightness * self.frame_count + current_brightness) / (self.frame_count + 1)
            self.frame_count += 1
            return 0
        
        # 輝度の変化率を計算
        brightness_change = abs(current_brightness - self.baseline_brightness) / self.baseline_brightness
        proximity_score = min(brightness_change * 100, 100)
        
        return proximity_score

    def detect_proximity(self, frame):
        """近接検知のメイン処理（改善版）"""
        movement_score = self.get_movement_score(frame)
        proximity_score = self.get_proximity_score(frame)
        
        # スコアの重み付け（移動を重視）
        total_score = movement_score * 0.6 + proximity_score * 0.4  # 0.7/0.3から0.6/0.4に調整
        
        # デバッグ情報
        print(f"移動スコア: {movement_score:.2f}, 近接スコア: {proximity_score:.2f}, "
              f"合計スコア: {total_score:.2f}")
        
        # より厳密な閾値判定
        is_close = total_score > 25.0  # 閾値を30から25に調整
        return is_close, total_score

    def draw_detection_area(self, frame, is_object_close, score):
        x, y = self.center_region['x'], self.center_region['y']
        w, h = self.center_region['width'], self.center_region['height']
        
        # スコアに基づいて色をグラデーション
        if is_object_close:
            intensity = min(score / 100.0, 1.0)
            color = (0, int(255 * (1 - intensity)), int(255 * intensity))
            status = "物体が近くにあります！"
        else:
            color = (0, 255, 0)
            status = "通常状態"
        
        # 検出領域の描画
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # ステータステキストの描画
        cv2.putText(frame, f"{status} (スコア: {score:.1f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 
                    cv2.LINE_AA)
        
        return frame

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("フレームの取得に失敗しました")
                    break
                
                # 近接検知
                is_object_close, score = self.detect_proximity(frame)
                
                # 検出領域と状態を描画
                frame = self.draw_detection_area(frame, is_object_close, score)
                
                # キャプチャした画像を表示
                cv2.imshow('Camera', frame)
                
                # 'q'キーで終了
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.close()

    def close(self):
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