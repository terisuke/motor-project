import cv2
import numpy as np
from gpiozero import Motor
from time import sleep
from datetime import datetime
import os

class RecordingIntegratedController:
    def __init__(self, camera_index=0):
        # カメラの初期化
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("カメラを開けませんでした")
        
        # カメラパラメータの設定を修正
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # FPSを30に設定
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズを最小に
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # 動画保存用ディレクトリの作成
        self.video_dir = "/home/terisuke/develop/motor_project/recordings"
        os.makedirs(self.video_dir, exist_ok=True)
        
        # 検出領域の設定
        self.center_region = {
            'x': int(self.frame_width * 0.25),
            'y': int(self.frame_height * 0.25),
            'width': int(self.frame_width * 0.5),
            'height': int(self.frame_height * 0.5)
        }
        
        # 背景差分の設定を調整
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,  # より長い履歴
            varThreshold=40,  # ノイズに対する許容度を上げる
            detectShadows=False
        )
        
        # キャリブレーション関連の設定を調整
        self.baseline_brightness = None
        self.calibration_frames = 60  # キャリブレーションフレーム数を増やす
        self.frame_count = 0
        self.brightness_history = []  # 明るさの履歴を保存
        
        # モーター初期化
        self.motor1 = Motor(14, 15)
        self.motor2 = Motor(24, 23)
        
        self.is_running = False
        self.emergency_stop = False
        self.last_detection_time = datetime.now()  # 検知時刻を記録
        self.detection_cooldown = 5  # 検知後のクールダウン期間（秒）
        self.restart_delay = 3  # 再開までの待機時間（秒）
        self.motor_speed = 0.5  # 通常動作時のモーター速度
        
        # 動画ライター初期化
        self.video_writer = None
        self.init_video_writer()
        
        print("システム初期化完了")
        print(f"カメラ解像度: {self.frame_width}x{self.frame_height}")
        print(f"フレームレート: {self.fps} FPS")
        print(f"動画保存先: {self.video_dir}")

    def init_video_writer(self):
        """動画ライターの初期化（コーデックを変更）"""
        if self.video_writer is not None:
            self.video_writer.release()
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(self.video_dir, f'recording_{timestamp}.avi')
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
            True  # isColor=True を明示的に指定
        )
        print(f"録画開始: {video_path}")

    def draw_indicators(self, frame, is_object_close, score):
        """フレームに情報を描画"""
        x, y = self.center_region['x'], self.center_region['y']
        w, h = self.center_region['width'], self.center_region['height']
        
        # 検出領域の描画
        color = (0, 0, 255) if is_object_close else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # 状態とスコアの表示
        status = "物体検知" if is_object_close else "通常運転"
        cv2.putText(frame,
                   f"{status} - Score: {score:.1f}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   color,
                   2)
        
        return frame

    def get_movement_score(self, frame):
        """移動量のスコアを計算"""
        x, y = self.center_region['x'], self.center_region['y']
        w, h = self.center_region['width'], self.center_region['height']
        roi = frame[y:y+h, x:x+w]
        
        fg_mask = self.bg_subtractor.apply(roi)
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        movement_area = np.count_nonzero(fg_mask)
        max_area = w * h
        
        movement_score = (movement_area / max_area) * 100
        return min(movement_score * 1.5, 100)

    def get_proximity_score(self, frame):
        """近接度のスコアを計算（改善版）"""
        x, y = self.center_region['x'], self.center_region['y']
        w, h = self.center_region['width'], self.center_region['height']
        roi = frame[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)
        
        if self.frame_count < self.calibration_frames:
            self.brightness_history.append(current_brightness)
            if self.frame_count == self.calibration_frames - 1:
                # キャリブレーション完了時に中央値を基準値として使用
                self.baseline_brightness = np.median(self.brightness_history)
            self.frame_count += 1
            return 0
        
        # 明るさの変化を計算
        brightness_change = abs(current_brightness - self.baseline_brightness) / self.baseline_brightness
        return min(brightness_change * 100, 100)

    def get_absolute_proximity_score(self, frame):
        """絶対的な近接度を計算"""
        x, y = self.center_region['x'], self.center_region['y']
        w, h = self.center_region['width'], self.center_region['height']
        roi = frame[y:y+h, x:x+w]
        
        # HSV色空間に変換
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 暗い領域（=物体が近い可能性が高い）を検出
        dark_mask = cv2.inRange(hsv, 
                           np.array([0, 0, 0]), 
                           np.array([180, 255, 100]))  # Value閾値を調整
        
        # 暗い領域の割合を計算
        dark_ratio = np.count_nonzero(dark_mask) / (w * h)
        
        # スコアに変換（暗い領域が多いほど高スコア）
        return min(dark_ratio * 200, 100)  # 係数は環境に応じて調整

    def detect_proximity(self, frame):
        """近接検知の処理（絶対的な距離も考慮）"""
        movement_score = self.get_movement_score(frame)
        change_score = self.get_proximity_score(frame)
        absolute_score = self.get_absolute_proximity_score(frame)
        
        # スコアの重み付けを調整
        total_score = (
            movement_score * 0.3 +  # 動きの検出
            change_score * 0.3 +    # 明るさの変化
            absolute_score * 0.4    # 絶対的な暗さ
        )
        
        # クールダウン期間中は検知を無視
        if (datetime.now() - self.last_detection_time).total_seconds() < self.detection_cooldown:
            return False, total_score
            
        is_close = total_score > 20.0
        
        if is_close:
            self.last_detection_time = datetime.now()
        
        return is_close, total_score

    def quick_detect_proximity(self, frame):
        """高速な近接検知（簡易版）"""
        # クールダウン期間中は検知を無視
        if (datetime.now() - self.last_detection_time).total_seconds() < self.detection_cooldown:
            return False
            
        x, y = self.center_region['x'], self.center_region['y']
        w, h = self.center_region['width'], self.center_region['height']
        roi = frame[y:y+h, x:x+w]
        
        # グレースケールの平均値のみをチェック
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # 急激な暗さの変化をチェック
        if hasattr(self, 'last_quick_brightness'):
            if abs(avg_brightness - self.last_quick_brightness) > 30:  # 閾値は調整可能
                self.last_quick_brightness = avg_brightness
                self.last_detection_time = datetime.now()  # 検知時刻を更新
                return True
        
        self.last_quick_brightness = avg_brightness
        return False

    def handle_detection(self):
        """物体検知時の共通処理"""
        if self.is_running:
            self.emergency_stop = True
            self.stop_motors()
            print("\n物体を検知しました - 停止")
            print(f"{self.restart_delay}秒後に再開します...")
            sleep(self.restart_delay)
            self.emergency_stop = False
            print("モーター再開")
            self.set_motors(self.motor_speed)

    def set_motors(self, speed):
        """モーターの速度設定（両方のモーターの方向を反転）"""
        if not self.emergency_stop:
            if speed > 0:
                # 前進時は両方のモーターを逆回転
                self.motor1.backward(speed)  # forwardからbackwardに変更
                self.motor2.backward(speed)  # forwardからbackwardに変更
            elif speed < 0:
                # 後進時は両方のモーターを順回転
                self.motor1.forward(-speed)  # backwardからforwardに変更
                self.motor2.forward(-speed)  # backwardからforwardに変更
            self.is_running = speed != 0
        else:
            self.stop_motors()

    def stop_motors(self):
        """モーターの停止"""
        self.motor1.stop()
        self.motor2.stop()
        self.is_running = False

    def run(self):
        """メインループ（応答性改善版）"""
        try:
            print("システム開始")
            print("Ctrl+Cで終了")
            
            # キャリブレーション期間
            print("キャリブレーション中...")
            for _ in range(self.calibration_frames):
                ret, frame = self.cap.read()
                if not ret:
                    print("フレーム取得エラー")
                    return
                _ = self.get_proximity_score(frame)
                cv2.waitKey(1)
            
            print("キャリブレーション完了")
            print(f"モーター開始（{int(self.motor_speed*100)}%スピード）")
            self.is_running = True
            self.set_motors(self.motor_speed)
            
            # バッファをクリア
            for _ in range(5):
                self.cap.read()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("フレーム取得エラー")
                    break
                
                # 物体検知（簡易版を先に実行）
                if self.quick_detect_proximity(frame):
                    self.handle_detection()
                
                # 詳細な物体検知（通常の処理）
                is_object_close, score = self.detect_proximity(frame)
                
                # フレームに情報を描画と保存
                frame_with_info = self.draw_indicators(frame.copy(), is_object_close, score)
                self.video_writer.write(frame_with_info)
                
                # スコア表示
                print(f"\r検知スコア: {score:.1f}", end="")
                
                # 通常の検知処理
                if is_object_close:
                    self.handle_detection()
                
                cv2.waitKey(1)
                
        finally:
            self.cleanup()

    def cleanup(self):
        """終了処理"""
        self.stop_motors()
        if self.video_writer is not None:
            self.video_writer.release()
        self.cap.release()
        print("\nシステムを終了しました")
        print(f"録画ファイルは {self.video_dir} に保存されています")

def main():
    try:
        controller = RecordingIntegratedController()
        controller.run()
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()