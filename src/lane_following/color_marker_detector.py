import cv2
import numpy as np
import argparse
import time

class ColorMarkerDetector:
    def __init__(self, camera_index=0):
        """
        色マーカー検出クラスの初期化
        
        Parameters:
        camera_index (int): カメラデバイスのインデックス
        """
        # カメラの初期化
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("カメラを開けませんでした")
            
        # カメラパラメータの設定
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # HSV色閾値の初期化
        # 赤色（Hueが0付近と180付近の両方をカバー）
        self.red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
        self.red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        self.red_lower2 = np.array([160, 100, 100], dtype=np.uint8)
        self.red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # 緑色
        self.green_lower = np.array([40, 100, 100], dtype=np.uint8)
        self.green_upper = np.array([80, 255, 255], dtype=np.uint8)
        
        # 青色
        self.blue_lower = np.array([100, 100, 100], dtype=np.uint8)
        self.blue_upper = np.array([140, 255, 255], dtype=np.uint8)
        
        # 黄色
        self.yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
        self.yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
        
        # 検出感度設定
        self.min_area = 500  # 最小マーカー面積
        self.detection_cooldown = 1.0  # 検出間のクールダウン時間（秒）
        self.last_detection_time = {}  # 最後の検出時間を色ごとに記録
        
        # 検出結果を初期化
        self.detected_markers = {
            'red': False,
            'green': False,
            'blue': False,
            'yellow': False
        }
        
        print(f"カメラ解像度: {self.frame_width}x{self.frame_height}")
        print("色マーカー検出器の初期化完了")
    
    def detect_color(self, frame, lower, upper, color_name):
        """
        特定の色範囲のマーカーを検出
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        lower (numpy.ndarray): HSV下限値
        upper (numpy.ndarray): HSV上限値
        color_name (str): 色の名前
        
        Returns:
        tuple: (検出結果, マスク, 輪郭, 中心座標)
        """
        # HSVに変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 指定した色範囲でマスク作成
        mask = cv2.inRange(hsv, lower, upper)
        
        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 最大の輪郭を検出
        max_contour = None
        max_area = 0
        center = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > self.min_area:
                max_area = area
                max_contour = contour
        
        # 検出されたマーカーの中心を計算
        if max_contour is not None:
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)
                
                # クールダウン時間を確認
                current_time = time.time()
                if color_name not in self.last_detection_time or \
                   (current_time - self.last_detection_time.get(color_name, 0)) > self.detection_cooldown:
                    self.detected_markers[color_name] = True
                    self.last_detection_time[color_name] = current_time
            
        return (max_contour is not None and max_area > self.min_area), mask, max_contour, center
    
    def detect_red_marker(self, frame):
        """
        赤色マーカーを検出（赤は色相環の両端にあるため特殊処理）
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        
        Returns:
        tuple: (検出結果, マスク, 輪郭, 中心座標)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 赤色の2つの範囲でマスクを作成
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 最大の輪郭を検出
        max_contour = None
        max_area = 0
        center = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area > self.min_area:
                max_area = area
                max_contour = contour
        
        # 検出されたマーカーの中心を計算
        if max_contour is not None:
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)
                
                # クールダウン時間を確認
                current_time = time.time()
                if 'red' not in self.last_detection_time or \
                   (current_time - self.last_detection_time.get('red', 0)) > self.detection_cooldown:
                    self.detected_markers['red'] = True
                    self.last_detection_time['red'] = current_time
        
        return (max_contour is not None and max_area > self.min_area), mask, max_contour, center
    
    def detect_all_markers(self, frame):
        """
        すべての色マーカーを検出
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        
        Returns:
        dict: 検出結果（検出されたマーカーの情報を含む辞書）
        """
        # 各マーカーの検出結果をリセット
        self.detected_markers = {
            'red': False,
            'green': False,
            'blue': False,
            'yellow': False
        }
        
        # 各色マーカーを検出
        red_result, red_mask, red_contour, red_center = self.detect_red_marker(frame)
        green_result, green_mask, green_contour, green_center = self.detect_color(
            frame, self.green_lower, self.green_upper, 'green')
        blue_result, blue_mask, blue_contour, blue_center = self.detect_color(
            frame, self.blue_lower, self.blue_upper, 'blue')
        yellow_result, yellow_mask, yellow_contour, yellow_center = self.detect_color(
            frame, self.yellow_lower, self.yellow_upper, 'yellow')
        
        # 検出結果を辞書にまとめる
        results = {
            'red': {
                'detected': red_result,
                'mask': red_mask,
                'contour': red_contour,
                'center': red_center
            },
            'green': {
                'detected': green_result,
                'mask': green_mask,
                'contour': green_contour,
                'center': green_center
            },
            'blue': {
                'detected': blue_result,
                'mask': blue_mask,
                'contour': blue_contour,
                'center': blue_center
            },
            'yellow': {
                'detected': yellow_result,
                'mask': yellow_mask,
                'contour': yellow_contour,
                'center': yellow_center
            }
        }
        
        return results
    
    def get_marker_action(self):
        """
        検出されたマーカーに基づいてアクションを返す
        
        Returns:
        str: アクション名（'stop', 'accelerate', 'lane_change', 'normal', 'none'）
        """
        # 赤：停止
        if self.detected_markers['red']:
            self.detected_markers['red'] = False  # 検出をリセット
            return 'stop'
        
        # 緑：加速
        if self.detected_markers['green']:
            self.detected_markers['green'] = False  # 検出をリセット
            return 'accelerate'
        
        # 青：車線変更
        if self.detected_markers['blue']:
            self.detected_markers['blue'] = False  # 検出をリセット
            return 'lane_change'
        
        # 黄：通常走行に戻る
        if self.detected_markers['yellow']:
            self.detected_markers['yellow'] = False  # 検出をリセット
            return 'normal'
        
        # 何も検出されなかった場合
        return 'none'
    
    def draw_markers(self, frame, results):
        """
        検出されたマーカーを描画
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        results (dict): 検出結果
        
        Returns:
        numpy.ndarray: マーカーが描画されたフレーム
        """
        # 各色の表示カラー
        colors = {
            'red': (0, 0, 255),      # 赤
            'green': (0, 255, 0),    # 緑
            'blue': (255, 0, 0),     # 青
            'yellow': (0, 255, 255)  # 黄
        }
        
        # 各色のアクション
        actions = {
            'red': '停止',
            'green': '加速',
            'blue': '車線変更',
            'yellow': '通常走行'
        }
        
        # 各マーカーを描画
        result_frame = frame.copy()
        
        for color, color_results in results.items():
            if color_results['detected']:
                # 輪郭を描画
                cv2.drawContours(result_frame, [color_results['contour']], -1, colors[color], 2)
                
                # 中心を描画
                if color_results['center'] is not None:
                    cv2.circle(result_frame, color_results['center'], 5, colors[color], -1)
                    
                    # アクションを表示
                    text = f"{color}: {actions[color]}"
                    cv2.putText(result_frame, text,
                               (color_results['center'][0] - 50, color_results['center'][1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[color], 2)
        
        return result_frame
    
    def run(self, display=True):
        """
        マーカー検出のメインループ
        
        Parameters:
        display (bool): 結果を表示するかどうか
        
        Returns:
        None
        """
        try:
            while True:
                # フレームの取得
                ret, frame = self.cap.read()
                if not ret:
                    print("フレームの取得に失敗しました")
                    break
                
                start_time = time.time()
                
                # マーカー検出
                results = self.detect_all_markers(frame)
                
                # アクションの取得
                action = self.get_marker_action()
                
                # 結果を表示
                if display:
                    # マーカー描画
                    result_frame = self.draw_markers(frame, results)
                    
                    # アクション表示
                    if action != 'none':
                        text = f"アクション: {action}"
                        cv2.putText(result_frame, text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # 処理時間を表示
                    process_time = (time.time() - start_time) * 1000
                    fps = 1.0 / (time.time() - start_time)
                    cv2.putText(result_frame, f"処理時間: {process_time:.1f}ms, FPS: {fps:.1f}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.imshow('Color Marker Detection', result_frame)
                    
                    # 'q'キーで終了
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='色マーカー検出')
    parser.add_argument('--camera', type=int, default=0,
                        help='カメラデバイスのインデックス')
    args = parser.parse_args()
    
    try:
        detector = ColorMarkerDetector(camera_index=args.camera)
        detector.run()
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()