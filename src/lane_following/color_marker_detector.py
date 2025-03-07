import cv2
import numpy as np
import argparse
import time
from camera_utils import setup_camera

class ColorMarkerDetector:
    def __init__(self, camera_index=0, external_camera=None):
        """
        色マーカー検出クラスの初期化
        
        Parameters:
        camera_index (int): カメラデバイスのインデックス
        external_camera (cv2.VideoCapture, optional): 外部から渡されるカメラオブジェクト
        """
        # 外部カメラが指定されている場合はそれを使用
        if external_camera is not None:
            self.cap = external_camera
            self.external_camera = True
            print("外部カメラオブジェクトを使用します")
        else:
            # カメラの初期化
            try:
                # setup_cameraは常にタプル (cv2.VideoCapture, 設定dict) を返す
                camera_result = setup_camera(camera_index, optimize_for='color_marker')
                self.cap = camera_result[0]  # 最初の要素がカメラオブジェクト
                self.camera_settings = camera_result[1] if len(camera_result) > 1 else {}
                print("カメラ設定を適用しました")
            except Exception as e:
                print(f"カメラ設定エラー: {e}")
                # エラー時は標準初期化を試みる
                self.cap = cv2.VideoCapture(camera_index)
            
            self.external_camera = False
            
            if not self.cap.isOpened():
                raise RuntimeError("カメラを開けませんでした")
            
            # カメラの色調整 - v4l2-ctlの出力に基づいて調整
            try:
                # 自動ホワイトバランスを無効化
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                
                # ホワイトバランス温度を調整（5000は中間値、高くすると暖色系、低くすると寒色系）
                self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5500)  # 少し暖かい色調に
                
                # 明るさを調整（-255〜255の範囲、デフォルト10）
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)  # 少し明るく
                
                # 彩度を調整（0〜127の範囲、デフォルト36）
                self.cap.set(cv2.CAP_PROP_SATURATION, 45)  # 少し彩度を上げる
                
                # コントラストを調整（0〜30の範囲、デフォルト15）
                self.cap.set(cv2.CAP_PROP_CONTRAST, 18)  # 少しコントラストを上げる
                
                print("カメラパラメータを調整しました")
            except Exception as e:
                print(f"カメラパラメータの調整に失敗しました: {e}")
                print("デフォルト設定で続行します")
            
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
        self.blue_lower = np.array([110, 120, 120], dtype=np.uint8)
        self.blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        
        # 黄色
        self.yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
        self.yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
        
        # 検出感度設定（環境によって調整が必要）
        self.min_area = 500  # 最小マーカー面積（環境に合わせて調整）
        self.detection_cooldown = 1.5  # 検出間のクールダウン時間（秒）
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
        tuple: (検出結果, マスク, 輪郭, 中心座標, 中央からの距離)
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
        
        # フレームの中心を計算
        h, w = frame.shape[:2]
        frame_center_x = w // 2
        frame_center_y = h // 2
        
        # 最も中央に近い輪郭を検出
        best_contour = None
        min_distance = float('inf')
        best_area = 0
        center = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # 輪郭の中心を計算
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 中央からの距離を計算
                    distance = np.sqrt((cx - frame_center_x)**2 + (cy - frame_center_y)**2)
                    
                    # 中央に近く、十分な大きさがある輪郭を選択
                    if distance < min_distance:
                        min_distance = distance
                        best_contour = contour
                        best_area = area
                        center = (cx, cy)
        
        # 検出されたマーカーの中心を計算
        if best_contour is not None:
            # クールダウン時間を確認
            current_time = time.time()
            if color_name not in self.last_detection_time or \
               (current_time - self.last_detection_time.get(color_name, 0)) > self.detection_cooldown:
                self.detected_markers[color_name] = True
                self.last_detection_time[color_name] = current_time
            
        return (best_contour is not None and best_area > self.min_area), mask, best_contour, center, min_distance if best_contour is not None else float('inf')
    
    def detect_red_marker(self, frame):
        """
        赤色マーカーを検出（赤は色相環の両端にあるため特殊処理）
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        
        Returns:
        tuple: (検出結果, マスク, 輪郭, 中心座標, 中央からの距離)
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
        
        # フレームの中心を計算
        h, w = frame.shape[:2]
        frame_center_x = w // 2
        frame_center_y = h // 2
        
        # 最も中央に近い輪郭を検出
        best_contour = None
        min_distance = float('inf')
        best_area = 0
        center = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # 輪郭の中心を計算
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 中央からの距離を計算
                    distance = np.sqrt((cx - frame_center_x)**2 + (cy - frame_center_y)**2)
                    
                    # 中央に近く、十分な大きさがある輪郭を選択
                    if distance < min_distance:
                        min_distance = distance
                        best_contour = contour
                        best_area = area
                        center = (cx, cy)
        
        # 検出されたマーカーの中心を計算
        if best_contour is not None:
            # クールダウン時間を確認
            current_time = time.time()
            if 'red' not in self.last_detection_time or \
               (current_time - self.last_detection_time.get('red', 0)) > self.detection_cooldown:
                self.detected_markers['red'] = True
                self.last_detection_time['red'] = current_time
        
        return (best_contour is not None and best_area > self.min_area), mask, best_contour, center, min_distance if best_contour is not None else float('inf')
    
    def detect_all_markers(self, frame=None):
        """
        すべての色マーカーを検出
        
        Parameters:
        frame (numpy.ndarray, optional): 入力フレーム。Noneの場合はカメラから取得
        
        Returns:
        dict: 検出結果（検出されたマーカーの情報を含む辞書）
        """
        # フレームが指定されていない場合はカメラから取得
        if frame is None:
            if not self.external_camera:
                ret, frame = self.cap.read()
                if not ret:
                    print("フレームの取得に失敗しました")
                    return {}
            else:
                print("外部カメラ使用時はフレームを指定してください")
                return {}
                
        # 関心領域（ROI）の設定
        # 画面中央のさらに小さい領域を検出対象とする（中央のマーカーのみを認識）
        h, w = frame.shape[:2]
        
        # 画面中央を計算
        center_x = w // 2
        center_y = h // 2
        
        # ROIを中央に集中させる
        roi_w = w // 5  # 幅をさらに狭くする
        roi_h = h // 5  # 高さをさらに狭くする
        roi_x = center_x - (roi_w // 2)  # 中央に配置
        roi_y = center_y - (roi_h // 2)  # 中央に配置
        
        # ROIを抽出
        roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # 各マーカーの検出結果をリセット
        self.detected_markers = {
            'red': False,
            'green': False,
            'blue': False,
            'yellow': False
        }
        
        # ROI内でマーカー検出
        red_result, red_mask, red_contour, red_center, red_distance = self.detect_red_marker(roi_frame)
        green_result, green_mask, green_contour, green_center, green_distance = self.detect_color(
            roi_frame, self.green_lower, self.green_upper, 'green')
        blue_result, blue_mask, blue_contour, blue_center, blue_distance = self.detect_color(
            roi_frame, self.blue_lower, self.blue_upper, 'blue')
        yellow_result, yellow_mask, yellow_contour, yellow_center, yellow_distance = self.detect_color(
            roi_frame, self.yellow_lower, self.yellow_upper, 'yellow')
        
        # 中心座標をROIから全体フレームの座標に変換
        if red_center:
            red_center = (red_center[0] + roi_x, red_center[1] + roi_y)
        if green_center:
            green_center = (green_center[0] + roi_x, green_center[1] + roi_y)
        if blue_center:
            blue_center = (blue_center[0] + roi_x, blue_center[1] + roi_y)
        if yellow_center:
            yellow_center = (yellow_center[0] + roi_x, yellow_center[1] + roi_y)
        
        # 検出結果を辞書にまとめる
        results = {
            'red': {
                'detected': red_result,
                'mask': red_mask,
                'contour': red_contour,
                'center': red_center,
                'distance': red_distance
            },
            'green': {
                'detected': green_result,
                'mask': green_mask,
                'contour': green_contour,
                'center': green_center,
                'distance': green_distance
            },
            'blue': {
                'detected': blue_result,
                'mask': blue_mask,
                'contour': blue_contour,
                'center': blue_center,
                'distance': blue_distance
            },
            'yellow': {
                'detected': yellow_result,
                'mask': yellow_mask,
                'contour': yellow_contour,
                'center': yellow_center,
                'distance': yellow_distance
            }
        }
        
        # ROIを可視化
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 2)
        
        return results
    
    def get_marker_action(self):
        """
        検出されたマーカーに基づいてアクションを返す
        
        Returns:
        str: アクション名（'stop', 'accelerate', 'decelerate', 'start', 'none'）
        """
        # 赤：停止（最優先）
        if self.detected_markers['red']:
            # 他のすべてのマーカー検出をリセット
            for color in self.detected_markers:
                self.detected_markers[color] = False
            print("赤色マーカーを検出: 停止")
            return 'stop'
        
        # 緑：加速
        if self.detected_markers['green']:
            self.detected_markers['green'] = False  # 検出をリセット
            # 他のマーカー検出もリセット（競合を防ぐ）
            self.detected_markers['blue'] = False
            self.detected_markers['yellow'] = False
            print("緑色マーカーを検出: 加速")
            return 'accelerate'
        
        # 黄：減速
        if self.detected_markers['yellow']:
            self.detected_markers['yellow'] = False  # 検出をリセット
            # 他のマーカー検出もリセット（競合を防ぐ）
            self.detected_markers['blue'] = False
            print("黄色マーカーを検出: 減速")
            return 'decelerate'
        
        # 青：スタート
        if self.detected_markers['blue']:
            self.detected_markers['blue'] = False  # 検出をリセット
            print("青色マーカーを検出: スタート")
            return 'start'
        
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
            'blue': '減速',  # '車線変更'から'減速'に変更
            'yellow': '通常走行'
        }
        
        # 各マーカーを描画
        result_frame = frame.copy()
        
        # フレームの中心を描画
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        cv2.circle(result_frame, (center_x, center_y), 10, (255, 255, 255), 1)
        cv2.line(result_frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 255), 1)
        cv2.line(result_frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 255), 1)
        
        for color, color_results in results.items():
            if color_results['detected']:
                # 輪郭を描画
                cv2.drawContours(result_frame, [color_results['contour']], -1, colors[color], 2)
                
                # 中心を描画
                if color_results['center'] is not None:
                    cv2.circle(result_frame, color_results['center'], 5, colors[color], -1)
                    
                    # 中央からの距離を表示
                    distance_text = f"距離: {color_results['distance']:.1f}"
                    cv2.putText(result_frame, distance_text,
                               (color_results['center'][0] - 50, color_results['center'][1] + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[color], 1)
                    
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
        if self.external_camera:
            print("外部カメラ使用時はrun()メソッドは使用できません")
            return
            
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
            if not self.external_camera:
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