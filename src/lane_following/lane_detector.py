import cv2
import numpy as np
import argparse
import time

class LaneDetector:
    def __init__(self, camera_index=0):
        """
        レーン検出クラスの初期化
        
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
        # 白レーン検出用
        self.white_lower = np.array([0, 0, 200], dtype=np.uint8)
        self.white_upper = np.array([180, 30, 255], dtype=np.uint8)
        
        # 黄色レーン検出用
        self.yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
        self.yellow_upper = np.array([30, 255, 255], dtype=np.uint8)
        
        # ハフ変換パラメータの設定
        self.rho = 1                # 解像度（ピクセル単位）
        self.theta = np.pi/180      # 角度解像度（ラジアン単位）
        self.min_threshold = 20     # 最小投票数
        self.min_line_length = 20   # 最小線分長
        self.max_line_gap = 300     # 線分間の最大ギャップ
        
        # 関心領域（ROI）の設定
        # 画像の下半分を関心領域とする
        self.roi_vertices = np.array([
            [0, self.frame_height],
            [0, self.frame_height * 0.6],
            [self.frame_width, self.frame_height * 0.6],
            [self.frame_width, self.frame_height]
        ], dtype=np.int32)
        
        print(f"カメラ解像度: {self.frame_width}x{self.frame_height}")
        print("レーン検出器の初期化完了")
        
    def apply_roi(self, image):
        """
        関心領域（ROI）を適用する
        
        Parameters:
        image (numpy.ndarray): 入力画像
        
        Returns:
        numpy.ndarray: 関心領域が適用された画像
        """
        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        cv2.fillPoly(mask, [self.roi_vertices], ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
        
    def detect_lane_lines(self, frame):
        """
        フレームからレーン線を検出する
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        
        Returns:
        tuple: 左右のレーン線の情報（lines, left_lane, right_lane, lane_image）
        """
        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # HSVに変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 白と黄色のレーンをマスク
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # マスクを適用
        masked = cv2.bitwise_and(gray, combined_mask)
        
        # エッジ検出（Canny）
        edges = cv2.Canny(masked, 50, 150)
        
        # 関心領域を適用
        roi_edges = self.apply_roi(edges)
        
        # ハフ変換によるライン検出
        lines = cv2.HoughLinesP(
            roi_edges,
            self.rho,
            self.theta,
            self.min_threshold,
            np.array([]),
            self.min_line_length,
            self.max_line_gap
        )
        
        # 左右のレーンを保存する変数
        left_lane = []
        right_lane = []
        
        # 検出されたラインをフレームに描画
        lane_image = np.zeros_like(frame)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 垂直に近いラインは除外
                if x2 - x1 == 0:
                    continue
                
                # 傾きを計算
                slope = (y2 - y1) / (x2 - x1)
                
                # 水平に近いラインは除外
                if abs(slope) < 0.3:
                    continue
                
                # 左右のレーンを区別
                if slope < 0:  # 負の傾き = 左レーン
                    left_lane.append(line[0])
                    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 赤色
                else:  # 正の傾き = 右レーン
                    right_lane.append(line[0])
                    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 緑色
        
        return lines, left_lane, right_lane, lane_image
    
    def get_lane_center_offset(self, left_lane, right_lane):
        """
        左右のレーンからセンターオフセットを計算する
        
        Parameters:
        left_lane (list): 左レーンのライン情報
        right_lane (list): 右レーンのライン情報
        
        Returns:
        float: センターオフセット値（-1.0～1.0の範囲、0が中央）
        """
        frame_center = self.frame_width / 2
        
        if not left_lane and not right_lane:
            return 0.0  # レーンが検出されない場合は中央を維持
        
        # 左レーンの中心線を計算
        left_center = 0
        if left_lane:
            left_x_sum = sum([x1 + x2 for x1, _, x2, _ in left_lane])
            left_center = left_x_sum / (len(left_lane) * 2)
        
        # 右レーンの中心線を計算
        right_center = self.frame_width
        if right_lane:
            right_x_sum = sum([x1 + x2 for x1, _, x2, _ in right_lane])
            right_center = right_x_sum / (len(right_lane) * 2)
        
        # レーンの中心を計算
        lane_center = (left_center + right_center) / 2
        
        # フレーム中心からのオフセットを計算し、-1.0～1.0の範囲に正規化
        center_offset = (lane_center - frame_center) / (frame_width / 2)
        
        # オフセットを-1.0～1.0の範囲に制限
        center_offset = max(-1.0, min(1.0, center_offset))
        
        return center_offset
    
    def draw_lane_overlay(self, frame, left_lane, right_lane, center_offset=0.0):
        """
        レーン情報をフレームに描画する
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        left_lane (list): 左レーンのライン情報
        right_lane (list): 右レーンのライン情報
        center_offset (float): センターオフセット値
        
        Returns:
        numpy.ndarray: レーン情報が描画されたフレーム
        """
        overlay = np.zeros_like(frame)
        
        # 左右のレーンに基づいて塗りつぶし領域を作成
        if left_lane and right_lane:
            # レーンの平均線を取得
            left_points = []
            right_points = []
            
            for line in left_lane:
                x1, y1, x2, y2 = line
                left_points.extend([(x1, y1), (x2, y2)])
            
            for line in right_lane:
                x1, y1, x2, y2 = line
                right_points.extend([(x1, y1), (x2, y2)])
            
            # y座標でソート（下から上へ）
            left_points = sorted(left_points, key=lambda p: -p[1])
            right_points = sorted(right_points, key=lambda p: -p[1])
            
            # 簡単のために最初と最後のポイントのみを使用
            if len(left_points) > 1 and len(right_points) > 1:
                pts = np.array([
                    left_points[0],
                    left_points[-1],
                    right_points[-1],
                    right_points[0]
                ], dtype=np.int32)
                
                # 緑色の半透明の塗りつぶし
                cv2.fillPoly(overlay, [pts], (0, 200, 0, 128))
        
        # オーバーレイをフレームに合成
        result = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)
        
        # 中央線とオフセットを描画
        cv2.line(result, 
                (int(self.frame_width / 2), self.frame_height),
                (int(self.frame_width / 2), int(self.frame_height * 0.6)),
                (255, 255, 0), 2)  # 黄色の中央線
        
        # オフセット方向を矢印で表示
        arrow_start = (int(self.frame_width / 2), int(self.frame_height * 0.8))
        arrow_end = (int(self.frame_width / 2 + center_offset * 50), 
                    int(self.frame_height * 0.8))
        cv2.arrowedLine(result, arrow_start, arrow_end, (0, 0, 255), 2)
        
        # オフセット値のテキスト表示
        cv2.putText(result, f"オフセット: {center_offset:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return result
    
    def run(self, display=True):
        """
        レーン検出のメインループ
        
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
                
                # レーン検出
                lines, left_lane, right_lane, lane_image = self.detect_lane_lines(frame)
                
                # センターオフセットの計算
                center_offset = self.get_lane_center_offset(left_lane, right_lane)
                
                # 結果を表示
                if display:
                    # レーン検出結果を表示
                    lane_overlay = cv2.addWeighted(frame, 0.8, lane_image, 1.0, 0)
                    
                    # レーン情報を描画
                    result = self.draw_lane_overlay(lane_overlay, left_lane, right_lane, center_offset)
                    
                    # 処理時間を表示
                    process_time = (time.time() - start_time) * 1000
                    fps = 1.0 / (time.time() - start_time)
                    cv2.putText(result, f"処理時間: {process_time:.1f}ms, FPS: {fps:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.imshow('Lane Detection', result)
                    
                    # 'q'キーで終了
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='レーン検出')
    parser.add_argument('--camera', type=int, default=0,
                        help='カメラデバイスのインデックス')
    args = parser.parse_args()
    
    try:
        detector = LaneDetector(camera_index=args.camera)
        detector.run()
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()