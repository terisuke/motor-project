import cv2
import numpy as np
import time
import argparse
from datetime import datetime
import os
import sys
from camera_utils import setup_camera

# 自作モジュールのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 自作モジュールをインポート
from src.lane_following.lane_detector import LaneDetector
from src.lane_following.lane_following_control import LaneFollowingController
from src.lane_following.color_marker_detector import ColorMarkerDetector

class IntegratedLaneFollower:
    def __init__(self, camera_index=0, no_motors=False, base_speed=0.5, 
                 max_steering=0.5, steering_sensitivity=1.0, color_markers=False,
                 record=False, debug=False):
        """
        統合レーン追従システムの初期化
        
        Parameters:
        camera_index (int): カメラデバイスのインデックス
        no_motors (bool): モーター制御を無効化するかどうか
        base_speed (float): 基本速度 (0.0-1.0)
        max_steering (float): 最大ステアリング量 (0.0-1.0)
        steering_sensitivity (float): ステアリング感度
        color_markers (bool): カラーマーカー検出を有効化するかどうか
        record (bool): 走行映像の録画を有効化するかどうか
        debug (bool): デバッグ情報の表示を有効化するかどうか
        """
        # 設定のコピー
        self.no_motors = no_motors
        self.color_markers = color_markers
        self.record = record
        self.debug = debug
        
        # カメラの初期化（一度だけ）
        camera_result = setup_camera(camera_index)
        
        # setup_cameraの戻り値がタプルの場合は最初の要素を取得
        if isinstance(camera_result, tuple):
            self.cap = camera_result[0]  # 最初の要素がカメラオブジェクト
            print("setup_cameraからタプルを受け取りました。カメラオブジェクトを抽出します。")
        else:
            self.cap = camera_result
        
        if not self.cap.isOpened():
            raise RuntimeError("カメラを開けませんでした")
            
        # カメラパラメータの設定
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # レーン検出器の初期化（外部カメラを使用）
        self.lane_detector = LaneDetector(external_camera=self.cap)
        
        # モーター制御の初期化（必要な場合）
        if not self.no_motors:
            self.motor_controller = LaneFollowingController(
                base_speed=base_speed,
                max_steering=max_steering,
                steering_sensitivity=steering_sensitivity
            )
            
        # カラーマーカー検出器の初期化（必要な場合、外部カメラを使用）
        if self.color_markers:
            self.marker_detector = ColorMarkerDetector(external_camera=self.cap)
        
        # 録画の設定（必要な場合）
        self.video_writer = None
        if self.record:
            self.init_video_writer()
        
        # 状態変数
        self.running = True
        self.current_action = 'normal'
        self.lane_change_timer = 0
        self.lane_change_direction = 'left'  # 'left' or 'right'
        self.acceleration_timer = 0
        
        # 統計情報
        self.frame_count = 0
        self.start_time = time.time()
        
        print("===== 統合レーン追従システム初期化完了 =====")
        print(f"モーター制御: {'無効' if self.no_motors else '有効'}")
        print(f"カラーマーカー検出: {'有効' if self.color_markers else '無効'}")
        print(f"録画: {'有効' if self.record else '無効'}")
        print(f"デバッグモード: {'有効' if self.debug else '無効'}")
        print("============================================")
    
    def init_video_writer(self):
        """動画ライターの初期化"""
        if self.video_writer is not None:
            self.video_writer.release()
            
        # 保存先ディレクトリの作成
        video_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'recordings')
        os.makedirs(video_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = os.path.join(video_dir, f'lane_following_{timestamp}.avi')
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
            True
        )
        print(f"録画開始: {video_path}")
    
    def process_frame(self, frame):
        """
        フレームを処理してレーン検出と操作を行う
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        
        Returns:
        tuple: (processed_frame, center_offset, steering, marker_action)
        """
        # レーン検出
        lines, left_lane, right_lane, lane_image = self.lane_detector.detect_lane_lines(frame)
        
        # センターオフセットの計算
        center_offset = self.lane_detector.get_lane_center_offset(left_lane, right_lane)
        
        # レーン情報をフレームに描画
        processed_frame = self.lane_detector.draw_lane_overlay(
            cv2.addWeighted(frame, 0.8, lane_image, 1.0, 0),
            left_lane, right_lane, center_offset
        )
        
        # ステアリング値の計算
        steering = 0.0
        if not self.no_motors:
            steering = self.motor_controller.calculate_steering(center_offset)
        
        # カラーマーカー検出（有効な場合）
        marker_action = 'none'
        if self.color_markers:
            marker_results = self.marker_detector.detect_all_markers(frame)
            
            # マーカー描画
            processed_frame = self.marker_detector.draw_markers(processed_frame, marker_results)
            
            # マーカーに基づくアクション
            marker_action = self.marker_detector.get_marker_action()
        
        return processed_frame, center_offset, steering, marker_action
    
    def apply_marker_action(self, action):
        """
        マーカーアクションに基づいて操作を実行
        
        Parameters:
        action (str): アクション ('stop', 'accelerate', 'lane_change', 'normal', 'none')
        """
        if action == 'none':
            return
            
        self.current_action = action
        
        if not self.no_motors:
            if action == 'stop':
                # 停止
                self.motor_controller.stop_motors()
                print("アクション: 停止")
                
            elif action == 'accelerate':
                # 加速（5秒間）
                self.motor_controller.adjust_speed(1.5)  # 1.5倍速
                self.acceleration_timer = time.time() + 5.0
                print("アクション: 加速（5秒間）")
                
            elif action == 'lane_change':
                # 車線変更
                # デフォルトでは左に車線変更
                self.lane_change_direction = 'left' if self.lane_change_direction == 'right' else 'right'
                
                if self.lane_change_direction == 'left':
                    self.motor_controller.turn_left(duration=1.0, intensity=0.8)
                else:
                    self.motor_controller.turn_right(duration=1.0, intensity=0.8)
                    
                self.lane_change_timer = time.time() + 2.0  # 2秒間は通常制御を一時停止
                print(f"アクション: 車線変更（{self.lane_change_direction}）")
                
            elif action == 'normal':
                # 通常走行に戻る
                self.motor_controller.set_motor_speeds(self.motor_controller.base_speed, 0.0)
                self.acceleration_timer = 0
                self.current_action = 'normal'
                print("アクション: 通常走行")
    
    def update_timers(self):
        """タイマーの更新処理"""
        current_time = time.time()
        
        # 加速タイマー
        if self.acceleration_timer > 0 and current_time > self.acceleration_timer:
            if not self.no_motors:
                self.motor_controller.adjust_speed(1.0)  # 元の速度に戻す
            self.acceleration_timer = 0
            self.current_action = 'normal'
            print("加速終了: 通常速度に戻ります")
                
        # 車線変更タイマー
        if self.lane_change_timer > 0 and current_time > self.lane_change_timer:
            self.lane_change_timer = 0
            self.current_action = 'normal'
            print("車線変更終了: 通常制御に戻ります")
    
    def draw_debug_info(self, frame, center_offset, steering, fps, process_time):
        """
        デバッグ情報を描画
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        center_offset (float): センターオフセット
        steering (float): ステアリング値
        fps (float): フレームレート
        process_time (float): 処理時間（ms）
        
        Returns:
        numpy.ndarray: デバッグ情報が描画されたフレーム
        """
        # 基本情報を描画
        cv2.putText(frame, f"センターオフセット: {center_offset:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"ステアリング: {steering:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 処理情報を描画
        cv2.putText(frame, f"FPS: {fps:.1f}, 処理時間: {process_time:.1f}ms", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 現在のアクションを描画
        action_color = (0, 255, 255)  # デフォルト色（黄色）
        
        if self.current_action == 'stop':
            action_color = (0, 0, 255)  # 赤
        elif self.current_action == 'accelerate':
            action_color = (0, 255, 0)  # 緑
        elif self.current_action == 'lane_change':
            action_color = (255, 0, 0)  # 青
            
        cv2.putText(frame, f"アクション: {self.current_action}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        
        return frame
    
    def run(self):
        """メインループ"""
        try:
            print("レーン追従システム開始")
            print("Ctrl+Cで終了")
            
            # モーター開始（モーター制御有効の場合）
            if not self.no_motors:
                self.motor_controller.start()
            
            while self.running:
                # フレームの取得
                ret, frame = self.cap.read()
                if not ret:
                    print("フレームの取得に失敗しました")
                    break
                
                start_time = time.time()
                
                # フレーム処理
                processed_frame, center_offset, steering, marker_action = self.process_frame(frame)
                
                # マーカーアクションの適用
                self.apply_marker_action(marker_action)
                
                # タイマー更新
                self.update_timers()
                
                # モーター制御（有効かつ車線変更中でない場合）
                if not self.no_motors and self.lane_change_timer == 0 and self.current_action != 'stop':
                    self.motor_controller.steer(center_offset)
                
                # 処理時間とFPSの計算
                process_time = (time.time() - start_time) * 1000
                self.frame_count += 1
                total_time = time.time() - self.start_time
                fps = self.frame_count / total_time if total_time > 0 else 0
                
                # デバッグ情報の描画（有効な場合）
                if self.debug:
                    processed_frame = self.draw_debug_info(
                        processed_frame, center_offset, steering, fps, process_time)
                
                # 結果の表示
                cv2.imshow('Lane Following', processed_frame)
                
                # 録画（有効な場合）
                if self.record and self.video_writer is not None:
                    self.video_writer.write(processed_frame)
                
                # キー入力の処理
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # 'q'キーで終了
                    self.running = False
                elif key == ord('s'):
                    # 's'キーでモーター停止/開始切り替え
                    if not self.no_motors:
                        if self.motor_controller.is_running:
                            self.motor_controller.stop_motors()
                            print("モーター停止")
                        else:
                            self.motor_controller.start()
                            print("モーター開始")
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """終了処理"""
        # モーターの停止
        if not self.no_motors:
            self.motor_controller.cleanup()
            
        # カメラの解放
        self.cap.release()
        
        # 録画の終了
        if self.record and self.video_writer is not None:
            self.video_writer.release()
            
        # ウィンドウの解放
        cv2.destroyAllWindows()
        
        print("\nレーン追従システムを終了しました")
        
        # 統計情報の表示
        total_time = time.time() - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0
        print(f"実行時間: {total_time:.1f}秒, 総フレーム数: {self.frame_count}, 平均FPS: {fps:.1f}")

def main():
    parser = argparse.ArgumentParser(description='レーン追従システム')
    parser.add_argument('--camera', type=int, default=0,
                        help='カメラデバイスのインデックス（デフォルト: 0）')
    parser.add_argument('--no-motors', action='store_true',
                        help='モーター制御を無効化（テスト用）')
    parser.add_argument('--speed', type=float, default=0.5,
                        help='基本速度 0-1（デフォルト: 0.5）')
    parser.add_argument('--max-steering', type=float, default=0.5,
                        help='最大ステアリング量（デフォルト: 0.5）')
    parser.add_argument('--steering-sensitivity', type=float, default=1.0,
                        help='ステアリング感度（デフォルト: 1.0）')
    parser.add_argument('--color-markers', action='store_true',
                        help='カラーマーカー検出を有効化')
    parser.add_argument('--record', action='store_true',
                        help='走行映像の録画を有効化')
    parser.add_argument('--debug', action='store_true',
                        help='デバッグ情報の表示')
    
    args = parser.parse_args()
    
    try:
        follower = IntegratedLaneFollower(
            camera_index=args.camera,
            no_motors=args.no_motors,
            base_speed=args.speed,
            max_steering=args.max_steering,
            steering_sensitivity=args.steering_sensitivity,
            color_markers=args.color_markers,
            record=args.record,
            debug=args.debug
        )
        
        follower.run()
        
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()