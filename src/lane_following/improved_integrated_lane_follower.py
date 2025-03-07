#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改良版 - レーン追従統合システム
拡張された機能と最適化された設定を持つ統合システム
"""

import cv2
import numpy as np
import time
import argparse
from datetime import datetime
import os
import sys
import signal
from camera_utils import setup_camera  # enhanced_camera_utils から camera_utils に変更

# 自作モジュールのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 自作モジュールをインポート
from src.lane_following.lane_detector import LaneDetector
from src.lane_following.lane_following_control import LaneFollowingController
from src.lane_following.color_marker_detector import ColorMarkerDetector

class ImprovedIntegratedLaneFollower:
    def __init__(self, camera_index=0, no_motors=False, base_speed=0.5, 
                 max_steering=0.5, steering_sensitivity=1.0, color_markers=False,
                 record=False, debug=False, data_logging=False, headless=False):
        """
        改良版 - 統合レーン追従システムの初期化
        
        Parameters:
        camera_index (int): カメラデバイスのインデックス
        no_motors (bool): モーター制御を無効化するかどうか
        base_speed (float): 基本速度 (0.0-1.0)
        max_steering (float): 最大ステアリング量 (0.0-1.0)
        steering_sensitivity (float): ステアリング感度
        color_markers (bool): カラーマーカー検出を有効化するかどうか
        record (bool): 走行映像の録画を有効化するかどうか
        debug (bool): デバッグ情報の表示を有効化するかどうか
        data_logging (bool): データログ記録を有効化するかどうか
        headless (bool): ヘッドレスモード（GUI表示なし）
        """
        # 設定のコピー
        self.no_motors = no_motors
        self.color_markers = color_markers
        self.record = record
        self.debug = debug
        self.data_logging = data_logging
        self.headless = headless
        
        # システム状態モニタリング（先に初期化）
        self.system_status = {
            'camera': 'OK',
            'lane_detection': 'N/A',
            'color_marker': 'N/A',
            'motor_control': 'N/A',
            'last_error': None
        }
        
        # モジュール初期化フラグ
        self.camera_initialized = False
        self.lane_detector_initialized = False
        self.marker_detector_initialized = False
        self.motor_controller_initialized = False
        
        # カメラの初期化（最適化設定を適用）
        print("カメラの初期化中...")
        optimize_for = 'general'
        if color_markers and not self.is_lane_detection_enabled():
            optimize_for = 'color_marker'
        elif self.is_lane_detection_enabled() and not color_markers:
            optimize_for = 'lane_detection'
            
        try:
            camera_result = setup_camera(
                camera_index=camera_index,
                optimize_for=optimize_for
            )
            
            # setup_cameraの戻り値がタプルの場合は最初の要素を取得
            if isinstance(camera_result, tuple):
                self.cap = camera_result[0]  # 最初の要素がカメラオブジェクト
                print("setup_cameraからタプルを受け取りました。カメラオブジェクトを抽出します。")
                self.camera_settings = camera_result[1] if len(camera_result) > 1 else {}
            else:
                self.cap = camera_result
                self.camera_settings = {}
            
            if not self.cap.isOpened():
                raise RuntimeError("カメラを開けませんでした")
                
            self.camera_initialized = True
                
            # カメラパラメータの取得
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        except Exception as e:
            print(f"カメラの初期化に失敗しました: {e}")
            self.system_status['camera'] = 'ERROR'
            self.system_status['last_error'] = str(e)
            raise
        
        # 安全な終了のためのシグナルハンドラ設定
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # モジュールの初期化（必要なものだけ）
        self.init_modules(
            base_speed=base_speed,
            max_steering=max_steering,
            steering_sensitivity=steering_sensitivity
        )
        
        # 録画の設定（必要な場合）
        self.video_writer = None
        if self.record:
            self.init_video_writer()
        
        # データログの設定（必要な場合）
        self.data_log = None
        self.log_file = None
        if self.data_logging:
            self.init_data_logging()
        
        # 状態変数
        self.running = True
        self.current_action = 'normal'
        self.lane_change_timer = 0
        self.lane_change_direction = 'left'  # 'left' or 'right'
        self.acceleration_timer = 0
        
        # 統計情報
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update_time = time.time()
        self.last_frames = 0
        self.current_fps = 0
        
        print("===== 改良版 - 統合レーン追従システム初期化完了 =====")
        print(f"カメラ解像度: {self.frame_width}x{self.frame_height}")
        print(f"モーター制御: {'無効' if self.no_motors else '有効'}")
        print(f"カラーマーカー検出: {'有効' if self.color_markers else '無効'}")
        print(f"録画: {'有効' if self.record else '無効'}")
        print(f"デバッグモード: {'有効' if self.debug else '無効'}")
        print(f"データログ: {'有効' if self.data_logging else '無効'}")
        print(f"ヘッドレスモード: {'有効' if self.headless else '無効'}")
        print("====================================================")
    
    def is_lane_detection_enabled(self):
        """レーン検出が有効かどうかを返す"""
        # 現在のバージョンではデフォルトで有効
        return True
    
    def init_modules(self, base_speed, max_steering, steering_sensitivity):
        """必要なモジュールの初期化"""
        # レーン検出器の初期化
        if self.is_lane_detection_enabled():
            try:
                print("レーン検出器の初期化...")
                # 外部カメラオブジェクトを渡す
                self.lane_detector = LaneDetector(external_camera=self.cap)
                self.lane_detector_initialized = True
                self.system_status['lane_detection'] = 'OK'
            except Exception as e:
                print(f"レーン検出器の初期化に失敗: {e}")
                self.system_status['lane_detection'] = 'ERROR'
                self.system_status['last_error'] = str(e)
        
        # モーター制御の初期化（必要な場合）
        if not self.no_motors:
            try:
                print("モーター制御の初期化...")
                self.motor_controller = LaneFollowingController(
                    base_speed=base_speed,
                    max_steering=max_steering,
                    steering_sensitivity=steering_sensitivity
                )
                self.motor_controller_initialized = True
                self.system_status['motor_control'] = 'OK'
            except Exception as e:
                print(f"モーター制御の初期化に失敗: {e}")
                self.system_status['motor_control'] = 'ERROR'
                self.system_status['last_error'] = str(e)
                
        # カラーマーカー検出器の初期化（必要な場合）
        if self.color_markers:
            try:
                print("カラーマーカー検出器の初期化...")
                # 外部カメラオブジェクトを渡す
                self.marker_detector = ColorMarkerDetector(external_camera=self.cap)
                self.marker_detector_initialized = True
                self.system_status['color_marker'] = 'OK'
            except Exception as e:
                print(f"カラーマーカー検出器の初期化に失敗: {e}")
                self.system_status['color_marker'] = 'ERROR'
                self.system_status['last_error'] = str(e)
    
    def init_video_writer(self):
        """動画ライターの初期化（改良版）"""
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
    
    def init_data_logging(self):
        """データログ機能の初期化"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'lane_following_log_{timestamp}.csv')
        
        # ログファイルを初期化
        with open(self.log_file, 'w') as f:
            header = "timestamp,frame,fps,center_offset,steering,action"
            
            if self.color_markers:
                header += ",red_detected,green_detected,blue_detected,yellow_detected"
                
            if not self.no_motors:
                header += ",left_speed,right_speed"
                
            f.write(header + "\n")
            
        print(f"データログ開始: {self.log_file}")
    
    def log_data(self, frame_data):
        """データのログ記録"""
        if not self.data_logging or self.log_file is None:
            return
            
        try:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                log_line = f"{timestamp},{self.frame_count},{self.current_fps:.1f}"
                
                # 基本データ
                log_line += f",{frame_data.get('center_offset', 0.0):.2f}"
                log_line += f",{frame_data.get('steering', 0.0):.2f}"
                log_line += f",{self.current_action}"
                
                # マーカー検出データ（有効な場合）
                if self.color_markers:
                    markers = frame_data.get('markers', {})
                    log_line += f",{1 if markers.get('red', {}).get('detected', False) else 0}"
                    log_line += f",{1 if markers.get('green', {}).get('detected', False) else 0}"
                    log_line += f",{1 if markers.get('blue', {}).get('detected', False) else 0}"
                    log_line += f",{1 if markers.get('yellow', {}).get('detected', False) else 0}"
                
                # モーター速度（有効な場合）
                if not self.no_motors and self.motor_controller_initialized:
                    log_line += f",{self.motor_controller.current_speeds[0]:.2f}"
                    log_line += f",{self.motor_controller.current_speeds[1]:.2f}"
                
                f.write(log_line + "\n")
                
        except Exception as e:
            print(f"データログ記録中にエラーが発生しました: {e}")
    
    def update_fps(self):
        """FPS計算の更新"""
        current_time = time.time()
        time_diff = current_time - self.last_fps_update_time
        
        # 1秒ごとに更新
        if time_diff >= 1.0:
            self.current_fps = (self.frame_count - self.last_frames) / time_diff
            self.last_frames = self.frame_count
            self.last_fps_update_time = current_time
    
    def process_frame(self, frame):
        """
        フレームを処理してレーン検出と操作を行う（改良版）
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        
        Returns:
        tuple: (processed_frame, frame_data)
        """
        frame_data = {}
        processed_frame = frame.copy()
        
        # レーン検出（有効な場合）
        center_offset = 0.0
        steering = 0.0
        
        if self.is_lane_detection_enabled() and self.lane_detector_initialized:
            try:
                # レーン検出
                lines, left_lane, right_lane, lane_image = self.lane_detector.detect_lane_lines(frame)
                
                # センターオフセットの計算
                center_offset = self.lane_detector.get_lane_center_offset(left_lane, right_lane)
                
                # レーン情報をフレームに描画
                processed_frame = self.lane_detector.draw_lane_overlay(
                    cv2.addWeighted(processed_frame, 0.8, lane_image, 1.0, 0),
                    left_lane, right_lane, center_offset
                )
                
                # レーン検出情報を保存
                frame_data['lines'] = lines
                frame_data['left_lane'] = left_lane
                frame_data['right_lane'] = right_lane
                frame_data['center_offset'] = center_offset
                
                # ステアリング値の計算
                if not self.no_motors and self.motor_controller_initialized:
                    steering = self.motor_controller.calculate_steering(center_offset)
                    frame_data['steering'] = steering
                
                # 状態を更新
                self.system_status['lane_detection'] = 'OK'
                
            except Exception as e:
                print(f"レーン検出中にエラーが発生しました: {e}")
                self.system_status['lane_detection'] = 'ERROR'
                self.system_status['last_error'] = str(e)
        
        # カラーマーカー検出（有効な場合）
        marker_action = 'none'
        if self.color_markers and self.marker_detector_initialized:
            try:
                # マーカー検出
                marker_results = self.marker_detector.detect_all_markers(frame)
                
                # マーカー描画
                processed_frame = self.marker_detector.draw_markers(processed_frame, marker_results)
                
                # マーカーに基づくアクション
                marker_action = self.marker_detector.get_marker_action()
                
                # マーカー検出情報を保存
                frame_data['markers'] = marker_results
                frame_data['marker_action'] = marker_action
                
                # 状態を更新
                self.system_status['color_marker'] = 'OK'
                
            except Exception as e:
                print(f"カラーマーカー検出中にエラーが発生しました: {e}")
                self.system_status['color_marker'] = 'ERROR'
                self.system_status['last_error'] = str(e)
        
        return processed_frame, frame_data
    
    def apply_marker_action(self, action):
        """
        マーカーアクションに基づいて操作を実行（改良版）
        
        Parameters:
        action (str): アクション ('stop', 'accelerate', 'lane_change', 'normal', 'none')
        """
        if action == 'none':
            return
            
        self.current_action = action
        
        if not self.no_motors and self.motor_controller_initialized:
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
                if hasattr(self.motor_controller, 'base_speed'):
                    self.motor_controller.set_motor_speeds(self.motor_controller.base_speed, 0.0)
                self.acceleration_timer = 0
                self.current_action = 'normal'
                print("アクション: 通常走行")
    
    def update_timers(self):
        """タイマーの更新処理（改良版）"""
        current_time = time.time()
        
        # 加速タイマー
        if self.acceleration_timer > 0 and current_time > self.acceleration_timer:
            if not self.no_motors and self.motor_controller_initialized:
                self.motor_controller.adjust_speed(1.0)  # 元の速度に戻す
            self.acceleration_timer = 0
            self.current_action = 'normal'
            print("加速終了: 通常速度に戻ります")
                
        # 車線変更タイマー
        if self.lane_change_timer > 0 and current_time > self.lane_change_timer:
            self.lane_change_timer = 0
            self.current_action = 'normal'
            print("車線変更終了: 通常制御に戻ります")
    
    def draw_debug_info(self, frame, frame_data, process_time):
        """
        デバッグ情報を描画（改良版）
        
        Parameters:
        frame (numpy.ndarray): 入力フレーム
        frame_data (dict): フレーム処理データ
        process_time (float): 処理時間（ms）
        
        Returns:
        numpy.ndarray: デバッグ情報が描画されたフレーム
        """
        result = frame.copy()
        
        # 基本情報を描画
        y_pos = 30
        line_height = 30
        
        # センターオフセット情報
        center_offset = frame_data.get('center_offset', 0.0)
        cv2.putText(result, f"センターオフセット: {center_offset:.2f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height
        
        # ステアリング情報
        steering = frame_data.get('steering', 0.0)
        cv2.putText(result, f"ステアリング: {steering:.2f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height
        
        # FPS・処理時間情報
        cv2.putText(result, f"FPS: {self.current_fps:.1f}, 処理時間: {process_time:.1f}ms", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += line_height
        
        # 現在のアクション表示
        action_color = (0, 255, 255)  # デフォルト色（黄色）
        
        if self.current_action == 'stop':
            action_color = (0, 0, 255)  # 赤
        elif self.current_action == 'accelerate':
            action_color = (0, 255, 0)  # 緑
        elif self.current_action == 'lane_change':
            action_color = (255, 0, 0)  # 青
            
        cv2.putText(result, f"アクション: {self.current_action}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        y_pos += line_height
        
        # システム状態の表示
        status_color = (0, 255, 0)  # デフォルト緑
        if any(s == 'ERROR' for s in self.system_status.values()):
            status_color = (0, 0, 255)  # エラー時は赤
            
        cv2.putText(result, f"システム状態: {'正常' if status_color == (0, 255, 0) else 'エラー'}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return result
    
    def signal_handler(self, sig, frame):
        """シグナルハンドラ（安全な終了のため）"""
        print("\nシステムを安全に終了します...")
        self.running = False
    
    def run(self):
        """メインループ（改良版）"""
        try:
            print("レーン追従システム開始")
            print("Ctrl+Cで終了")
            
            # モーター開始（モーター制御有効の場合）
            if not self.no_motors and self.motor_controller_initialized:
                self.motor_controller.start()
            
            while self.running:
                # フレームの取得
                ret, frame = self.cap.read()
                if not ret:
                    print("フレームの取得に失敗しました")
                    break
                
                start_time = time.time()
                
                # フレーム処理
                processed_frame, frame_data = self.process_frame(frame)
                
                # マーカーアクションの適用
                marker_action = frame_data.get('marker_action', 'none')
                self.apply_marker_action(marker_action)
                
                # タイマー更新
                self.update_timers()
                
                # モーター制御（有効かつ車線変更中でない場合）
                if (not self.no_motors and self.motor_controller_initialized and 
                    self.lane_change_timer == 0 and self.current_action != 'stop'):
                    center_offset = frame_data.get('center_offset', 0.0)
                    self.motor_controller.steer(center_offset)
                
                # 処理時間の計算
                process_time = (time.time() - start_time) * 1000
                
                # フレームカウンタと統計情報の更新
                self.frame_count += 1
                self.update_fps()
                
                # デバッグ情報の描画（有効な場合）
                if self.debug:
                    processed_frame = self.draw_debug_info(processed_frame, frame_data, process_time)
                
                # 結果の表示（ヘッドレスモードでない場合）
                if not self.headless:
                    cv2.imshow('Lane Following', processed_frame)
                
                # データログ記録（有効な場合）
                if self.data_logging:
                    self.log_data(frame_data)
                
                # 録画（有効な場合）
                if self.record and self.video_writer is not None:
                    self.video_writer.write(processed_frame)
                
                # キー入力の処理（ヘッドレスモードでない場合）
                key = cv2.waitKey(1) & 0xFF if not self.headless else 0xFF
                
                if key == ord('q'):
                    # 'q'キーで終了
                    self.running = False
                elif key == ord('s'):
                    # 's'キーでモーター停止/開始切り替え
                    if not self.no_motors and self.motor_controller_initialized:
                        if self.motor_controller.is_running:
                            self.motor_controller.stop_motors()
                            print("モーター停止")
                        else:
                            self.motor_controller.start()
                            print("モーター開始")
                elif key == ord('d'):
                    # 'd'キーでデバッグモード切り替え
                    self.debug = not self.debug
                    print(f"デバッグモード: {'有効' if self.debug else '無効'}")
                elif key == ord('r'):
                    # 'r'キーで録画開始/停止
                    if not self.record:
                        self.record = True
                        self.init_video_writer()
                        print("録画開始")
                    else:
                        self.record = False
                        if self.video_writer is not None:
                            self.video_writer.release()
                            self.video_writer = None
                        print("録画停止")
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """終了処理（改良版）"""
        # モーターの停止
        if not self.no_motors and self.motor_controller_initialized:
            try:
                self.motor_controller.cleanup()
            except Exception as e:
                print(f"モーター終了処理中にエラーが発生しました: {e}")
            
        # カメラの解放
        if self.camera_initialized:
            try:
                self.cap.release()
            except Exception as e:
                print(f"カメラ解放中にエラーが発生しました: {e}")
        
        # 録画の終了
        if self.record and self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception as e:
                print(f"録画終了処理中にエラーが発生しました: {e}")
            
        # ウィンドウの解放（ヘッドレスモードでない場合）
        if not self.headless:
            cv2.destroyAllWindows()
        
        print("\nレーン追従システムを終了しました")
        
        # 統計情報の表示
        total_time = time.time() - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0
        print(f"実行時間: {total_time:.1f}秒, 総フレーム数: {self.frame_count}, 平均FPS: {fps:.1f}")
        
        # エラーがあれば表示
        if self.system_status['last_error']:
            print(f"最後に発生したエラー: {self.system_status['last_error']}")

def main():
    parser = argparse.ArgumentParser(description='改良版 - レーン追従システム')
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
    parser.add_argument('--log-data', action='store_true',
                        help='データログ記録を有効化')
    parser.add_argument('--headless', action='store_true',
                        help='ヘッドレスモード（GUI表示なし）')
    
    args = parser.parse_args()
    
    try:
        follower = ImprovedIntegratedLaneFollower(
            camera_index=args.camera,
            no_motors=args.no_motors,
            base_speed=args.speed,
            max_steering=args.max_steering,
            steering_sensitivity=args.steering_sensitivity,
            color_markers=args.color_markers,
            record=args.record,
            debug=args.debug,
            data_logging=args.log_data,
            headless=args.headless
        )
        
        follower.run()
        
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    except Exception as e:
        print(f"プログラムの実行中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()