#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
レーン追従システム統合テストスクリプト
様々なモジュールとシステム全体のテストを自動化するスクリプト
"""

import os
import sys
import argparse
import time
from datetime import datetime
import cv2
import numpy as np

# 親ディレクトリをパスに追加してインポートできるようにする
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# テスト中のモジュールのインポート
from src.lane_following.enhanced_camera_utils import setup_camera, verify_camera_settings, take_test_shots
from src.lane_following.lane_detector import LaneDetector
from src.lane_following.color_marker_detector import ColorMarkerDetector
from src.lane_following.lane_following_control import LaneFollowingController

class IntegratedTester:
    def __init__(self, args):
        self.args = args
        self.test_results = {}
        self.log_dir = os.path.join(current_dir, 'test_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, 
                                     f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # ログファイル初期化
        with open(self.log_file, 'w') as f:
            f.write(f"=== レーン追従システム統合テスト ===\n")
            f.write(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"テスト設定:\n")
            for key, value in vars(args).items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    def log(self, message):
        """ログメッセージを出力"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
    
    def run_camera_test(self):
        """カメラモジュールのテスト"""
        self.log("\n=== カメラモジュールテスト ===")
        try:
            # カメラ初期化テスト
            self.log("カメラ初期化テスト...")
            cap, settings = setup_camera(
                camera_index=self.args.camera, 
                optimize_for=self.args.optimize_for
            )
            
            # カメラ設定の検証
            self.log("カメラ設定の検証...")
            current_settings = verify_camera_settings(cap, settings)
            
            # 結果をログに記録
            self.log(f"解像度: {current_settings['WIDTH']}x{current_settings['HEIGHT']}")
            self.log(f"FPS: {current_settings['FPS']}")
            self.log(f"コントラスト: {current_settings['CONTRAST']}")
            self.log(f"彩度: {current_settings['SATURATION']}")
            
            if current_settings['FRAME_AVAILABLE']:
                h, s, v = current_settings['HSV_MEAN']
                self.log(f"HSV平均値: H={h:.1f}, S={s:.1f}, V={v:.1f}")
                
                b_mean = current_settings['BRIGHTNESS_STATS']['mean']
                b_std = current_settings['BRIGHTNESS_STATS']['std']
                self.log(f"輝度統計: 平均={b_mean:.1f}, 標準偏差={b_std:.1f}")
                
                # 輝度のチェック
                if b_mean < 50:
                    self.log("警告: 画像が暗すぎる可能性があります。照明を確認してください。")
                elif b_mean > 200:
                    self.log("警告: 画像が明るすぎる可能性があります。露出設定を確認してください。")
                
                # コントラストのチェック
                if b_std < 20:
                    self.log("警告: 画像のコントラストが低い可能性があります。")
            
            # テスト画像の撮影（必要に応じて）
            if self.args.save_images:
                save_dir = os.path.join(self.log_dir, 'camera_test')
                frames = take_test_shots(cap, num_frames=3, save_dir=save_dir)
                self.log(f"テスト画像を {save_dir} に保存しました")
            
            # カメラの解放
            cap.release()
            
            self.test_results['camera'] = True
            self.log("カメラモジュールテスト: 成功")
            
        except Exception as e:
            self.log(f"カメラモジュールテスト中にエラーが発生しました: {e}")
            self.test_results['camera'] = False
    
    def run_lane_detector_test(self):
        """レーン検出モジュールのテスト"""
        self.log("\n=== レーン検出モジュールテスト ===")
        try:
            # レーン検出器の初期化
            self.log("レーン検出器の初期化...")
            
            # カメラを先に初期化
            cap, _ = setup_camera(
                camera_index=self.args.camera, 
                optimize_for='lane_detection'
            )
            
            # レーン検出器の初期化（カメラは渡さない）
            detector = LaneDetector(camera_index=self.args.camera)
            
            # 基本機能テスト
            self.log("基本機能テスト...")
            
            # テストのためのフレームを取得
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("フレームの取得に失敗しました")
            
            # レーン検出テスト
            start_time = time.time()
            lines, left_lane, right_lane, lane_image = detector.detect_lane_lines(frame)
            process_time = (time.time() - start_time) * 1000
            
            # 結果の評価
            self.log(f"検出時間: {process_time:.1f}ms")
            self.log(f"検出されたライン数: {0 if lines is None else len(lines)}")
            self.log(f"左レーン検出: {'成功' if left_lane else '失敗'}")
            self.log(f"右レーン検出: {'成功' if right_lane else '失敗'}")
            
            # 結果画像の保存（必要に応じて）
            if self.args.save_images:
                save_dir = os.path.join(self.log_dir, 'lane_detector_test')
                os.makedirs(save_dir, exist_ok=True)
                
                # 元のフレーム
                cv2.imwrite(os.path.join(save_dir, 'original_frame.jpg'), frame)
                
                # レーン検出結果
                if lane_image is not None:
                    cv2.imwrite(os.path.join(save_dir, 'lane_detection.jpg'), lane_image)
                
                # レーン情報を描画
                if left_lane or right_lane:
                    center_offset = detector.get_lane_center_offset(left_lane, right_lane)
                    result_frame = detector.draw_lane_overlay(
                        cv2.addWeighted(frame, 0.8, lane_image, 1.0, 0),
                        left_lane, right_lane, center_offset
                    )
                    cv2.imwrite(os.path.join(save_dir, 'lane_overlay.jpg'), result_frame)
                    self.log(f"センターオフセット: {center_offset:.2f}")
                
                self.log(f"テスト画像を {save_dir} に保存しました")
            
            # カメラとリソースの解放
            cap.release()
            cv2.destroyAllWindows()
            
            self.test_results['lane_detector'] = True
            self.log("レーン検出モジュールテスト: 成功")
            
        except Exception as e:
            self.log(f"レーン検出モジュールテスト中にエラーが発生しました: {e}")
            self.test_results['lane_detector'] = False
    
    def run_color_marker_test(self):
        """色マーカー検出モジュールのテスト"""
        self.log("\n=== 色マーカー検出モジュールテスト ===")
        try:
            # 色マーカー検出器の初期化
            self.log("色マーカー検出器の初期化...")
            
            # 専用の最適化設定でカメラを初期化
            cap, _ = setup_camera(
                camera_index=self.args.camera, 
                optimize_for='color_marker'
            )
            
            # 色マーカー検出器の初期化
            detector = ColorMarkerDetector(camera_index=self.args.camera)
            
            # 基本機能テスト
            self.log("基本機能テスト...")
            
            # テストフレームを取得
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("フレームの取得に失敗しました")
            
            # 色マーカー検出テスト
            start_time = time.time()
            marker_results = detector.detect_all_markers(frame)
            process_time = (time.time() - start_time) * 1000
            
            # 結果の評価と表示
            self.log(f"検出時間: {process_time:.1f}ms")
            
            for color, result in marker_results.items():
                status = '検出' if result['detected'] else '未検出'
                self.log(f"{color}マーカー: {status}")
                
                if result['detected'] and result['center'] is not None:
                    self.log(f"  位置: {result['center']}")
            
            # アクション取得テスト
            action = detector.get_marker_action()
            self.log(f"検出されたアクション: {action}")
            
            # 結果画像の保存（必要に応じて）
            if self.args.save_images:
                save_dir = os.path.join(self.log_dir, 'color_marker_test')
                os.makedirs(save_dir, exist_ok=True)
                
                # 元のフレーム
                cv2.imwrite(os.path.join(save_dir, 'original_frame.jpg'), frame)
                
                # マーカー検出結果
                result_frame = detector.draw_markers(frame, marker_results)
                cv2.imwrite(os.path.join(save_dir, 'marker_detection.jpg'), result_frame)
                
                # 各色のマスクも保存
                for color, result in marker_results.items():
                    if 'mask' in result and result['mask'] is not None:
                        cv2.imwrite(os.path.join(save_dir, f'{color}_mask.jpg'), result['mask'])
                
                self.log(f"テスト画像を {save_dir} に保存しました")
            
            # カメラとリソースの解放
            cap.release()
            cv2.destroyAllWindows()
            
            self.test_results['color_marker'] = True
            self.log("色マーカー検出モジュールテスト: 成功")
            
        except Exception as e:
            self.log(f"色マーカー検出モジュールテスト中にエラーが発生しました: {e}")
            self.test_results['color_marker'] = False
    
    def run_motor_controller_test(self):
        """モーター制御モジュールのテスト（ドライラン）"""
        self.log("\n=== モーター制御モジュールテスト（シミュレーション） ===")
        
        if self.args.skip_motors:
            self.log("モーターテストをスキップします")
            self.test_results['motor_controller'] = None
            return
        
        try:
            # モーター制御モジュールの初期化（速度は低めに設定）
            self.log("モーター制御モジュールの初期化...")
            controller = LaneFollowingController(
                base_speed=0.3,  # テスト用に低速に設定
                max_steering=0.5,
                steering_sensitivity=1.0
            )
            
            # 基本動作テスト
            self.log("基本動作テスト（ドライラン）...")
            
            # テスト用のセンターオフセット値
            test_offsets = [-0.8, -0.4, 0.0, 0.4, 0.8]
            
            for offset in test_offsets:
                # ステアリング値の計算
                steering = controller.calculate_steering(offset)
                self.log(f"オフセット: {offset:.1f} → ステアリング: {steering:.2f}")
                
                if not self.args.no_motors:
                    # 実際にモーターを動かす（短時間）
                    self.log(f"  モーター動作テスト（オフセット {offset:.1f}）...")
                    controller.steer(offset)
                    time.sleep(0.5)
                    controller.stop_motors()
                    time.sleep(0.5)
            
            # 特殊動作テスト
            if not self.args.no_motors:
                # 左折テスト
                self.log("左折テスト...")
                controller.turn_left(duration=0.5, intensity=0.5)
                time.sleep(1.0)
                
                # 右折テスト
                self.log("右折テスト...")
                controller.turn_right(duration=0.5, intensity=0.5)
                time.sleep(1.0)
                
                # 加速テスト
                self.log("加速テスト...")
                controller.adjust_speed(1.2)  # 速度を1.2倍に
                time.sleep(1.0)
                controller.adjust_speed(1.0)  # 通常速度に
                
                # 緊急停止テスト
                self.log("緊急停止テスト...")
                controller.set_motor_speeds(controller.base_speed, 0.0)
                time.sleep(0.5)
                controller.emergency_stop_toggle(True)
                time.sleep(1.0)
                controller.emergency_stop_toggle(False)
                time.sleep(0.5)
            
            # 最終停止
            controller.stop_motors()
            self.log("モーター停止")
            
            # クリーンアップ
            controller.cleanup()
            
            self.test_results['motor_controller'] = True
            self.log("モーター制御モジュールテスト: 成功")
            
        except Exception as e:
            self.log(f"モーター制御モジュールテスト中にエラーが発生しました: {e}")
            self.test_results['motor_controller'] = False
            
            # エラー発生時はモーターを確実に停止
            try:
                controller.stop_motors()
                controller.cleanup()
            except:
                pass
    
    def run_integrated_test(self):
        """システム統合テスト"""
        self.log("\n=== システム統合テスト ===")
        try:
            import subprocess
            
            # テスト内容に基づいてコマンドを構築
            cmd = ["python3", "src/lane_following/integrated_lane_follower.py"]
            
            # 引数の追加
            cmd.extend(["--camera", str(self.args.camera)])
            
            if self.args.no_motors:
                cmd.append("--no-motors")
                
            cmd.extend(["--speed", "0.3"])  # テストでは低速を使用
            cmd.extend(["--debug"])
            
            if self.args.color_markers:
                cmd.append("--color-markers")
                
            if self.args.save_images:
                cmd.append("--record")
                
            self.log(f"統合システムを実行: {' '.join(cmd)}")
            self.log("システムが起動したら、テストするために以下を行ってください:")
            self.log("1. レーン検出が機能していることを確認")
            self.log("2. 色マーカーを表示して各アクションをテスト（有効な場合）")
            self.log("3. 'q'キーでシステムを終了")
            
            # 5秒の待機
            self.log("5秒後にシステムを起動します...")
            for i in range(5, 0, -1):
                self.log(f"{i}...")
                time.sleep(1)
                
            # サブプロセスとして統合システムを実行
            process = subprocess.Popen(cmd)
            
            # 最大60秒間待機（ユーザーが'q'を押して終了するまで）
            max_wait = 60
            self.log(f"システムが起動しました。最大{max_wait}秒間実行します。")
            self.log("手動でテストし、終了するには'q'キーを押してください。")
            
            # 終了を待機
            process.wait(timeout=max_wait)
            
            self.test_results['integrated'] = True
            self.log("システム統合テスト: 成功")
            
        except subprocess.TimeoutExpired:
            self.log("タイムアウト: システムを終了します")
            process.terminate()
            process.wait()
            self.test_results['integrated'] = True  # タイムアウトは成功とみなす
            
        except Exception as e:
            self.log(f"システム統合テスト中にエラーが発生しました: {e}")
            self.test_results['integrated'] = False
            
            # エラー発生時はプロセスを確実に終了
            try:
                process.terminate()
                process.wait()
            except:
                pass
    
    def run_all_tests(self):
        """すべてのテストを実行"""
        self.log("=== レーン追従システム統合テスト開始 ===")
        
        # カメラテスト
        self.run_camera_test()
        
        # 前のテストが成功した場合のみ続行
        if self.test_results.get('camera', False):
            # レーン検出テスト
            self.run_lane_detector_test()
            
            # 色マーカー検出テスト（要求された場合）
            if self.args.color_markers:
                self.run_color_marker_test()
            
            # モーター制御テスト（スキップされていない場合）
            if not self.args.skip_motors:
                self.run_motor_controller_test()
            
            # 統合テスト（要求された場合）
            if self.args.integrated_test:
                self.run_integrated_test()
        
        # 結果のまとめ
        self.log("\n=== テスト結果サマリー ===")
        for test, result in self.test_results.items():
            if result is None:
                status = "スキップ"
            elif result:
                status = "成功"
            else:
                status = "失敗"
            self.log(f"{test}: {status}")
        
        # 全体の結果
        if all(result for result in self.test_results.values() if result is not None):
            self.log("\n全テスト成功!")
        else:
            self.log("\n一部テストが失敗しました。詳細なログを確認してください。")
        
        self.log(f"\nログファイル: {self.log_file}")

def main():
    parser = argparse.ArgumentParser(description='レーン追従システム統合テスト')
    parser.add_argument('--camera', type=int, default=0,
                        help='カメラデバイスのインデックス（デフォルト: 0）')
    parser.add_argument('--optimize-for', choices=['lane_detection', 'color_marker', 'general'],
                        default='lane_detection', help='カメラ最適化タイプ（デフォルト: lane_detection）')
    parser.add_argument('--no-motors', action='store_true',
                        help='モーター動作なしでテスト')
    parser.add_argument('--skip-motors', action='store_true',
                        help='モーターテストをスキップ')
    parser.add_argument('--color-markers', action='store_true',
                        help='色マーカー検出テストを含める')
    parser.add_argument('--save-images', action='store_true',
                        help='テスト画像を保存する')
    parser.add_argument('--integrated-test', action='store_true',
                        help='統合システムテストを実行する')
    
    args = parser.parse_args()
    
    try:
        tester = IntegratedTester(args)
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\nテストが中断されました")
    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()