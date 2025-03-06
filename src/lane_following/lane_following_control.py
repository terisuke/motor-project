import time
import numpy as np
from gpiozero import Motor

class LaneFollowingController:
    def __init__(self, base_speed=0.5, max_steering=0.5, steering_sensitivity=1.0):
        """
        レーン追従制御クラスの初期化
        
        Parameters:
        base_speed (float): 基本速度 (0.0-1.0)
        max_steering (float): 最大ステアリング量 (0.0-1.0)
        steering_sensitivity (float): ステアリング感度
        """
        # 制御パラメータ
        self.base_speed = base_speed
        self.max_steering = max_steering
        self.steering_sensitivity = steering_sensitivity
        
        # モーターの初期化
        self.motor1 = Motor(14, 15)  # 左モーター
        self.motor2 = Motor(24, 23)  # 右モーター
        
        # 動作状態の初期化
        self.is_running = False
        self.emergency_stop = False
        self.current_speed = 0.0
        self.current_steering = 0.0
        
        print("レーン追従制御の初期化完了")
        print(f"基本速度: {self.base_speed}, 最大ステアリング: {self.max_steering}, ステアリング感度: {self.steering_sensitivity}")
    
    def calculate_steering(self, center_offset):
        """
        センターオフセットからステアリング値を計算
        
        Parameters:
        center_offset (float): センターオフセット (-1.0 to 1.0)
        
        Returns:
        float: ステアリング値 (-max_steering to max_steering)
        """
        # センターオフセットにステアリング感度を適用
        steering = center_offset * self.steering_sensitivity
        
        # 最大ステアリング量に制限
        steering = max(-self.max_steering, min(self.max_steering, steering))
        
        return steering
    
    def set_motor_speeds(self, speed, steering):
        """
        モーターの速度を設定
        
        Parameters:
        speed (float): 基本速度 (0.0-1.0)
        steering (float): ステアリング値 (-max_steering to max_steering)
        """
        if self.emergency_stop:
            self.stop_motors()
            return
        
        # ステアリングに基づいて左右のモーター速度を調整
        left_speed = speed
        right_speed = speed
        
        if steering > 0:  # 右に曲がる（左モーターを速く）
            right_speed -= steering
        elif steering < 0:  # 左に曲がる（右モーターを速く）
            left_speed += steering
        
        # 速度の範囲を制限
        left_speed = max(0.0, min(1.0, left_speed))
        right_speed = max(0.0, min(1.0, right_speed))
        
        # 現在の速度とステアリングを保存
        self.current_speed = speed
        self.current_steering = steering
        
        # モーターの速度を設定
        if not self.emergency_stop:
            self.motor1.forward(left_speed)
            self.motor2.forward(right_speed)
            self.is_running = speed > 0
    
    def steer(self, center_offset):
        """
        センターオフセットに基づいて操縦
        
        Parameters:
        center_offset (float): センターオフセット (-1.0 to 1.0)
        """
        # ステアリング値を計算
        steering = self.calculate_steering(center_offset)
        
        # モーター速度を設定
        self.set_motor_speeds(self.base_speed, steering)
        
        return steering
    
    def adjust_speed(self, speed_factor):
        """
        速度を調整
        
        Parameters:
        speed_factor (float): 速度調整係数 (0.0-2.0)
        """
        new_speed = self.base_speed * speed_factor
        new_speed = max(0.0, min(1.0, new_speed))
        
        # ステアリングはそのままで速度のみ調整
        self.set_motor_speeds(new_speed, self.current_steering)
    
    def stop_motors(self):
        """モーターを停止"""
        self.motor1.stop()
        self.motor2.stop()
        self.is_running = False
    
    def start(self):
        """モーターを開始"""
        if not self.emergency_stop:
            self.set_motor_speeds(self.base_speed, 0.0)
    
    def emergency_stop_toggle(self, stop=True):
        """
        緊急停止を設定/解除
        
        Parameters:
        stop (bool): Trueで緊急停止、Falseで解除
        """
        self.emergency_stop = stop
        if stop:
            self.stop_motors()
        else:
            # 緊急停止解除後は前の速度に戻る
            self.set_motor_speeds(self.current_speed, self.current_steering)
    
    def turn_left(self, duration=1.0, intensity=1.0):
        """
        左に曲がる
        
        Parameters:
        duration (float): 曲がる時間（秒）
        intensity (float): 曲がる強さ（0.0-1.0）
        """
        if self.emergency_stop:
            return
            
        actual_intensity = self.max_steering * intensity
        self.set_motor_speeds(self.base_speed, -actual_intensity)
        time.sleep(duration)
        self.set_motor_speeds(self.base_speed, self.current_steering)
    
    def turn_right(self, duration=1.0, intensity=1.0):
        """
        右に曲がる
        
        Parameters:
        duration (float): 曲がる時間（秒）
        intensity (float): 曲がる強さ（0.0-1.0）
        """
        if self.emergency_stop:
            return
            
        actual_intensity = self.max_steering * intensity
        self.set_motor_speeds(self.base_speed, actual_intensity)
        time.sleep(duration)
        self.set_motor_speeds(self.base_speed, self.current_steering)
    
    def cleanup(self):
        """終了処理"""
        self.stop_motors()
        print("モーター制御を終了しました")

def test_lane_following():
    """レーン追従制御のテスト関数"""
    controller = LaneFollowingController(base_speed=0.5, max_steering=0.5)
    
    try:
        print("テスト開始")
        
        # 開始
        print("開始")
        controller.start()
        time.sleep(2)
        
        # 左に曲がる
        print("左に曲がる")
        controller.steer(-0.5)  # センターから左に0.5オフセット
        time.sleep(2)
        
        # 右に曲がる
        print("右に曲がる")
        controller.steer(0.5)  # センターから右に0.5オフセット
        time.sleep(2)
        
        # 速度調整
        print("速度を上げる")
        controller.adjust_speed(1.5)  # 速度を1.5倍に
        time.sleep(2)
        
        print("速度を下げる")
        controller.adjust_speed(0.5)  # 速度を0.5倍に
        time.sleep(2)
        
        # 緊急停止
        print("緊急停止")
        controller.emergency_stop_toggle(True)
        time.sleep(2)
        
        # 緊急停止解除
        print("緊急停止解除")
        controller.emergency_stop_toggle(False)
        time.sleep(2)
        
        # 停止
        print("テスト終了")
        controller.stop_motors()
        
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    
    finally:
        controller.cleanup()

if __name__ == "__main__":
    test_lane_following()