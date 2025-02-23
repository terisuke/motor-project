# dual_motor_control.py
from gpiozero import Motor
from time import sleep

class DualMotorController:
    def __init__(self):
        # モーター1のピン設定（forward_pin, backward_pin）
        self.motor1 = Motor(14, 15)
        
        # モーター2のピン設定（backward_pin, forward_pin）- ピンの順序を入れ替え
        self.motor2 = Motor(24, 23)  # 23, 24から24, 23に変更

    def set_motor1(self, speed):
        """
        モーター1の速度を設定（-1から1の値）
        正の値: 正転、負の値: 逆転
        """
        if speed > 0:
            self.motor1.backward(speed)
        elif speed < 0:
            self.motor1.forward(-speed)
        else:
            self.motor1.stop()

    def set_motor2(self, speed):
        """
        モーター2の速度を設定（-1から1の値）
        正の値: 正転、負の値: 逆転
        """
        if speed > 0:
            self.motor2.backward(speed)
        elif speed < 0:
            self.motor2.forward(-speed)
        else:
            self.motor2.stop()

    def stop_all(self):
        """両方のモーターを停止"""
        self.motor1.stop()
        self.motor2.stop()

def main():
    controller = DualMotorController()
    
    try:
        print("テスト開始")
        
        # 前進（50%スピード）
        print("前進")
        controller.set_motor1(0.5)
        controller.set_motor2(0.5)
        sleep(2)
        
        # 停止
        print("停止")
        controller.stop_all()
        sleep(1)
        
        # 後進（50%スピード）
        print("後進")
        controller.set_motor1(-0.5)
        controller.set_motor2(-0.5)
        sleep(2)
        
        # 停止
        print("停止")
        controller.stop_all()
        sleep(1)
        
        # 素早い前進
        print("素早い前進")
        controller.set_motor1(1)
        controller.set_motor2(1)
        sleep(0.5)
        
        # 停止
        print("停止")
        controller.stop_all()
        sleep(0.5)
        
        # 素早い後進
        print("素早い後進")
        controller.set_motor1(-1)
        controller.set_motor2(-1)
        sleep(0.5)
        
        # 最終停止
        print("テスト終了")
        controller.stop_all()
        
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    
    finally:
        controller.stop_all()

if __name__ == "__main__":
    main()