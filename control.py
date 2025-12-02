# control_keyboard_refactored.py

import gymnasium as gym
import pybullet as p
from stable_baselines3 import PPO
from two_wheel_env import TwoWheelRobotEnv 
import time

# --- CÁC THAM SỐ CẤU HÌNH ---
MODEL_PATH = "./hoanthanh/ppo_two_wheel_robot.zip"
URDF_PATH = "two_wheel_robot.urdf"
# Hằng số cho tốc độ xoay
TURNING_SPEED_CONSTANT = 4.5 

def control_with_keyboard():
    print("--- Bắt đầu điều khiển robot bằng bàn phím ---")
    print("Sử dụng các phím mũi tên:")
    print(" - Mũi tên LÊN:     Đi tiến")
    print(" - Mũi tên XUỐNG:   Đi lùi")
    print(" - Mũi tên TRÁI:    Rẽ trái")
    print(" - Mũi tên PHẢI:    Rẽ phải")
    print(" - (Không bấm gì):  Đứng im")
    print("\nNhấn phím 'ESC' trong cửa sổ mô phỏng để thoát.")

    env = TwoWheelRobotEnv(render_mode='human', urdf_path=URDF_PATH)

    try:    
        model = PPO.load(MODEL_PATH, env=env)
        print(f"[INFO] Đã tải thành công mô hình từ '{MODEL_PATH}'.")
    except Exception as e:
        print(f"[LỖI] Không thể tải mô hình '{MODEL_PATH}'.")
        print(f"       Chi tiết lỗi: {e}")
        env.close()
        return

    obs, info = env.reset()
    
    try:
        while True:
            # 1. Lấy sự kiện từ bàn phím
            keys = p.getKeyboardEvents()
            
            # Mặc định robot đứng yên và không xoay
            target_forward_velocity = 0.0
            turning_speed = 0.0
            
          
            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN: # Rẽ phải
                target_forward_velocity = 0
                turning_speed =TURNING_SPEED_CONSTANT
            elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN: # Rẽ trái
                target_forward_velocity = 0
                turning_speed = -TURNING_SPEED_CONSTANT

            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN: # Đi lùi
                target_forward_velocity = -1
                turning_speed = 0
            elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN: # Đi tiến
                target_forward_velocity = 1
                turning_speed = 0
            
            # Kiểm tra phím ESC để thoát
            if 65307 in keys and keys[65307] & p.KEY_IS_DOWN:
                print("\nPhím ESC được nhấn. Đang thoát...")
                break

            # 2. Cập nhật trạng thái cho môi trường và observation cho AI
            env.target_velocity = target_forward_velocity
            obs[3] = target_forward_velocity

            # 3. Cập nhật hiển thị
            p.addUserDebugText(
                f"Target Fwd: {target_forward_velocity:.2f} | Turn Spd: {turning_speed:.2f}", [0, -1, 1], 
                textColorRGB=[1, 0, 0], textSize=1.5, 
                replaceItemUniqueId=env.target_velocity_display_item,
                lifeTime=0.1 
            )

            # 4. AI ra quyết định 
            action, _states = model.predict(obs, deterministic=True)
            
            # 5. Thực hiện hành động trong môi trường với vận tốc xoay từ bàn phím
            obs, reward, terminated, truncated, info = env.step(action, turning_speed=turning_speed)
            
            # 6. Reset nếu robot ngã
            if terminated:
                print("Robot bị ngã! Đang reset...")
                obs, info = env.reset()
                time.sleep(1)

    finally:
        env.close()
        print("\n--- Chương trình kết thúc ---")

if __name__ == '__main__':
    control_with_keyboard()