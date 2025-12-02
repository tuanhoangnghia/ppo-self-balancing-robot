from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from two_wheel_env import TwoWheelRobotEnv 
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Tạo một lịch trình giảm learning rate tuyến tính.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


if __name__ == "__main__":

    print("--- Tạo môi trường huấn luyện đa tiến trình ---")
    num_cpu = 14 
    env = make_vec_env(lambda: TwoWheelRobotEnv(render_mode=None), n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    print(f"Không gian quan sát (Observation Space): {env.observation_space}")
    print(f"Không gian hành động (Action Space): {env.action_space}")

    print("\n--- Khởi tạo mô hình PPO trên CPU với siêu tham số được tinh chỉnh ---")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,     
        n_steps=4096,           
        batch_size=2048,      
        n_epochs=10,            
        gamma=0.99,            
        verbose=1,
        tensorboard_log="./ppo_robot_tensorboard_tuned/",
        device="cuda"
    )

    print("\n--- Bắt đầu huấn luyện tác nhân PPO ---")
    model.learn(total_timesteps=3000000, progress_bar=True)
    print("Huấn luyện hoàn tất.")

    model_save_path = "ppo_two_wheel_robot_tuned"
    model.save(model_save_path)
    print(f"\nMô hình đã được lưu tại: {model_save_path}")

    env.close()
    print("Môi trường đã đóng.")