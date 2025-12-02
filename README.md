# Two Wheel Robot Balancing Project

Dự án này mô phỏng và huấn luyện một robot hai bánh tự cân bằng sử dụng Học Tăng Cường (Reinforcement Learning) với thuật toán PPO (Proximal Policy Optimization). Môi trường mô phỏng được xây dựng trên PyBullet và Gymnasium.

## Cấu trúc dự án

- **`two_wheel_env.py`**: Định nghĩa môi trường Gym tùy chỉnh cho robot. Bao gồm logic vật lý, không gian hành động/quan sát và hàm tính thưởng.
- **`train.py`**: Script để huấn luyện mô hình PPO. Sử dụng `stable_baselines3` để huấn luyện đa tiến trình.
- **`control.py`**: Script để chạy kiểm thử và điều khiển robot bằng bàn phím sử dụng mô hình đã huấn luyện.
- **`two_wheel_robot.urdf`**: File mô tả mô hình robot (URDF).
- **`requirements.txt`**: Danh sách các thư viện cần thiết.

## Cài đặt

1.  Clone repository này về máy.
2.  Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

**Lưu ý:** Dự án yêu cầu Python 3.x và các thư viện như `gymnasium`, `pybullet`, `stable_baselines3`, `torch`.

## Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (Training)

Để bắt đầu huấn luyện robot từ đầu, chạy lệnh:

```bash
python train.py
```

- Quá trình huấn luyện sẽ sử dụng 14 CPU (mặc định trong code) để chạy song song.
- Logs sẽ được lưu vào thư mục `ppo_robot_tensorboard_tuned/`.
- Mô hình sau khi huấn luyện sẽ được lưu thành `ppo_two_wheel_robot_tuned.zip`.

### 2. Điều khiển & Kiểm thử (Inference/Control)

Để chạy mô phỏng với mô hình đã huấn luyện và điều khiển bằng bàn phím:

```bash
python control.py
```

**Điều khiển:**
- **Mũi tên LÊN**: Đi tiến
- **Mũi tên XUỐNG**: Đi lùi
- **Mũi tên TRÁI**: Rẽ trái
- **Mũi tên PHẢI**: Rẽ phải
- **ESC**: Thoát chương trình

**Lưu ý quan trọng:**
- Trong file `control.py`, đường dẫn mô hình đang được trỏ tới `./hoanthanh/ppo_two_wheel_robot8.zip`. Hãy đảm bảo bạn có file model tại đường dẫn này hoặc cập nhật biến `MODEL_PATH` trong `control.py` để trỏ tới file model bạn muốn sử dụng (ví dụ: `ppo_two_wheel_robot_tuned.zip` sau khi train xong).

## Chi tiết kỹ thuật

- **Thuật toán**: PPO (Proximal Policy Optimization).
- **Observation Space (4 chiều)**:
    - Góc nghiêng (Pitch)
    - Vận tốc tiến (Forward Velocity)
    - Vận tốc góc nghiêng (Pitch Velocity)
    - Vận tốc mục tiêu (Target Velocity)
- **Action Space**: Điều khiển vận tốc motor để giữ thăng bằng.
- **Reward Function**: Khuyến khích robot giữ thăng bằng (survival), bám sát vận tốc mục tiêu, và phạt khi robot bị ngã hoặc dao động quá mạnh.
