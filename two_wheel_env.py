# two_wheel_env_refactored.py

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time

class TwoWheelRobotEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None, urdf_path="two_wheel_robot.urdf"):
        super(TwoWheelRobotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.urdf_path = urdf_path

        if self.render_mode == 'human':
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self.target_velocity_display_item = p.addUserDebugText(
                "Target Fwd: 0.0 | Turn Spd: 0.0", [0, -1, 1], textColorRGB=[1, 0, 0], textSize=1.5
            )
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # Action space chỉ điều khiển vận tốc cân bằng tiến/lùi
        max_wheel_velocity = 25.0
        self.action_space = gym.spaces.Box(
            low=-max_wheel_velocity, high=max_wheel_velocity, shape=(1,), dtype=np.float32
        )

        # Observation space không thay đổi
        max_target_velocity = 5.0
        obs_low = np.array([-np.pi / 2, -15.0, -20.0, -max_target_velocity])
        obs_high = np.array([np.pi / 2, 15.0, 20.0, max_target_velocity])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0.35], p.getQuaternionFromEuler([0, 0, 0]))
        
        self.joint_names = {p.getJointInfo(self.robot_id, i)[1].decode("utf-8"): i 
                            for i in range(p.getNumJoints(self.robot_id))}
        self.left_wheel_joint_idx = self.joint_names['left_leg_to_wheel']
        self.right_wheel_joint_idx = self.joint_names['right_leg_to_wheel']
        
        self.fall_angle_threshold = 40 * (np.pi / 180.0)
        self.max_force = 100.0
        self.max_target_velocity = max_target_velocity
        self.target_velocity = 0.0
        # ### <<< THAY ĐỔI: Đã loại bỏ self.turn_mode >>>

    def _get_observation(self):
        _, orientation_q = p.getBasePositionAndOrientation(self.robot_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        
        orientation_euler = p.getEulerFromQuaternion(orientation_q)
        pitch = orientation_euler[1]
        
        rot_matrix = p.getMatrixFromQuaternion(orientation_q)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        forward_vector = rot_matrix[:, 0] 
        forward_velocity = np.dot(linear_vel, forward_vector)

        pitch_velocity = angular_vel[1]
        
        return np.array([pitch, forward_velocity, pitch_velocity, self.target_velocity], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'target_velocity' in options:
            self.target_velocity = options['target_velocity']
        else:
            if np.random.rand() < 0.5:
                self.target_velocity = 0.0
            else:
                self.target_velocity = np.random.uniform(-self.max_target_velocity, self.max_target_velocity)

        # ### <<< THAY ĐỔI: Đã loại bỏ logic chọn turn_mode ngẫu nhiên >>>
        
        random_pitch = np.random.uniform(-0.1, 0.1)
        reset_pos = [0, 0, 0.35]
        reset_orientation = p.getQuaternionFromEuler([0, random_pitch, 0])
        
        p.resetBasePositionAndOrientation(self.robot_id, reset_pos, reset_orientation)
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])

        observation = self._get_observation()
        return observation, {}

    # ### <<< THAY ĐỔI LỚN: Hàm step nhận thêm turning_speed >>>
    def step(self, action, turning_speed=0.0):
        # Action từ model PPO là vận tốc để giữ thăng bằng và đi thẳng
        balancing_and_forward_action = action[0]
        
        # Vận tốc cuối cùng của mỗi bánh là sự kết hợp của hành động từ AI và lệnh rẽ từ người dùng
        # turning_speed > 0: Rẽ phải
        # turning_speed < 0: Rẽ trái
        left_target_vel = balancing_and_forward_action + turning_speed
        right_target_vel = balancing_and_forward_action - turning_speed
        
        p.setJointMotorControl2(self.robot_id, self.left_wheel_joint_idx, p.VELOCITY_CONTROL, 
                                targetVelocity=left_target_vel, force=self.max_force)
        p.setJointMotorControl2(self.robot_id, self.right_wheel_joint_idx, p.VELOCITY_CONTROL, 
                                targetVelocity=right_target_vel, force=self.max_force)

        p.stepSimulation()
        if self.render_mode == 'human':
            time.sleep(self.time_step)

        obs = self._get_observation()
        pitch, forward_velocity, pitch_velocity, _ = obs

        # Phần thưởng không thay đổi, vì mục tiêu của AI vẫn là giữ thăng bằng và đạt vận tốc mục tiêu
        REWARD_SURVIVAL = 0.5
        REWARD_VELOCITY_SCALE = 10.0
        PENALTY_PITCH = 2.0
        PENALTY_PITCH_VELOCITY = 0.2
        PENALTY_ACTION = 0.01
        
        velocity_error = abs(self.target_velocity - forward_velocity)
        velocity_reward = REWARD_VELOCITY_SCALE * np.exp(-3.0 * velocity_error)
        
        reward = REWARD_SURVIVAL + velocity_reward
        reward -= PENALTY_PITCH * abs(pitch)
        reward -= PENALTY_PITCH_VELOCITY * abs(pitch_velocity)
        reward -= PENALTY_ACTION * np.square(action).sum()
        
        terminated = abs(pitch) > self.fall_angle_threshold
        
        if terminated:
            reward = -100.0

        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.client >= 0:
            p.disconnect(self.client)
            self.client = -1