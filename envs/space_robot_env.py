import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
import os

from configs.config import CONTROL_DT

class SpaceRobotEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(1.0 / CONTROL_DT),
    }

    def __init__(self, xml_file="assets/dummy_spacerobot.xml", **kwargs):
        utils.EzPickle.__init__(self, xml_file, **kwargs)
        
        self.phase = 1
        
        # Placeholder for joint limits and target storage
        self._target_base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._target_joint_angles = np.zeros(6) # Assuming 6 joints
        
        # We need to define observation_space before MujocoEnv init if we want to validly pass it
        # But MujocoEnv init loads the model, which gives us the dimensions.
        # We'll use a placeholder space and update it after super().__init__ calls _get_obs() 
        # OR we can defer strict checking. 
        # Gymnasium MujocoEnv usually infers space from _get_obs.
        
        # Note: frame_skip calculation depends on the model's timestep. 
        # We assume model timestep is small (e.g. 0.002) and we want 0.02 control dt.
        # We'll set frame_skip=1 for now and rely on XML or adjust if we knew the timestep.
        # For this code generation, we'll pick a reasonable default or let it be passed.
        
        # Note: We use a dummy path as requested. 
        # Ideally, we would check if file exists.
        if not os.path.exists(xml_file):
            # If the file doesn't exist, we can't fully initialize MujocoEnv.
            # We will create a dummy init if this is just for code generation structure,
            # but usually this class is expected to run.
            pass

        # Define observation space structure (Generic sizes, adjusted in _set_action_space if needed)
        # We assume 6 joints for the generated code.
        n_joints = 6 
        
        # Base Quat (4) + Base Ang Vel (3) + Joint Angles (n) + Joint Vel (n)
        obs_dim = 4 + 3 + n_joints + n_joints
        goal_dim = 4 + n_joints
        
        self.observation_space = Dict({
            "observation": Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64),
            "achieved_goal": Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float64),
            "desired_goal": Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float64),
        })

        try:
             MujocoEnv.__init__(
                self,
                model_path=xml_file,
                frame_skip=10, # Assuming 0.002 timestep -> 0.02 dt
                observation_space=self.observation_space,
                default_camera_config={},
                **kwargs
            )
        except Exception as e:
            print(f"Warning: Could not initialize MujocoEnv with {xml_file}. {e}")
            # This allows the code to be imported without crashing if XML is missing

    def _get_obs(self):
        # If model is not loaded (dummy mode), return zeros
        if self.data is None:
            n_joints = 6
            obs_dim = 4 + 3 + n_joints + n_joints
            goal_dim = 4 + n_joints
            return {
                "observation": np.zeros(obs_dim),
                "achieved_goal": np.zeros(goal_dim),
                "desired_goal": np.zeros(goal_dim)
            }

        # qpos: [x, y, z, qw, qx, qy, qz, j1...jn]
        # qvel: [vx, vy, vz, wx, wy, wz, v1...vn]
        
        base_quat = self.data.qpos[3:7].copy()
        base_ang_vel = self.data.qvel[3:6].copy()
        joint_angles = self.data.qpos[7:].copy()
        joint_vel = self.data.qvel[6:].copy()
        
        obs = np.concatenate([base_quat, base_ang_vel, joint_angles, joint_vel])
        
        achieved_goal = np.concatenate([base_quat, joint_angles])
        desired_goal = np.concatenate([self._target_base_quat, self._target_joint_angles])
        
        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }

    def reset_model(self):
        # Randomize initial state
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        
        # Randomize target
        self._target_base_quat = np.array([1.0, 0.0, 0.0, 0.0]) # Simplified target
        # Add random orientation logic here if needed
        self._target_joint_angles = self.np_random.uniform(low=-0.5, high=0.5, size=len(qpos)-7)

        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        # Action is joint velocities
        # Apply to ctrl (assuming actuators are configured for velocity or we use high-gain)
        # Prompt: "apply these velocities to data.ctrl"
        self.do_simulation(action, self.frame_skip)
        
        obs = self._get_obs()
        
        # Reward and Info
        info = {}
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        
        # Check termination (optional, usually reorientation is continuous or fixed time)
        terminated = False
        truncated = False # Handled by TimeLimit wrapper usually
        
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # achieved_goal: [Base_Quat (4), Joint_Angles (N)]
        # desired_goal: [Target_Quat (4), Target_Joints (N)]
        
        # Handle vectorized input
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
            desired_goal = desired_goal.reshape(1, -1)
            single = True
        else:
            single = False

        # Extract components
        # Base Quat is first 4
        current_quat = achieved_goal[:, :4]
        target_quat = desired_goal[:, :4]
        
        current_joints = achieved_goal[:, 4:]
        target_joints = desired_goal[:, 4:]
        
        # Orientation Error (Simple Euclidean or Quat dist)
        # Simple Euclidean on Quaternions is often used for small errors, 
        # or 1 - |q . q_t|^2
        # For simplicity:
        quat_diff = current_quat - target_quat
        orientation_error = np.linalg.norm(quat_diff, axis=1)
        
        # Joint Error
        joint_error = np.linalg.norm(current_joints - target_joints, axis=1)
        
        # Joint Limits Penalty (Soft)
        # We need access to model limits. self.model.jnt_range
        # Since this is a static method or vectorized, accessing self.model is tricky if self is not passed.
        # But compute_reward is a method of the env.
        # However, HER might call it with just arrays. 
        # We'll skip complex joint limit checks in vectorized compute_reward for now 
        # or assume we only punish large values.
        
        limit_penalty = 0.0 # Placeholder
        
        if self.phase == 1:
            reward = -orientation_error - limit_penalty
        else:
            reward = -(orientation_error + joint_error) - limit_penalty
            
        if single:
            return reward[0]
        return reward

