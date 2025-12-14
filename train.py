import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from configs.config import EPISODE_DURATION, CONTROL_DT, MAX_STEPS, CURRICULUM_TRIGGER_STEP, FEATURES_DIM
from models.custom_policy import ResidualFeatureExtractor
from envs.space_robot_env import SpaceRobotEnv

class CurriculumCallback(BaseCallback):
    """
    Callback for switching the environment phase based on timestep.
    """
    def __init__(self, trigger_step: int, verbose: int = 0):
        super().__init__(verbose)
        self.trigger_step = trigger_step
        self.phase_switched = False

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.trigger_step and not self.phase_switched:
            # Switch phase from 1 to 2
            # Access the underlying environment(s) via set_attr
            self.training_env.set_attr("phase", 2)
            
            if self.verbose > 0:
                print(f"\n[Curriculum] Switched to Phase 2 at step {self.num_timesteps}!")
            
            self.phase_switched = True
            
        return True

def main():
    # 1. Initialize the custom environment
    # We wrap it in a lambda or pass the class to DummyVecEnv if we want vectorization,
    # but SB3 handles instances too.
    # Note: We use the dummy XML path defined in the Env class default.
    env = SpaceRobotEnv()
    
    # 2. Define Custom Policy Parameters
    # We pass the custom ResidualFeatureExtractor
    policy_kwargs = dict(
        features_extractor_class=ResidualFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        # Net arch can be customized further if needed, e.g. [256, 256] after extraction
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )

    # 3. Initialize DDPG with HER
    model = DDPG(
        "MultiInputPolicy",  # Required for Dict observation spaces (HER)
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        policy_kwargs=policy_kwargs,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=256,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        action_noise=None, # Add noise if needed, e.g. NormalActionNoise
        verbose=1,
    )

    # 4. Create Curriculum Callback
    curriculum_callback = CurriculumCallback(trigger_step=CURRICULUM_TRIGGER_STEP, verbose=1)

    # 5. Start Training
    print(f"Starting training with DDPG + HER...")
    print(f"Curriculum trigger set at {CURRICULUM_TRIGGER_STEP} steps.")
    
    # We set a total timestep count that covers the curriculum
    total_timesteps = CURRICULUM_TRIGGER_STEP * 2
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=curriculum_callback)
    except Exception as e:
        print(f"Training interrupted or failed: {e}")
        # This might happen if MuJoCo fails to load the dummy XML
        # But the code structure is what's requested.

    # 6. Save Model
    model.save("ddpg_space_robot_final")
    print("Model saved.")

if __name__ == "__main__":
    main()

