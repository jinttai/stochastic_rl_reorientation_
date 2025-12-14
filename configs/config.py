
# Simulation Parameters
EPISODE_DURATION = 10.0  # seconds
CONTROL_DT = 0.02        # 50 Hz
MAX_STEPS = int(EPISODE_DURATION / CONTROL_DT)  # 500 steps

# Training Curriculum
CURRICULUM_TRIGGER_STEP = 500_000

# Network Architecture
# Dimension of the latent feature space produced by the custom extractor
FEATURES_DIM = 256
# Hidden layer size for the residual blocks
HIDDEN_SIZE = 256

