import os
import sys
import numpy as np
import mujoco

# Ensure stdout is unbuffered/utf-8
sys.stdout.reconfigure(encoding='utf-8')

def test_mujoco_load_and_control():
    print("--- MuJoCo Load and Control Test ---", flush=True)
    
    # Locate the assets directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    assets_dir = os.path.join(project_root, "assets")
    model_filename = "spacerobot_cjt.xml"
    
    # Change working directory to assets to avoid path encoding issues with C++ bindings
    # and to ensure relative mesh paths in XML work correctly
    original_cwd = os.getcwd()
    try:
        print(f"Changing working directory to: {assets_dir}")
        os.chdir(assets_dir)
        
        if not os.path.exists(model_filename):
            print(f"Error: Model file '{model_filename}' not found in {assets_dir}")
            return

        # Load the model
        print(f"Loading XML: {model_filename}")
        model = mujoco.MjModel.from_xml_path(model_filename)
        data = mujoco.MjData(model)
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    finally:
        # Restore CWD
        os.chdir(original_cwd)

    print(f"\nModel Statistics:")
    print(f"  Actuators: {model.nu}")
    print(f"  Joints:    {model.njnt}")
    print(f"  Qpos dim:  {model.nq}")
    print(f"  Qvel dim:  {model.nv}")

    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Apply control (torque)
    # Applying random small torques to available actuators
    action = np.array([10.0, -10.0, 5.0, -2.0, 1.0, -1.0])
    
    # Adjust action size to match model
    if len(action) > model.nu:
        action = action[:model.nu]
    elif len(action) < model.nu:
        action = np.pad(action, (0, model.nu - len(action)))
        
    print(f"\nApplying control inputs (torque): {action}")
    data.ctrl[:] = action

    # Step simulation
    steps = 100
    print(f"Stepping simulation for {steps} steps...")
    for _ in range(steps):
        mujoco.mj_step(model, data)

    # Show results
    print(f"\nFinal Joint Positions (qpos):")
    print(data.qpos)
    
    # Basic check
    if not np.allclose(data.qpos, 0):
        print("\nSUCCESS: Robot has moved from initial state.")
    else:
        print("\nWARNING: Robot has not moved.")

if __name__ == "__main__":
    test_mujoco_load_and_control()
