import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# Ensure stdout is unbuffered/utf-8
sys.stdout.reconfigure(encoding='utf-8')

class PDController:
    def __init__(self, kp, kd, null_kp=None, null_kd=None):
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        # Optional: null space control gains (not fully utilized in simple joint PD)
        self.null_kp = null_kp
        self.null_kd = null_kd

    def update(self, model, data, qpos, qvel, target_qpos, target_qvel=None):
        if target_qvel is None:
            target_qvel = np.zeros_like(qpos)
        
        # Computed Torque Control / Feedback Linearization
        # tau = M(q) * ( kp*(q_des - q) + kd*(q_vel_des - q_vel) ) + C(q, qdot) + G(q)
        # In MuJoCo: 
        #   mj_fullM(model, M, data.qM) gets the mass matrix (dense)
        #   data.qfrc_bias contains Coriolis + Gravity + centrifugal forces
        
        # Error terms
        q_error = target_qpos - qpos
        qvel_error = target_qvel - qvel
        
        # Desired acceleration (PD part)
        # u_pd = Kp * e + Kd * e_dot
        u_pd = self.kp * q_error + self.kd * qvel_error
        
        # Get Mass Matrix M(q)
        # Note: qM is stored in compressed format, need to convert to dense if using numpy matmul
        # Or better, use mj_mulM which computes res = M * vec efficiently
        
        # We need M * u_pd. 
        # Since u_pd is for the actuated joints (6 DoF), and the full Mass matrix includes base (6 DoF free joint),
        # we need to be careful with indices.
        # The robot joints are indices 6-11 (last 6).
        
        # Construct full vector for M multiplication (nv dim)
        # Assuming base is not actuated or handled separately (floating base)
        # For simple testing with fixed base or just focusing on arm:
        # We'll create a full vector of accelerations, with 0 for base and u_pd for arm.
        
        nv = model.nv
        acc_des_full = np.zeros(nv)
        acc_des_full[6:] = u_pd # Apply PD acceleration to arm joints
        
        # Compute M * acc_des_full
        # res = zeros(nv)
        # mujoco.mj_mulM(model, data, res, acc_des_full) -> This is the force required to produce acc_des_full
        
        M_u_pd = np.zeros(nv)
        mujoco.mj_mulM(model, data, M_u_pd, acc_des_full)
        
        # Get Bias forces (C + G)
        # data.qfrc_bias is size nv
        bias_forces = data.qfrc_bias
        
        # Total torque = M * u_pd + Bias
        # tau_full = M_u_pd + bias_forces
        
        # Extract torques for the actuated joints
        # Actuators correspond to the last 6 DoFs
        tau = M_u_pd[6:] + bias_forces[6:]
        
        # If we just want simple Gravity/Coriolis compensation + PD:
        # This implementation effectively does: tau = M(q) * (Kp*e + Kd*edot) + h(q,qdot)
        # This is strictly "Computed Torque Control" or "Inverse Dynamics Control" in joint space.
        
        return tau

def load_model():
    # Locate the assets directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    assets_dir = os.path.join(project_root, "assets")
    model_filename = "spacerobot_cjt.xml"
    
    # Change working directory to assets for resource loading
    original_cwd = os.getcwd()
    os.chdir(assets_dir)
    try:
        model = mujoco.MjModel.from_xml_path(model_filename)
        data = mujoco.MjData(model)
    finally:
        os.chdir(original_cwd)
        
    return model, data

import matplotlib.pyplot as plt

def evaluate_gains(model, data, kp, kd, target_type='static', duration=5.0, visualize=False, use_final_error=False, plot_results=False, wave_params=None):
    """
    Evaluates a set of PD gains for a specific target type.
    Returns a cost (lower is better), e.g., sum of squared errors.
    """
    mujoco.mj_resetData(model, data)
    
    # Default wave params if not provided
    if wave_params is None:
        wave_params = {'freq': 0.5, 'amp': 0.5}
    
    # Define actuator indices (assuming actuators 0-5 map to joints)
    # Note: The model has 13 qpos (7 for base freejoint + 6 joints)
    # The actuators control the 6 joints.
    # Joint qpos indices: 7, 8, 9, 10, 11, 12
    joint_indices = range(7, 13)
    actuator_indices = range(6)
    
    controller = PDController(kp, kd)
    
    dt = model.opt.timestep
    steps = int(duration / dt)
    
    total_error = 0.0
    
    # Data collection for plotting
    times = []
    actual_qpos = []
    target_qpos_list = []
    
    # Initial configuration (home)
    q0 = np.zeros(6)
    data.qpos[joint_indices] = q0
    
    if visualize:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for t in range(steps):
                current_time = t * dt
                
                # Step simulation and get target
                q_target = step_simulation_and_get_target(model, data, controller, joint_indices, actuator_indices, target_type, current_time, wave_params)
                
                viewer.sync()
                
                # Accumulate error
                q_curr = data.qpos[joint_indices]
                
                if plot_results:
                    times.append(current_time)
                    actual_qpos.append(q_curr.copy())
                    target_qpos_list.append(q_target.copy())

                if not use_final_error:
                    total_error += np.sum((q_curr - q_target)**2)
                elif t == steps - 1: # For final error, only take the last step
                    total_error = np.sum((q_curr - q_target)**2)
                    
                time.sleep(dt) # slow down for visualization
    else:
        for t in range(steps):
            current_time = t * dt
            
            # Step simulation and get target
            q_target = step_simulation_and_get_target(model, data, controller, joint_indices, actuator_indices, target_type, current_time, wave_params)
            
            # Accumulate error
            q_curr = data.qpos[joint_indices]

            if plot_results:
                times.append(current_time)
                actual_qpos.append(q_curr.copy())
                target_qpos_list.append(q_target.copy())
            
            if not use_final_error:
                total_error += np.sum((q_curr - q_target)**2)
            elif t == steps - 1: # For final error, only take the last step
                total_error = np.sum((q_curr - q_target)**2)

    if plot_results:
        plot_trajectory(times, actual_qpos, target_qpos_list, target_type, kp, kd, wave_params)
        
    return total_error

def step_simulation_and_get_target(model, data, controller, joint_indices, actuator_indices, target_type, current_time, wave_params=None):
    # Current state
    q_curr = data.qpos[joint_indices]
    qvel_curr = data.qvel[6:12] 
    
    if wave_params is None:
        wave_params = {'freq': 0.5, 'amp': 0.5}

    # Define target
    if target_type == 'static':
        q_target = np.ones(6) * 1.0 
        dq_target = np.zeros(6)
    elif target_type == 'dynamic':
        freq = wave_params['freq']
        amp = wave_params['amp']
        q_target = amp * np.sin(2 * np.pi * freq * current_time) * np.ones(6)
        dq_target = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * current_time) * np.ones(6)
        
    # Compute control
    tau = controller.update(model, data, q_curr, qvel_curr, q_target, dq_target)
    
    # Apply control
    data.ctrl[actuator_indices] = tau
    
    # Step physics
    mujoco.mj_step(model, data)
    
    return q_target

def plot_trajectory(times, actual, target, target_type, kp, kd, wave_params=None):
    actual = np.array(actual)
    target = np.array(target)
    
    num_joints = actual.shape[1]
    fig, axes = plt.subplots(num_joints, 1, figsize=(10, 15), sharex=True)
    
    if num_joints == 1:
        axes = [axes]
    
    # Check if kp/kd are scalar or vector/list
    if isinstance(kp, (list, np.ndarray)):
        title_kp = "Vector"
    else:
        title_kp = str(kp)
        
    title_extra = ""
    if wave_params:
        title_extra = f" (Freq={wave_params['freq']}, Amp={wave_params['amp']})"
        
    fig.suptitle(f'{target_type.capitalize()} Tracking (Kp={title_kp}){title_extra}', fontsize=16)
    
    for i in range(num_joints):
        ax = axes[i]
        ax.plot(times, target[:, i], 'r--', label='Desired' if i == 0 else "")
        ax.plot(times, actual[:, i], 'b-', label='Actual' if i == 0 else "")
        ax.set_ylabel(f'Joint {i+1} (rad)')
        ax.grid(True)
        if i == 0:
            ax.legend()
            
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Remove old step_simulation function as it is replaced
# def step_simulation(...) 


def tune_joint_wise_gains(model, data, initial_kp, initial_kd, target_type='static'):
    print(f"\n--- Joint-wise Tuning for {target_type.capitalize()} Target ---", flush=True)
    
    # Initialize gains
    current_kp = np.array([initial_kp] * 6, dtype=np.float64)
    current_kd = np.array([initial_kd] * 6, dtype=np.float64)
    
    # Tuning ranges - Finer granularity for Kp <= 100
    kp_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                     12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 
                     32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
                     52, 54, 56, 58, 60, 62, 64, 66, 68, 70,
                     72, 74, 76, 78, 80, 82, 84, 86, 88, 90,
                     92, 94, 96, 98, 100]
                     
    # Kd ratios - expanded to handle heavy joints which might need higher damping relative to Kp
    # Especially for low Kp, we might need Kd > Kp (ratio > 1.0) to achieve critical damping if Inertia is large.
    kd_ratios = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    
    # We will iterate through each joint and optimize its gains while holding others fixed.
    # We can do multiple passes.
    num_passes = 3
    
    # Define wave parameters to test
    wave_configs = [
        {'freq': 0.5, 'amp': 0.5, 'name': 'Fast/Large (Original)'},
        {'freq': 0.2, 'amp': 0.5, 'name': 'Slow/Large'},
        {'freq': 0.5, 'amp': 0.2, 'name': 'Fast/Small'},
        {'freq': 0.2, 'amp': 1.0, 'name': 'Slow/XLarge'}
    ]
    
    # Use the second config (Slow/Large) as the primary tuning target to ensure stability first?
    # Or just stick to one. The user asked to "try different sine functions".
    # Let's tune on one "Standard" one, but maybe slower to help heavy joints?
    # User said "1,2,3 low gain... try different sine function".
    # Let's switch default tuning to a slower frequency (0.25 Hz) to see if they can track that better.
    tune_config = {'freq': 0.25, 'amp': 0.8}
    
    # Evaluate initial cost
    best_cost = evaluate_gains(model, data, current_kp, current_kd, target_type, duration=4.0, 
                             use_final_error=(target_type=='static'), wave_params=tune_config)
    print(f"Initial Cost: {best_cost:.4f}")
    
    for p in range(num_passes):
        print(f"Pass {p+1}/{num_passes}")
        for joint_idx in range(6):
            print(f"  Tuning Joint {joint_idx}...", end='', flush=True)
            
            local_best_cost = best_cost
            best_local_kp = current_kp[joint_idx]
            best_local_kd = current_kd[joint_idx]
            
            for kp_val in kp_candidates:
                for ratio in kd_ratios:
                    kd_val = kp_val * ratio
                    
                    # Constraint: Kd should not exceed 5.0
                    if kd_val > 5.0:
                        continue
                    
                    # Temporarily update gains
                    test_kp = current_kp.copy()
                    test_kd = current_kd.copy()
                    test_kp[joint_idx] = kp_val
                    test_kd[joint_idx] = kd_val
                    
                    cost = evaluate_gains(model, data, test_kp, test_kd, target_type, duration=4.0, 
                                        use_final_error=(target_type=='static'), wave_params=tune_config)
                    
                    if cost < local_best_cost:
                        local_best_cost = cost
                        best_local_kp = kp_val
                        best_local_kd = kd_val
            
            # Update best for this joint
            if local_best_cost < best_cost:
                current_kp[joint_idx] = best_local_kp
                current_kd[joint_idx] = best_local_kd
                best_cost = local_best_cost
                print(f" Improved! Kp={best_local_kp}, Kd={best_local_kd:.4f} -> Cost: {best_cost:.2f}")
            else:
                print(f" No improvement.")
                
    print(f"Final Tuned Gains for {target_type}:")
    print(f"Kp: {current_kp}")
    print(f"Kd: {current_kd}")
    
    return current_kp, current_kd, tune_config

def tune_pd_gains():
    print("--- PID Gain Tuning (Dynamic Only) ---", flush=True)
    model, data = load_model()
    
    # 1. Joint-wise Tuning for Dynamic directly
    # Starting with a reasonable scalar guess
    initial_kp = 10
    initial_kd = 0.5
    
    best_dynamic_kp, best_dynamic_kd, tune_config = tune_joint_wise_gains(model, data, initial_kp, initial_kd, target_type='dynamic')
    
    print("\nVisualizing Best Dynamic Result...", flush=True)
    evaluate_gains(model, data, best_dynamic_kp, best_dynamic_kd, target_type='dynamic', duration=10.0, visualize=False, plot_results=True, wave_params=tune_config)

    return None, (best_dynamic_kp, best_dynamic_kd)

# Wrapper to allow vector init in tune_joint_wise_gains
def tune_joint_wise_gains_vector_init(model, data, initial_kp_vec, initial_kd_vec, target_type='static'):
    print(f"\n--- Joint-wise Tuning for {target_type.capitalize()} Target ---", flush=True)
    
    current_kp = np.array(initial_kp_vec, dtype=np.float64)
    current_kd = np.array(initial_kd_vec, dtype=np.float64)
    
    kp_candidates = [5, 10, 20, 30, 40, 50]
    kd_ratios = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    num_passes = 1 # Single pass is enough if starting from good point
    
    best_cost = evaluate_gains(model, data, current_kp, current_kd, target_type, duration=4.0, use_final_error=(target_type=='static'))
    print(f"Initial Cost: {best_cost:.4f}")
    
    for p in range(num_passes):
        for joint_idx in range(6):
            print(f"  Tuning Joint {joint_idx}...", end='', flush=True)
            local_best_cost = best_cost
            best_local_kp = current_kp[joint_idx]
            best_local_kd = current_kd[joint_idx]
            
            for kp_val in kp_candidates:
                for ratio in kd_ratios:
                    kd_val = kp_val * ratio
                    
                    test_kp = current_kp.copy()
                    test_kd = current_kd.copy()
                    test_kp[joint_idx] = kp_val
                    test_kd[joint_idx] = kd_val
                    
                    cost = evaluate_gains(model, data, test_kp, test_kd, target_type, duration=4.0, use_final_error=(target_type=='static'))
                    
                    if cost < local_best_cost:
                        local_best_cost = cost
                        best_local_kp = kp_val
                        best_local_kd = kd_val
            
            if local_best_cost < best_cost:
                current_kp[joint_idx] = best_local_kp
                current_kd[joint_idx] = best_local_kd
                best_cost = local_best_cost
                print(f" Improved! Kp={best_local_kp}, Kd={best_local_kd:.2f} -> Cost: {best_cost:.2f}")
            else:
                print(f" No improvement.")
                
    return current_kp, current_kd

if __name__ == "__main__":
    tune_pd_gains()

