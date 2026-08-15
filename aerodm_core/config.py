"""Configuration for AeroDM training and evaluation."""

class Config:
    # Training parameters
    num_epochs = 100
    batch_size = 32
    _base_num_trajectories = 20000 # Base number of trajectories before augmentation (will be increased by concatenation)
    num_test_samples = 50

    # Model dimensions
    latent_dim = 128
    obs_latent_dim = 128
    num_layers = 4
    num_heads = 4
    dropout = 0.1

    # Diffusion parameters
    diffusion_steps = 60 # Number of diffusion steps for training (e.g., 50-100)
    inference_diffusion_steps = 60  # Can be set lower for faster inference (e.g., 20-50)
    beta_start = 0.0001
    beta_end = 0.02

    # Sequence parameters
    seq_len = 60  # N_a = 60 time steps; 6s-long future trajectory sequence
    state_dim = 10  # x_i ∈ R^10: p(3) + v(3) + r(4) + style(1) = 11, but we will use 10 by excluding style from the state vector and handling it as a condition
    history_len = 20  # 20-frame historical observations

    # Condition dimensions
    target_dim = 4  # p_t ∈ R^3 + valid flag (1)
    action_dim = 14   # 14 maneuver styles

    # Obstacle parameters
    max_obstacles = 10  # Maximum number of obstacles to process
    obstacle_feat_dim = 4  # [x, y, z, radius]
    enable_obstacle_encoding = False  # Toggle obstacle encoding in the model
    use_obstacle_loss = enable_obstacle_encoding  # Toggle obstacle loss term in training

    # CBF Guidance parameters (from CoDiG paper)
    # enable_cbf_guidance = True  # Disabled by default; toggle for inference
    guidance_gamma = 130.0  # Base gamma for barrier guidance
    safe_extra_factor=0.20 # Safety buffer as fraction of radius (e.g., 20%)
    barrier_sigma = 0.5 # Smoothness parameter for logistic barrier function (tune for best results: )
    cbf_decay_rate = 0.3  # Discrete-time CBF gamma in (0, 1]
    guidance_scale = 2.0  # Classifier-free guidance scale for sampling (tune for best results)

    last_xyz_weight=50.0 # Extra weight for final timestep's position error
    xyz_weight=1.0 # Extra weight for Z-axis (height) in aviation
    vel_weight=1.0 # Weight for velocity term
    other_weight=1.0 # Weight for other losses (Speed loss, Attitude loss)
    obstacle_weight=1.0 # Weight for obstacle term
    continuity_weight=50.0 # Weight for continuity term (MSE between last history and first pred timestep)
    acc_weight=10.0 # Weight for acceleration term
    dynamics_consistency_weight=0.0 # Weight for p[k+1] = p[k] + delta_T * v[k]
    delta_T = 0.1 # Time step duration (0.1s for 10Hz control frequency)

    drop_style_prob = 0.1  # Probability of dropping style information during training for robustness
    drop_target_prob = 0.1  # Probability of dropping target waypoint information during training for robustness

    # Plotting control
    show_flag = False  # Set to False to save plots as SVG instead of displaying
    show_collision_points = True  # Set to False to hide collision-point markers

    # Training control flag
    train_model = False  # Set to True to train, False to load existing model

    # Model save/load paths
    model_save_dir = "model"
    model_filename = "aerodm_v2_test_20260730_005041.pth"
