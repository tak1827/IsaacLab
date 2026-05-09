import math

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets import G1_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

from .agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from .flat_env_cfg import G1FlatEnvCfg
from .rough_env_cfg import G1Rewards


@configclass
class G1FlatEnvGaitCfg(G1FlatEnvCfg):
    """Placeholder config for gait-specific flat G1 training."""

    rewards: "G1FlatEnvGaitRewardsCfg" = None
    events: "G1FlatEnvGaitEventCfg" = None
    curriculum_phase: int = 1

    def __post_init__(self):
        if self.rewards is None:
            self.rewards = G1FlatEnvGaitRewardsCfg()
        if self.events is None:
            self.events = G1FlatEnvGaitEventCfg()
        super().__post_init__()

        # Change from Minimal(`G1_MINIMAL_CFG`) to Full (`G1_CFG`) G1 robot.
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Friction randomization (highest priority): override default fixed ranges from base config.
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.9)

        # Slight variation in initial joint positions for robustness to init error.
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)

        self.phase_command_curriculum = {
            "1": {
                "resampling_time_range": (10.0, 10.0),
                "rel_standing_envs": 0.02,
                "lin_vel_x": (0.0, 1.0),
                "lin_vel_y": (-0.5, 0.5),
                "ang_vel_z": (-1.0, 1.0),
                "heading": (-math.pi, math.pi),
            },
            "2": {
                "resampling_time_range": (5.0, 10.0),
                "rel_standing_envs": 0.3,
                "lin_vel_x": (0.0, 1.0),
                "lin_vel_y": (-0.5, 0.5),
                "ang_vel_z": (-1.0, 1.0),
                "heading": (-math.pi, math.pi),
            },
            "3": {
                "resampling_time_range": (5.0, 10.0),
                "rel_standing_envs": 0.1,
                "lin_vel_x": (0.0, 2.5),
                "lin_vel_y": (-0.5, 0.5),
                "ang_vel_z": (-1.0, 1.0),
                "heading": (-math.pi, math.pi),
            },
        }

        # Rewards
        self.rewards.joint_deviation_arms.weight = 0.0


@configclass
class G1FlatEnvGaitCfg_PLAY(G1FlatEnvGaitCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# Recover None events from base config.
@configclass
class G1FlatEnvGaitEventCfg(EventCfg):
    """Gait-specific event configuration."""
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        }
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
        }
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(12.0, 18.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


# =========================================================
# 🧠 G1-style rewards
#   Reference: https://arxiv.org/pdf/2505.20619
# =========================================================
@configclass
class G1FlatEnvGaitRewardsCfg(G1Rewards):
    """Gait-oriented rewards for flat G1 training."""
    s_stance_knee_extension = RewTerm(
        func=mdp.stance_knee_extension,
        weight=0.5,
        params={
            "knee_cfg": SceneEntityCfg("robot", joint_names=[".*left_knee_joint", ".*right_knee_joint"]),
            "foot_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]
            ),
            "target": 0.12,
            "scale": 5.0,
            "target_gait_id": 0,
        },
    )
    s_joint_deviation_l1 = RewTerm(
        func=mdp.gait_joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            ),
            "target_gait_id": 0,
        },
    )

    # --------------- Target Gait ID: 1 (Walking) ---------------
    w_stance_knee_extension = RewTerm(
        func=mdp.stance_knee_extension,
        weight=0.2,
        params={
            "knee_cfg": SceneEntityCfg("robot", joint_names=[".*left_knee_joint", ".*right_knee_joint"]),
            "foot_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]
            ),
            "target": 0.18,
            "scale": 8.0,
            "target_gait_id": 1,
        },
    )
    w_joint_deviation_l1 = RewTerm(
        func=mdp.gait_joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            ),
            "target_gait_id": 1,
        },
    )

    # --------------- Target Gait ID: 2 (Walk to Stand) ---------------
    w2s_stance_knee_extension = RewTerm(
        func=mdp.stance_knee_extension,
        weight=0.2,
        params={
            "knee_cfg": SceneEntityCfg("robot", joint_names=[".*left_knee_joint", ".*right_knee_joint"]),
            "foot_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*left_ankle_roll_link", ".*right_ankle_roll_link"]
            ),
            "target": 0.14,
            "scale": 5.0,
            "target_gait_id": 2,
        },
    )
    w2s_joint_deviation_l1 = RewTerm(
        func=mdp.gait_joint_deviation_l1,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            ),
            "target_gait_id": 2,
        },
    )

    # --------------- Target Gait ID: 3 (Running) ---------------
    r_joint_deviation_l1 = RewTerm(
        func=mdp.gait_joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            ),
            "target_gait_id": 3,
        },
    )

    # --------------- Target Gait ID: 4 (Run to Walk) ---------------
    r2w_joint_deviation_l1 = RewTerm(
        func=mdp.gait_joint_deviation_l1,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            ),
            "target_gait_id": 4,
        },
    )

@configclass
class G1FlatPPORunnerGaitCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_flat_gait"
        self.num_steps_per_env = 48 # Increased. 24 steps (0.5s) are too short for LSTM to learn transitions.
        self.save_interval = 200
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            rnn_type="lstm",
            rnn_hidden_dim=512, # Size of the LSTM hidden state.
            rnn_num_layers=1,   # Number of stacked LSTM layers.
        )
