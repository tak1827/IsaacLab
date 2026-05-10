import math

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets import G1_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

from .agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from .flat_env_cfg import G1FlatEnvCfg
from .rough_env_cfg import G1Rewards
from .unitree import UNITREE_G1_29DOF_CFG


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
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
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
                    ".*_elbow_joint",
                    ".*_wrist_.*",
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
                    ".*_elbow_joint",
                    ".*_wrist_.*",
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
                    ".*_elbow_joint",
                    ".*_wrist_.*",
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
                    ".*_elbow_joint",
                    ".*_wrist_.*",
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
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            ),
            "target_gait_id": 4,
        },
    )


@configclass
class G1FlatEnvGaitObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations (no base linear velocity)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1FlatPPORunnerGaitCfg(G1FlatPPORunnerCfg):
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_flat_gait"
        self.save_interval = 200
        # self.num_steps_per_env = 48 # Increased. 24 steps (0.5s) are too short for LSTM to learn transitions.
        # self.policy = RslRlPpoActorCriticRecurrentCfg(
        #     init_noise_std=1.0,
        #     actor_obs_normalization=False,
        #     critic_obs_normalization=False,
        #     actor_hidden_dims=[512, 256, 128],
        #     critic_hidden_dims=[512, 256, 128],
        #     activation="elu",
        #     rnn_type="lstm",
        #     rnn_hidden_dim=512, # Size of the LSTM hidden state.
        #     rnn_num_layers=1,   # Number of stacked LSTM layers.
        # )


@configclass
class G1FlatEnvGaitCfg(G1FlatEnvCfg):
    """Placeholder config for gait-specific flat G1 training."""

    observations: G1FlatEnvGaitObservationsCfg = G1FlatEnvGaitObservationsCfg()
    rewards: G1FlatEnvGaitRewardsCfg = G1FlatEnvGaitRewardsCfg()
    events: G1FlatEnvGaitEventCfg = G1FlatEnvGaitEventCfg()
    curriculum_phase: int = 1

    def __post_init__(self):
        super().__post_init__()

        # Change from Minimal(`G1_MINIMAL_CFG`) to Full (`G1_CFG`) G1 robot.aaa
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        # Disable reward terms that reference joints unavailable in UNITREE_G1_29DOF_CFG.
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None


@configclass
class G1FlatEnvGaitCfg_PLAY(G1FlatEnvGaitCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        # disable randomization for play
        # self.observations.policy.enable_corruption = False
        # self.observations.critic.enable_corruption = False
        # # remove random pushing
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
