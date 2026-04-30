# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

# Phase-aligned contact pattern
def gait_phase_contact(
    env,
    left_foot: str,
    right_foot: str,
    cycle_time: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Reward phase-aligned left/right contacts with smooth cyclic targets."""
    if cycle_time <= 0.0:
        raise ValueError(f"`cycle_time` must be positive, got: {cycle_time}")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    left_foot_id = contact_sensor.find_bodies([left_foot])[0][0]
    right_foot_id = contact_sensor.find_bodies([right_foot])[0][0]

    contact_force_hist = contact_sensor.data.net_forces_w_history
    # Use the latest contact-force sample and a sigmoid gate to reduce contact jitter.
    left_force = torch.norm(contact_force_hist[:, -1, left_foot_id, :], dim=-1)
    right_force = torch.norm(contact_force_hist[:, -1, right_foot_id, :], dim=-1)
    left_contact = torch.sigmoid((left_force - 1.0) * 5.0)
    right_contact = torch.sigmoid((right_force - 1.0) * 5.0)

    speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    adaptive_cycle_time = torch.clamp(cycle_time * 0.6 + 0.4 / (speed + 1.0e-3), 0.5, 1.2)
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt, adaptive_cycle_time) / adaptive_cycle_time
    phase_angle = 2.0 * math.pi * phase

    left_target = 0.5 * (1.0 + torch.sin(phase_angle))
    right_target = 0.5 * (1.0 + torch.sin(phase_angle + math.pi))

    left_reward = 1.0 - torch.square(left_contact - left_target)
    right_reward = 1.0 - torch.square(right_contact - right_target)

    support = left_contact + right_contact
    support_reward = torch.exp(-5.0 * torch.square(support - 1.0))

    reward = 0.7 * 0.5 * (left_reward + right_reward) + 0.3 * support_reward
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return reward * (speed > 0.1) * gait_mask


# Straight knee during stance
def stance_knee_extension(
    env,
    knee_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    target: float = 0.1,
    scale: float = 5.0,
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Simple, robust reward for straight knees during stance."""
    asset = env.scene[knee_cfg.name]
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- knee positions (N, K)
    knee_pos = asset.data.joint_pos[:, knee_cfg.joint_ids]

    # --- contact forces (use norm, not just z)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)
    # --- soft stance (simple and smooth)
    stance = torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # --- squared error
    knee_error = torch.square(knee_pos - target)
    if knee_error.shape[1] != stance.shape[1]:
        raise ValueError(
            f"Mismatch between knees ({knee_error.shape[1]}) and feet ({stance.shape[1]}). "
            "Set `knee_cfg.joint_names` and `foot_sensor_cfg.body_names` to matching left/right pairs."
        )

    # --- apply stance weighting
    weighted_error = knee_error * stance
    # --- normalize so flight phase doesn't give free reward
    stance_sum = torch.sum(stance, dim=1) + 1.0e-6
    avg_error = torch.sum(weighted_error, dim=1) / stance_sum
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return torch.exp(-scale * avg_error) * gait_mask


# Contact Pattern
# Standing: Encourage consistent double-foot support for static balance.
# Walking: Encourage alternating single-leg contact and flight phases.
# Running: Encourage flight phases.
def contact_pattern_reward(
    env,
    foot_sensor_cfg: SceneEntityCfg,
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Unified contact-pattern reward selected by gait id.

    Pattern by `target_gait_id`:
    - 0 (stand): prefer double support
    - 1 (walk): prefer single support
    - 3 (run): prefer flight
    """
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- (N, 2, 3) -> (N, 2)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)

    # --- soft contact (0~1)
    contact = torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # --- continuous number of contacts (0~2)
    num_contacts = torch.sum(contact, dim=1)

    # --- smooth reward
    single_support = torch.exp(-5.0 * (num_contacts - 1.0) ** 2)
    flight = torch.exp(-5.0 * (num_contacts - 0.0) ** 2)
    double_support = torch.exp(-5.0 * (num_contacts - 2.0) ** 2)

    gait_mask = (env.current_gait_id == target_gait_id).float()

    if target_gait_id == 0:
        # Stand: strong double-support, discourage shuffle/flight.
        reward = double_support - 0.5 * single_support - 1.0 * flight
    elif target_gait_id == 3:
        # Run: prioritize flight, weakly discourage support phases.
        reward = flight - 0.5 * single_support - 1.0 * double_support
    else:
        # Walk (and fallback): prioritize alternating single support.
        reward = single_support + 0.5 * flight - 1.0 * double_support

    return reward * gait_mask

# Base Stability:
# Penalize base and joint motion to maintain upright posture.
def base_stability_standing(
    env,
    std_base: float = 0.25,
    std_joint: float = 0.1,
    target_gait_id: int = 0,
) -> torch.Tensor:
    """Penalize base/joint motion and tilt while in standing gait mode."""
    asset = env.scene["robot"]

    # ---- base motion ----
    lin_vel = asset.data.root_lin_vel_w
    ang_vel = asset.data.root_ang_vel_w
    base_error = (
        torch.sum(lin_vel**2, dim=1) +
        0.5 * torch.sum(ang_vel**2, dim=1)   # reduce angular dominance
    )

    # ---- joint motion ----
    joint_vel = asset.data.joint_vel
    joint_error = torch.mean(joint_vel**2, dim=1)  # mean is more stable than sum

    # ---- upright ----
    # only x,y tilt matters
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    upright_error = torch.sum(gravity_xy**2, dim=1)

    # ---- combine (still simple) ----
    reward = torch.exp(-base_error / std_base) \
           * torch.exp(-joint_error / std_joint) \
           * torch.exp(-upright_error / 0.1)

    gait_mask = (env.current_gait_id == target_gait_id).float()
    return reward * gait_mask


# Push-Off Dynamics
# Reward strong vertical and forward velocity during push-off.
def push_off_velocity_reward(
    env,
    asset_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    command_name: str,
    scale: float = 1.0,
    target_gait_id: int = 3,
) -> torch.Tensor:
    """Reward velocity along commanded direction during push-off."""
    asset = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- commanded velocity (N, 2)
    v_cmd = env.command_manager.get_command(command_name)[:, :2]

    # --- actual velocity in world frame (N, 2)
    v = asset.data.root_lin_vel_w[:, :2]

    # --- projection (how well we move in commanded direction)
    proj_vel = torch.sum(v * v_cmd, dim=1)

    # --- contact (N, 2)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)
    contacts = torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # --- push-off phase (partial contact)
    push_off = contacts * (1.0 - contacts)
    push_off_strength = torch.sum(push_off, dim=1)
    gait_mask = (env.current_gait_id == target_gait_id).float()

    return proj_vel * push_off_strength * scale * gait_mask


# Short Contact
# Penalize prolonged stance to promote dynamic running
def short_contact_reward(
    env,
    foot_sensor_cfg: SceneEntityCfg,
    max_steps: int = 15,
    scale: float = 1.0,
    target_gait_id: int = 3,
) -> torch.Tensor:
    """Penalize long foot contact duration to encourage dynamic running."""
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- (N, F, 3) -> (N, F)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)

    # --- soft contact (0~1)
    contact = torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # contact duration buffer (per-env, per-foot) (N, F)
    duration = getattr(env, "_contact_duration", None)
    if duration is None or duration.shape != contact.shape:
        duration = torch.zeros_like(contact)

    # Reset durations for envs marked for reset in this step.
    # `reset_buf` is computed before reward terms, so this catches episode boundaries reliably.
    if hasattr(env, "reset_buf"):
        duration = torch.where(env.reset_buf.unsqueeze(1), torch.zeros_like(duration), duration)

    # update: accumulate only when in contact
    duration = (duration + 1.0) * contact
    env._contact_duration = duration

    # penalty only when duration exceeds threshold
    excess = torch.relu(duration - max_steps)
    penalty = torch.sum(excess, dim=1)
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return torch.exp(-scale * penalty) * gait_mask


# Feet Swing Height Penalty
# Ensure the robot lifts its feet sufficiently during the swing phase to prevent tripping and dragging
def feet_swing_height_penalty(
    env,
    foot_sensor_cfg: SceneEntityCfg,
    target_height: float = 0.1,
    scale: float = 20.0,
    target_gait_id: int = 1,
) -> torch.Tensor:
    """Encourage sufficient foot clearance during swing."""
    asset = env.scene["robot"]
    sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]

    # --- contact force (N, F)
    forces = sensor.data.net_forces_w[:, foot_sensor_cfg.body_ids, :]
    foot_force = torch.norm(forces, dim=-1)

    # --- smooth swing (1 = swing, 0 = stance)
    swing = 1.0 - torch.clamp(foot_force / (foot_force + 10.0), 0.0, 1.0)

    # --- foot height (world)
    # Contact sensors provide forces but not body poses. Resolve matching bodies on the robot asset.
    foot_body_ids, _ = asset.find_bodies(foot_sensor_cfg.body_names, preserve_order=foot_sensor_cfg.preserve_order)
    foot_pos = asset.data.body_pos_w[:, foot_body_ids, :]
    foot_height = foot_pos[..., 2]

    # --- terrain-relative foot height
    # `TerrainImporter` does not expose `get_heights` in this setup.
    # For flat-plane tasks, ground is z=0; keep a guarded path for terrains that implement height queries.
    foot_xy = foot_pos[..., :2].reshape(-1, 2)
    get_heights_fn = getattr(env.scene.terrain, "get_heights", None)
    if callable(get_heights_fn):
        terrain_h = get_heights_fn(foot_xy).reshape(foot_height.shape)
    else:
        terrain_h = torch.zeros_like(foot_height)
    rel_height = foot_height - terrain_h

    # --- penalty only when below target (smooth)
    height_error = torch.relu(target_height - rel_height)
    penalty = torch.square(height_error) * swing

    gait_mask = (env.current_gait_id == target_gait_id).float()
    return torch.exp(-scale * torch.sum(penalty, dim=1)) * gait_mask


# Arm–leg momentum balance
# Penalizes residual whole-body angular momentum, particularly in the yaw (vertical) direction.
# Encourages anti-phase yaw swing, where the arms move in opposition to the legs to cancel out leg-induced rotation
# - Whole-body Momentum Minimization
# - Arm Symmetry and Coordination
def arm_leg_momentum_balance(
    env,
    robot_cfg: SceneEntityCfg,
    left_arm_cfg: SceneEntityCfg,
    right_arm_cfg: SceneEntityCfg,
    scale: float = 1.0,
    target_gait_id: int = 3,
) -> torch.Tensor:
    """Simplified angular momentum reward (yaw only)."""
    asset = env.scene[robot_cfg.name]

    body_pos = asset.data.body_pos_w  # (N, B, 3)
    body_vel = asset.data.body_lin_vel_w  # (N, B, 3)

    # Cache body masses for efficiency and rebuild when shape changes.
    masses = getattr(env, "_body_masses", None)
    expected_shape = (body_pos.shape[0], body_pos.shape[1], 1)
    if masses is None or tuple(masses.shape) != expected_shape:
        raw_masses = asset.root_physx_view.get_masses()
        if raw_masses.ndim == 1:
            raw_masses = raw_masses.unsqueeze(0).expand(body_pos.shape[0], -1)
        masses = raw_masses.unsqueeze(-1).to(body_pos.device)
        env._body_masses = masses

    # --- CoM
    com = torch.sum(body_pos * masses, dim=1) / torch.sum(masses, dim=1)

    def momentum_z(body_ids) -> torch.Tensor:
        r = body_pos[:, body_ids, :] - com.unsqueeze(1)
        v = body_vel[:, body_ids, :]
        m = masses[:, body_ids, :]
        # r x (m v) -> z = x*vy - y*vx
        return torch.sum(m * (r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0]), dim=1)

    l_total = momentum_z(slice(None))
    l_la = momentum_z(left_arm_cfg.body_ids)
    l_ra = momentum_z(right_arm_cfg.body_ids)

    reward = -(l_total**2) - 0.4 * (l_la - l_ra) ** 2
    gait_mask = (env.current_gait_id == target_gait_id).float()
    return reward * scale * gait_mask
