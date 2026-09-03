#!/usr/bin/env python3
"""Real-robot rollout client for VLA-JEPA on kobo (single-arm now -- panda_right's
control box is down, panda_left is the surviving/only arm; see
single_panda_multi_mode_controller.launch's comment on hubble).

**DO NOT RUN THIS ON THE REAL ROBOT WITHOUT LINE-BY-LINE REVIEW**, plus a
--dry-run pass first, then a supervised low-num-samples run WITH motion
enabled and a human at the E-stop the whole time. This is adapted from
openpi/scripts/inference_online.py (itself marked DRAFT/UNTESTED, written
with no ROS/hardware environment to test against) -- the ROS
topics/SHM-camera protocol/safety-clamping logic are carried over from
that file's hard-won knowledge, but the single-arm topic remap and the
VLA-JEPA websocket integration are new and UNTESTED against real hardware.

Architecture (confirmed 2026-09-03 against the actual kobo_ros_core_ias
catkin workspace on hubble, not guessed):
  - Camera: reads a POSIX shared-memory segment named "zed_shm" (NOT a ROS
    Image topic) -- same binary layout as inference_online.py: 168-byte
    header (uint64 frame_count, int32 active_buf) then 3 rotating buffer
    slots, each holding one RGB frame's worth of bytes for the wrist camera
    followed by the external camera (both 1280x720x3 uint8, ordered
    [[likely rgb, rgb_external]] -- copied verbatim from the dual-arm
    reference, NOT independently re-verified that the slot order/camera
    identity is unchanged post-repair).
  - Robot control: ROS topics under the `panda_dual` namespace (kept
    unchanged in single-arm mode -- confirmed via
    single_panda_multi_mode_controller.launch, NOT renamed to
    "panda_single" or similar):
      /panda_dual/multi_mode_controller/desired_joint_position (JointState) -- joint-space target
      /panda_dual/multi_mode_controller/panda_left/target_pose (PoseStamped) -- cartesian-space target
      /panda_dual/joint_states (JointState, subscribed) -- joint feedback
      /panda_dual/multi_mode_controller/switch_control/goal (SwitchControlActionGoal) -- mode switch
      /panda_dual/panda_left/franka_gripper/move (actionlib MoveAction) -- gripper
    TF frames: panda_left_hand, panda_left_link0, base_link (world frame).
  - VLA-JEPA inference: NOT in-process (unlike inference_online.py's JAX
    model) -- talks over websocket to a SEPARATELY-running
    deployment/model_server/server_qc_policy.py (or server_policy.py for
    plain BC), using the SAME WebsocketClientPolicy client class the
    LIBERO/LIBERO-Plus/SimplerEnv sim eval already uses (portable, no ROS
    deps -- see deployment/model_server/tools/websocket_policy_client.py),
    and the SAME request contract confirmed against
    examples/LIBERO/model2libero_interface.py's step(): {"batch_images":
    [[img, img_external]], "instructions": [prompt], "state": [state_vec],
    ...} -> response["data"]["normalized_actions"].

UNVERIFIED / needs confirming at the robot before a real run:
  - Single-arm startup joint pose (init_joint_config below is a PLACEHOLDER
    -- the dual-arm reference's 18-dim vector doesn't apply to one arm, and
    no single-arm equivalent has been recorded yet. Fill this in from a
    real `rostopic echo /panda_dual/joint_states` snapshot at a known-good
    starting pose before running with --motion.)
  - Whether "zed_shm"'s slot layout / camera-to-slot-index mapping is
    unchanged since the arm repair (camera hardware itself wasn't
    mentioned as touched, but not independently re-checked here).
  - VLA-JEPA's expected state vector layout/dim for kobo (KoboDataConfig's
    "observation.state" via config.py) -- assumed 8-dim (7-DOF task-space
    pose + gripper) matching openpi's kobo_policy.py convention, since
    VLA-JEPA is being fine-tuned on the SAME kobo dataset -- not yet
    confirmed against the actual fine-tuned checkpoint's norm_stats/config.
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import actionlib
import cv2
import mmap
import numpy as np
import posix_ipc
import rospy
import tf
import tf.transformations as tft
from dual_panda_multi_mode_controllers.msg import ControlMode, SwitchControlActionGoal
from franka_gripper.msg import MoveAction, MoveGoal
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for deployment.model_server.tools import
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy  # noqa: E402

# =======================================
# ===== OopsieData project specific =====
# =======================================
# pip install oopsie-data-tools ; oopsie-data init ; oopsie-data new-profile
# (writes a robot profile describing robot_id, control_freq, joint names,
# camera names -- required, no bundled default, see their own docs). Saves
# BOTH successful and failed rollouts, for contribution to
# https://oopsie-data.com (failure data is the whole point of that project).
from oopsie_data_tools.annotation_tool.rollout_annotator import WebRolloutAnnotator
from oopsie_data_tools.utils.robot_profile.robot_profile import load_robot_profile
# =======================================

# --- SHM camera protocol constants (copied verbatim from
# openpi/scripts/inference_online.py -- see that file's own header for
# where these numbers come from; NOT independently re-derived here) ---
WIDTH, HEIGHT = 1280, 720
CHANNELS = 3
IMG_BYTES = WIDTH * HEIGHT * CHANNELS
DEPTH_BYTES = WIDTH * HEIGHT * 4
CAM_SET_SIZE = (IMG_BYTES * 3) + (DEPTH_BYTES * 3)
HEADER_SIZE = 168

# Single-arm namespace, confirmed against
# single_panda_multi_mode_controller.launch on hubble (2026-09-03) -- NOT
# renamed from the dual-arm "panda_dual", just one arm's resources active.
ROBOT_NS = "panda_dual"
ARM_ID = "panda_left"  # the surviving arm; panda_right's control box is down

# Per-step motion safety limits -- SAME values as inference_online.py's,
# carried over as a conservative starting point, not re-tuned for this
# single-arm setup or VLA-JEPA's action scale specifically. Re-check before
# trusting these on a real run.
MAX_DELTA_TRANSLATION_M = 0.1
MAX_DELTA_ROTATION_RAD = np.radians(15.0)

# PLACEHOLDER -- see module docstring. Fill in from a real joint_states
# snapshot at a known-good start pose before running with --motion.
SINGLE_ARM_INIT_JOINT_CONFIG = None  # e.g. np.array([q1..q7, finger1, finger2])


def _clamp_target_pose(current_tr, current_quat, target_tr, target_quat, max_translation, max_rotation_rad):
    """Identical to inference_online.py's version -- clamps target pose to
    be at most max_translation/max_rotation_rad away from current pose,
    preserving direction. Quaternions [x,y,z,w] (tf convention)."""
    current_tr = np.asarray(current_tr, dtype=np.float64)
    target_tr = np.asarray(target_tr, dtype=np.float64)
    delta = target_tr - current_tr
    dist = np.linalg.norm(delta)
    if dist > max_translation and dist > 1e-9:
        target_tr = current_tr + delta / dist * max_translation

    dot = np.clip(np.abs(np.dot(current_quat, target_quat)), -1.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    if angle > max_rotation_rad and angle > 1e-9:
        fraction = max_rotation_rad / angle
        target_quat = np.asarray(tft.quaternion_slerp(current_quat, target_quat, fraction))

    return target_tr, target_quat


class VlaJepaOnlineRosInterface:
    def __init__(
        self,
        host: str,
        port: int,
        num_samples: int,
        horizon_length: int,
        prompt: str,
        robot_profile_path: Path,
        data_root_dir: Path,
        operator_name: str,
        annotator_name: str,
        annotator_port: int,
        wait_for_annotation: bool,
        dry_run: bool,
    ):
        rospy.init_node("vlajepa_online_inference_node")

        self.horizon_length = horizon_length
        self.num_samples = num_samples
        self.prompt = prompt
        self.dry_run = dry_run
        if dry_run:
            rospy.logwarn("DRY RUN MODE: no robot motion will be commanded.")

        # 1. Connect to the (separately-running) VLA-JEPA policy server --
        # deployment/model_server/server_qc_policy.py or server_policy.py,
        # started independently before this script runs.
        rospy.loginfo(f"Connecting to VLA-JEPA policy server at {host}:{port}...")
        self.policy_client = WebsocketClientPolicy(host=host, port=port)
        rospy.loginfo(f"Connected. Server metadata: {self.policy_client.get_server_metadata()}")

        # 2. ROS Publishers & Action Clients -- single-arm subset of
        # inference_online.py's dual-arm setup, same topic structure under
        # the panda_dual namespace, panda_left only.
        self.pub_joint_target = rospy.Publisher(
            f"/{ROBOT_NS}/multi_mode_controller/desired_joint_position", JointState, queue_size=10
        )
        self.subscribe_joint_states = rospy.Subscriber(
            f"/{ROBOT_NS}/joint_states", JointState, self.__process_joint_states, queue_size=1
        )
        self.pub_cartesian_target = rospy.Publisher(
            f"/{ROBOT_NS}/multi_mode_controller/{ARM_ID}/target_pose", PoseStamped, queue_size=0
        )
        self.gripper_move_client = actionlib.SimpleActionClient(
            f"/{ROBOT_NS}/{ARM_ID}/franka_gripper/move", MoveAction
        )
        rospy.loginfo("Waiting for gripper server...")
        self.gripper_move_client.wait_for_server(rospy.Duration(5.0))

        self.transform_listener = tf.TransformListener()

        # 3. Shared-memory camera connection
        self.setup_shm()

        self.rate = rospy.Rate(30)
        self.prev_frame_count = 0

        # =======================================
        # ===== OopsieData project specific =====
        # =======================================
        robot_profile = load_robot_profile(robot_profile_path)
        self.rollout_annotator = WebRolloutAnnotator(
            robot_profile=robot_profile,
            data_root_dir=data_root_dir,
            port=annotator_port,
            wait_for_annotation=wait_for_annotation,
            operator_name=operator_name,
            annotator_name=annotator_name,
        )
        self.rollout_annotator.start()
        # =======================================

    def setup_shm(self):
        try:
            self.memory = posix_ipc.SharedMemory("zed_shm")
            self.map_file = mmap.mmap(self.memory.fd, self.memory.size)
            self.mv = memoryview(self.map_file)
            rospy.loginfo("Connected to shared memory (zed_shm).")
        except Exception as e:
            rospy.logerr(f"SHM connection failed: {e}")
            raise

    def __process_joint_states(self, data):
        self.joint_state_pos = np.array(data.position)

    def startup_procedure(self):
        rospy.sleep(1)
        rospy.loginfo("Loading joint controller")
        self.switch_controller("joint")

        if SINGLE_ARM_INIT_JOINT_CONFIG is None:
            rospy.logwarn(
                "SINGLE_ARM_INIT_JOINT_CONFIG is not set -- skipping move-to-init-pose. "
                "Position the arm manually (e.g. via guiding mode) before starting a real rollout."
            )
        else:
            rate = rospy.Rate(200)
            target_state = JointState()
            target_state.name = [
                f"{ARM_ID}_joint1", f"{ARM_ID}_joint2", f"{ARM_ID}_joint3", f"{ARM_ID}_joint4",
                f"{ARM_ID}_joint5", f"{ARM_ID}_joint6", f"{ARM_ID}_joint7",
                f"{ARM_ID}_finger_joint1", f"{ARM_ID}_finger_joint2",
            ]
            max_joint_diff = rospy.get_param(f"/PandaJointImpedanceController_{ARM_ID}/max_joint_diff") * np.pi / 180
            max_joint_diff /= 2

            rospy.loginfo("Going to initial pose")
            while np.linalg.norm(self.joint_state_pos - SINGLE_ARM_INIT_JOINT_CONFIG) > 0.08 and not rospy.is_shutdown():
                delta = SINGLE_ARM_INIT_JOINT_CONFIG - self.joint_state_pos
                mask = np.abs(delta) >= max_joint_diff
                delta[mask] = max_joint_diff * np.sign(delta[mask])
                target_state.position = self.joint_state_pos + delta
                self.pub_joint_target.publish(target_state)
                rate.sleep()
            time.sleep(5)

        rospy.loginfo("Loading cartesian controller")
        self.switch_controller("cartesian")

    def switch_controller(self, start_controllers):
        pub = rospy.Publisher(
            f"/{ROBOT_NS}/multi_mode_controller/switch_control/goal", SwitchControlActionGoal, queue_size=1
        )
        switcher = SwitchControlActionGoal()
        mode = ControlMode()
        mode.ctrl_resources = [ARM_ID]
        mode.ctrl_type = start_controllers
        switcher.goal.ctrl_modes.mode_list = [mode]
        switcher.goal_id.stamp = switcher.header.stamp = rospy.Time.now()
        pub.publish(switcher)
        rospy.sleep(1)
        pub.publish(switcher)
        rospy.sleep(1)

    def send_gripper_command(self, width):
        msg = MoveGoal()
        msg.width = float(np.clip(width * 2.0, 0, 0.08))
        msg.speed = 0.1
        self.gripper_move_client.send_goal(msg)

    def _read_camera_and_state(self, world_frame: str):
        """Blocks until a new SHM camera frame is available, then reads
        current hand pose + gripper width. Returns (img_wrist, img_external,
        state_8, hand_quat). See module docstring for the UNVERIFIED note on
        slot layout/camera identity post-repair."""
        while True:
            if rospy.is_shutdown():
                raise rospy.ROSInterruptException("Shutdown requested while waiting for a new camera frame.")
            header_peek = self.mv[:12]
            frame_count, active_buf = struct.unpack("Qi", header_peek)
            if frame_count != self.prev_frame_count:
                break
            time.sleep(0.001)
        self.prev_frame_count = frame_count

        slot_offset = HEADER_SIZE + (((active_buf - 1) % 3) * CAM_SET_SIZE)
        rgb_data = self.mv[slot_offset + IMG_BYTES: slot_offset + (2 * IMG_BYTES)]
        rgb_data_external = self.mv[slot_offset + 2 * IMG_BYTES: slot_offset + (3 * IMG_BYTES)]
        img_wrist = np.frombuffer(rgb_data, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
        img_external = np.frombuffer(rgb_data_external, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
        img_external = cv2.flip(cv2.flip(img_external, 0), 1)
        img_wrist = cv2.flip(cv2.flip(img_wrist, 0), 1)

        ee_tr, ee_rot = self.transform_listener.lookupTransform(world_frame, f"{ARM_ID}_hand", rospy.Time(0))

        # Quaternion hemisphere: inference_online.py found the live TF
        # quaternion consistently flipped vs. training data's hemisphere for
        # the pi0/kobo_cube dataset (see that file's long comment). NOT
        # re-verified against VLA-JEPA's own kobo fine-tune data -- if
        # actions look consistently wrong-signed on a dry run, check this.
        if ee_rot[3] > 0.0:
            ee_rot = tuple(-c for c in ee_rot)
        qx, qy, qz, qw = ee_rot

        gripper_open = (2.0 * self.joint_state_pos[7]) > 0.04 if hasattr(self, "joint_state_pos") else True
        gripper_flag = 1.0 if gripper_open else 0.0

        state_8 = np.array([*ee_tr, qx, qy, qz, qw, gripper_flag], dtype=np.float32)
        return img_wrist, img_external, state_8, np.asarray(ee_rot)

    def run(self, max_timesteps: int, open_loop_horizon: int):
        if self.dry_run:
            rospy.logwarn("DRY RUN: skipping startup_procedure() -- no controller switch, no motion.")
            rospy.sleep(2)
        else:
            self.startup_procedure()
        WORLD_FRAME = "base_link"

        try:
            tl, ql = self.transform_listener.lookupTransform(WORLD_FRAME, f"{ARM_ID}_link0", rospy.Time(0))
            self.T_base_l0 = tft.concatenate_matrices(tft.translation_matrix(tl), tft.quaternion_matrix(ql))
        except Exception as e:
            rospy.logerr(f"Static TF lookup failed: {e}")
            raise
        self.T_hand_ee = tft.translation_matrix([0.0, 0.0, 0.1034])  # copied from inference_online.py, not re-derived

        rospy.loginfo("Press Ctrl+C to end the session.")
        try:
            while not rospy.is_shutdown():
                if self.dry_run:
                    # No real transitions this run -- don't touch the
                    # annotator at all (matches inference_online.py's
                    # replay-buffer reasoning: unexecuted "actions"
                    # shouldn't be recorded as if they were real).
                    instruction = self.prompt
                else:
                    # =======================================
                    # ===== OopsieData project specific =====
                    # =======================================
                    self.rollout_annotator.reset_episode_recorder()
                    task = self.rollout_annotator.wait_for_task()
                    # =======================================
                    instruction = task if task else self.prompt

                actions_from_chunk_completed = 0
                action_chunk = None

                for t_step in range(max_timesteps):
                    try:
                        img_wrist, img_external, state_8, ee_rot = self._read_camera_and_state(WORLD_FRAME)

                        if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= open_loop_horizon:
                            actions_from_chunk_completed = 0
                            request = {
                                "batch_images": [[img_wrist, img_external]],
                                "instructions": [instruction],
                                "state": [state_8],
                                "do_sample": False,
                            }
                            response = self.policy_client.infer(request)
                            action_chunk = np.asarray(response["data"]["normalized_actions"])[0]  # [horizon, action_dim]

                        action = action_chunk[actions_from_chunk_completed]
                        actions_from_chunk_completed += 1

                        target_tr = action[:3]
                        pred_quat = action[3:7]
                        grasp_width = float(action[7])

                        if np.dot(ee_rot, pred_quat) < 0.0:
                            pred_quat = -pred_quat
                        norm = np.linalg.norm(pred_quat)
                        safe_quat = pred_quat / norm if norm > 0 else pred_quat

                        current_hand_tr, current_hand_quat = self.transform_listener.lookupTransform(
                            WORLD_FRAME, f"{ARM_ID}_hand", rospy.Time(0)
                        )
                        target_tr, safe_quat = _clamp_target_pose(
                            current_hand_tr, current_hand_quat, target_tr, safe_quat,
                            MAX_DELTA_TRANSLATION_M, MAX_DELTA_ROTATION_RAD,
                        )

                        T_world_target = tft.concatenate_matrices(
                            tft.translation_matrix(target_tr), tft.quaternion_matrix(safe_quat)
                        )
                        T_world_ee_target = T_world_target @ self.T_hand_ee
                        T_l0_target = np.linalg.inv(self.T_base_l0) @ T_world_ee_target

                        msg = PoseStamped()
                        msg.header.stamp = rospy.Time.now()
                        msg.header.frame_id = f"{ARM_ID}_link0"
                        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = T_l0_target[:3, 3]
                        q = tft.quaternion_from_matrix(T_l0_target)
                        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q

                        if self.dry_run:
                            rospy.loginfo(
                                f"DRY RUN step {t_step}: would publish target_tr={target_tr}, "
                                f"quat={safe_quat}, grasp_width={grasp_width:.3f} (not sent)"
                            )
                        else:
                            self.pub_cartesian_target.publish(msg)
                            self.send_gripper_command(grasp_width)

                        if not self.dry_run:
                            # =======================================
                            # ===== OopsieData project specific =====
                            # =======================================
                            self.rollout_annotator.record_step(
                                observation={
                                    "image_observation": {"wrist": img_wrist, "external": img_external},
                                    "robot_state": {
                                        "cartesian_position": state_8[:3],
                                        "gripper_position": state_8[7:8],
                                    },
                                },
                                action={
                                    "cartesian_position": target_tr,
                                    "gripper_position": np.array([grasp_width]),
                                },
                            )
                            # =======================================

                        self.rate.sleep()
                    except KeyboardInterrupt:
                        break

                if not self.dry_run:
                    # =======================================
                    # ===== OopsieData project specific =====
                    # =======================================
                    self.rollout_annotator.finish_rollout(instruction=instruction)
                    # =======================================
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    # VLA-JEPA policy server connection (start deployment/model_server/
    # server_qc_policy.py or server_policy.py separately first)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--horizon-length", type=int, default=5)
    parser.add_argument("--open-loop-horizon", type=int, default=5)
    parser.add_argument("--max-timesteps", type=int, default=600)
    parser.add_argument("--prompt", type=str, default="pick up the cube and place it on the red tape")

    # OopsieData
    parser.add_argument("--robot-profile", type=Path, required=True, help="from `oopsie-data new-profile`")
    parser.add_argument("--data-root-dir", type=Path, default=Path("./data"))
    parser.add_argument("--operator-name", type=str, default="<operator_name>")
    parser.add_argument("--annotator-name", type=str, default="<annotator_name>")
    parser.add_argument("--annotator-port", type=int, default=5003)
    parser.add_argument("--no-wait-for-annotation", dest="wait_for_annotation", action="store_false", default=True)

    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate camera/state reading + policy inference with ZERO robot motion. "
             "Run this before any --motion run.",
    )
    args, _ = parser.parse_known_args()

    try:
        node = VlaJepaOnlineRosInterface(
            host=args.host, port=args.port, num_samples=args.num_samples, horizon_length=args.horizon_length,
            prompt=args.prompt, robot_profile_path=args.robot_profile, data_root_dir=args.data_root_dir,
            operator_name=args.operator_name, annotator_name=args.annotator_name,
            annotator_port=args.annotator_port, wait_for_annotation=args.wait_for_annotation,
            dry_run=args.dry_run,
        )
        node.run(max_timesteps=args.max_timesteps, open_loop_horizon=args.open_loop_horizon)
    except rospy.ROSInterruptException:
        pass
