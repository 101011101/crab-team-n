"""Mission configuration. Everything you tune during the demo lives here."""

from dataclasses import dataclass, field


@dataclass
class BotConfig:
    bot_id: int
    name: str
    ip: str
    sector: str           # "left" or "right": which way this bot prefers to sweep
    stream_port: int = 81
    command_port: int = 80

    @property
    def stream_url(self) -> str:
        return f"http://{self.ip}:{self.stream_port}/stream"

    @property
    def snapshot_url(self) -> str:
        return f"http://{self.ip}:{self.stream_port}/jpg"

    @property
    def command_url(self) -> str:
        return f"http://{self.ip}:{self.command_port}"


# ---------------------------------------------------------------
# FLEET -- must match the BOT_ID and static IPs you flashed
# ---------------------------------------------------------------

BOTS = [
    BotConfig(bot_id=1, name="scout-1", ip="192.168.1.51", sector="left"),
    BotConfig(bot_id=2, name="scout-2", ip="192.168.1.52", sector="right"),
]


# ---------------------------------------------------------------
# GREEN BLOCK DETECTION (HSV, OpenCV ranges: H 0-179, S 0-255, V 0-255)
# ---------------------------------------------------------------
# Run calibrate.py in the real arena lighting and paste the result here.
# This is the single most demo-critical number set in the project.

@dataclass
class VisionConfig:
    hsv_low: tuple = (35, 80, 60)
    hsv_high: tuple = (85, 255, 255)

    min_area_px: int = 120          # ignore specks and sensor noise
    max_detections: int = 5

    # A blob wider than this fraction of the frame means we have arrived.
    claim_area_fraction: float = 0.10

    # Morphology kernel size, removes speckle before contour finding.
    kernel: int = 5


# ---------------------------------------------------------------
# DRIVE
# ---------------------------------------------------------------
# Speeds are PCA9685 units, -4095..4095. The stock sketch used 2000.

@dataclass
class DriveConfig:
    cruise: int = 1900              # forward speed when approaching
    search_forward: int = 1700      # forward speed while sweeping
    spin: int = 1750                # speed of each track during a spin
    steer_gain: float = 1600.0      # how hard to correct per unit of bearing
    min_move: int = 1200            # below this the motors stall instead of turning

    command_hz: float = 10.0        # heartbeat rate; robot stops after 500 ms


# ---------------------------------------------------------------
# MOTION -- waypoint following
# ---------------------------------------------------------------
# WARNING: track_width_m and max_speed_mps are PLACEHOLDERS. Nobody has
# measured this chassis. Every number the waypoint controller produces is
# derived from them, so measure both before trusting a waypoint run:
#   track width -- wheel centre to wheel centre, metres, with a ruler
#   top speed   -- time the robot over 2 m commanded at 4095

@dataclass
class MotionConfig:
    track_width_m: float = 0.15     # PLACEHOLDER, NOT MEASURED
    max_speed_mps: float = 0.35     # PLACEHOLDER, NOT MEASURED

    # Bearing error above turn_in_place_rad stops the robot and spins it.
    # It drives again only below resume_drive_rad. The gap is hysteresis:
    # one threshold makes the robot chatter between spinning and creeping.
    turn_in_place_rad: float = 0.35
    resume_drive_rad: float = 0.15

    turn_gain: float = 2.5          # rad/s of turn per rad of bearing error
    max_turn_rate: float = 2.0      # rad/s, about what DRIVE.spin produces
    approach_gain: float = 0.8      # m/s per metre left; slows into the goal

    # Stall detection. A robot commanded to move that travels less than
    # stall_move_m per tick, for stall_ticks running, is wedged. At 20 Hz
    # and 0.35 m/s a free robot covers about 17 mm per tick, so 2 mm is
    # comfortably below anything real and above pose noise.
    stall_move_m: float = 0.002
    stall_ticks: int = 20           # one second at 20 Hz

    @property
    def k(self) -> float:
        """The constant in l = v + k*w. Half the track width, by geometry."""
        return self.track_width_m / 2.0


# ---------------------------------------------------------------
# BEHAVIOUR TIMING (seconds)
# ---------------------------------------------------------------

@dataclass
class BehaviorConfig:
    search_forward_s: float = 1.6   # creep forward this long ...
    search_spin_s: float = 0.9      # ... then turn this long, and repeat
    approach_timeout_s: float = 12.0
    claim_hold_s: float = 1.2       # sit still at the block so it is visible on video
    backoff_reverse_s: float = 1.0
    backoff_turn_s: float = 1.1
    refractory_s: float = 8.0       # after claiming, ignore blocks for this long

    # A detection must persist this long before we commit to approaching,
    # which rejects single-frame colour noise.
    lock_on_s: float = 0.25

    # No frame newer than this means the bot is considered offline.
    offline_after_s: float = 2.5


# ---------------------------------------------------------------
# TARGET DEDUPLICATION
# ---------------------------------------------------------------

@dataclass
class TargetConfig:
    # Without global coordinates we cannot prove two sightings are the same
    # block. We merge a new claim into an existing one when it arrives soon
    # after, at a similar apparent size. See README for the limits of this.
    merge_window_s: float = 10.0
    merge_size_ratio: float = 0.45  # sizes within +/-45% count as similar


VISION = VisionConfig()
DRIVE = DriveConfig()
BEHAVIOR = BehaviorConfig()
MOTION = MotionConfig()
TARGETS = TargetConfig()


# ---------------------------------------------------------------
# MARKER POSE
# ---------------------------------------------------------------
# Rung 1. The robots have no encoders, so every pose in the project comes
# from a printed tag whose position somebody measured with a tape.

@dataclass
class MarkerConfig:
    # Which ArUco family the tags were printed from. 4X4_50 is the smallest
    # bit grid, so its cells are the largest for a given printed size, which
    # is what matters at 320x240. Any cv2.aruco.DICT_* name works, but the
    # printed sheet and the map file must agree with this.
    dictionary: str = "DICT_4X4_50"

    # Printed side length of the black square, metres. Research says 10-20 cm
    # for 2-3 m detection range at this resolution. A per-marker size_m in the
    # map file overrides this, so mixed sizes are allowed.
    default_size_m: float = 0.15

    map_path: str = "marker_map.json"
    calib_dir: str = "calib"
    calib_name: str = "bot{bot_id}.npz"

    # Guessed intrinsics for an uncalibrated camera. OV2640 with the stock
    # lens. A guess gets a robot moving; it does not pass the rung-1 gate.
    default_hfov_deg: float = 65.0

    # A tag smaller than this across is too few pixels for its corners to
    # carry a pose, whatever the detector claims.
    min_side_px: float = 12.0
    min_perimeter_rate: float = 0.02   # detector: ignore tags under 2% of image perimeter

    # Beyond this angle off the tag's face there is not enough of the pattern
    # left to read. Used by the synthetic forward model so tests only ask for
    # poses a real camera could deliver.
    max_view_angle_deg: float = 70.0

    # Sanity bounds on a solved pose. The floor is flat and the camera is
    # bolted on, so a solution outside these is not a competing hypothesis.
    max_body_tilt_deg: float = 25.0
    max_body_z_m: float = 0.30

    # The lens height is measured, not solved, so a solution that disagrees
    # with it by more than this is a flipped branch, not a competing answer.
    # Generous enough to survive corner noise: at 1 px of corner jitter and
    # 3 m range a correct solve still wanders a few centimetres in z.
    max_camera_height_err_m: float = 0.12

    # A square tag has two poses that fit the corners almost equally well.
    # A second branch counts as a real contender only within these of the
    # best fit; outside them the best fit is simply right.
    #
    # The ratio carries this test, because the best fit's own residual is the
    # only estimate of the corner noise available: a branch fitting 3x worse
    # than the best is worse than the noise can explain. The absolute term is
    # a floor for numerical rounding ONLY. It used to be 0.8 px, which on
    # exact corners made every stuck solver branch a contender and rejected
    # 25 of 69 recoverable single-marker poses.
    ambiguity_ratio: float = 3.0
    ambiguity_abs_px: float = 0.05

    # Two surviving branches this close are the same answer as far as driving
    # is concerned. Further apart, and the frame cannot say which is right:
    # the pose is rejected rather than guessed.
    ambiguity_pos_m: float = 0.05
    ambiguity_head_deg: float = 4.0

    # Per-marker reprojection error above which a marker is dropped from a
    # multi-marker fit. Usually means its map entry was measured wrong.
    max_reproj_px: float = 4.0


# Where each camera sits on its chassis, and where it is aimed.
# WARNING: PLACEHOLDERS. Only z has even a guessed value. Measure the lens
# height above the floor per robot before trusting a pose, and keep pan/tilt
# at zero unless somebody has mapped servo degrees onto these radians.

@dataclass
class MountConfig:
    x: float = 0.0          # metres forward of the body origin
    y: float = 0.0          # metres left
    z: float = 0.10         # lens height above the floor -- PLACEHOLDER
    pan_rad: float = 0.0    # positive turns the lens to the robot's left
    tilt_rad: float = 0.0   # positive raises it
    roll_rad: float = 0.0


MARKERS = MarkerConfig()

MOUNTS = {
    "default": MountConfig(),
    1: MountConfig(),
    2: MountConfig(),
}
