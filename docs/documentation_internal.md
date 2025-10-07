# mjlab: Comprehensive Technical Guide

> **TL;DR**: mjlab = Isaac Lab's proven manager-based API + MuJoCo Warp's GPU-accelerated physics. Lightweight, fast iteration, MuJoCo-native implementation for RL robotics research.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Robot Arm Support](#robot-arm-support)
4. [Core Components](#core-components)
5. [Entity System](#entity-system)
6. [Manager-Based Environment](#manager-based-environment)
7. [Task Configuration](#task-configuration)
8. [Asset Zoo](#asset-zoo)
9. [Physics & Simulation](#physics--simulation)
10. [Training & Deployment](#training--deployment)
11. [Migration from Isaac Lab](#migration-from-isaac-lab)
12. [Performance & Best Practices](#performance--best-practices)
13. [Limitations & Roadmap](#limitations--roadmap)

---

## Overview

### What is mjlab?

mjlab is a **GPU-accelerated robotics simulation framework** for reinforcement learning that combines:
- **Isaac Lab's manager-based API** (proven abstractions for RL)
- **MuJoCo Warp physics** (fast, GPU-native rigid-body simulation)
- **Minimal dependencies** (no Omniverse, no USD overhead)

### Why mjlab?

**Problem with existing tools:**
- **Isaac Lab**: Great API but heavy installation, slow startup, Omniverse overhead
- **MJX**: Fast but JAX learning curve, poor collision scaling with 'jax' backend
- **Newton**: New generic simulator, but lacks MuJoCo's ecosystem maturity

**mjlab's Solution:**
- Direct MuJoCo Warp integration (no translation layers)
- Fast startup and debugging (standard Python pdb works)
- Focused scope: rigid-body robotics and RL (not trying to do everything)
- One-line installation: `uvx --from mjlab demo`

### Key Statistics
- **135 Python files** in codebase
- **~15K lines** of focused, maintainable code
- **2 reference robots** (Unitree G1 humanoid, Go1 quadruped)
- **2 task types** (velocity tracking, motion imitation)
- **Beta status** (breaking changes expected)

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────┐
│           Manager-Based RL Environment          │
│  ┌──────────────────────────────────────────┐  │
│  │  Managers (Observation, Reward, Action,  │  │
│  │  Command, Event, Termination, Curriculum)│  │
│  └──────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│                Scene Configuration              │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Entities │  │ Terrain  │  │ Physics Cfg  │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
├─────────────────────────────────────────────────┤
│            MuJoCo Warp Simulation               │
│  ┌──────────────────────────────────────────┐  │
│  │   GPU-accelerated rigid body physics     │  │
│  │   (mjModel, mjData, MjSpec)              │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Directory Structure

```
mjlab/
├── src/mjlab/
│   ├── envs/              # Environment base classes
│   │   ├── manager_based_env.py      # Base environment
│   │   └── manager_based_rl_env.py   # RL environment
│   ├── managers/          # Manager implementations
│   │   ├── action_manager.py
│   │   ├── observation_manager.py
│   │   ├── reward_manager.py
│   │   ├── command_manager.py
│   │   ├── event_manager.py
│   │   ├── termination_manager.py
│   │   └── curriculum_manager.py
│   ├── scene/             # Scene configuration
│   │   └── scene.py
│   ├── entity/            # Entity system (robots, objects)
│   │   ├── entity.py
│   │   └── data.py
│   ├── sim/               # MuJoCo simulation wrapper
│   │   ├── sim.py
│   │   ├── sim_data.py
│   │   └── randomization.py
│   ├── asset_zoo/         # Robot assets
│   │   └── robots/
│   │       ├── unitree_g1/    # Humanoid
│   │       └── unitree_go1/   # Quadruped
│   ├── tasks/             # Task implementations
│   │   ├── velocity/      # Velocity tracking
│   │   └── tracking/      # Motion imitation
│   ├── terrains/          # Terrain generation
│   └── utils/             # Utilities
└── tests/                 # Test suite
```

---

## Robot Arm Support

### **YES - Architecturally Supported** ✅

mjlab's `Entity` class explicitly supports **fixed-base articulated robots** (robot arms). From the source code:

```python
# Entity Type Matrix (from src/mjlab/entity/entity.py:93-113)
| Type                      | Example                    | is_fixed_base | is_articulated |
|---------------------------|----------------------------|---------------|----------------|
| Fixed Articulated         | Robot arm, door on hinges  | True          | True           |
| Floating Articulated      | Humanoid, quadruped        | False         | True           |
```

### **NO - No Out-of-Box Examples** ❌

**Current Asset Zoo:**
- Unitree G1 (humanoid with arms, but as part of whole-body system)
- Unitree Go1 (quadruped)

**To Add a Robot Arm:**

1. **Get MJCF model** (from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie))
2. **Create EntityCfg**:

```python
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import ActuatorCfg
import mujoco

# Load your arm's MJCF
def get_arm_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file("path/to/arm.xml")

# Define actuators
arm_actuators = ActuatorCfg(
    joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    effort_limit=100.0,
    stiffness=50.0,
    damping=5.0,
)

# Create entity config
ARM_CFG = EntityCfg(
    spec_fn=get_arm_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=(arm_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
    init_state=EntityCfg.InitialStateCfg(
        joint_pos={".*": 0.0},  # Home position
    ),
)
```

3. **Add to scene**:

```python
from mjlab.scene import SceneCfg

scene_cfg = SceneCfg(
    num_envs=4096,
    entities={"robot_arm": ARM_CFG}
)
```

4. **Create task configuration** (see [Task Configuration](#task-configuration))

**Note:** The G1 humanoid has arms (shoulder, elbow, wrist joints) but they're part of the whole-body humanoid control, not standalone arm control.

---

## Core Components

### 1. Entity System

**Location:** `src/mjlab/entity/entity.py`

The `Entity` class is the fundamental building block representing any physical object.

**Entity Types:**

| Base Type      | Articulated | Example Use Cases                    |
|---------------|-------------|--------------------------------------|
| Fixed         | No          | Table, wall, ground plane            |
| Fixed         | Yes         | **Robot arm**, door, cabinet drawer  |
| Floating      | No          | Box, ball, free-floating object      |
| Floating      | Yes         | Humanoid, quadruped, mobile manipulator |

**Key Properties:**
- `is_fixed_base`: Entity is welded to world (no freejoint)
- `is_articulated`: Has joints in kinematic tree
- `is_actuated`: Has actuated joints (motors)

**Entity Configuration:**

```python
@dataclass
class EntityCfg:
    spec_fn: Callable[[], mujoco.MjSpec]  # Returns MuJoCo XML spec
    articulation: EntityArticulationInfoCfg | None
    init_state: InitialStateCfg

    # MjSpec modifiers
    lights: tuple[LightCfg, ...]
    cameras: tuple[CameraCfg, ...]
    textures: tuple[TextureCfg, ...]
    materials: tuple[MaterialCfg, ...]
    sensors: tuple[SensorCfg, ...]
    collisions: tuple[CollisionCfg, ...]
```

**Entity Data Access:**

```python
entity = scene["robot"]

# State access
entity.data.root_state_w         # (N, 13): pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
entity.data.joint_pos            # (N, num_joints)
entity.data.joint_vel            # (N, num_joints)
entity.data.body_pos_w           # (N, num_bodies, 3)
entity.data.body_quat_w          # (N, num_bodies, 4)

# Jacobian access
entity.data.compute_jacobian()
jac_pos, jac_rot = entity.data.jacobian

# Control
entity.write_joint_position_to_sim(target_pos)
entity.write_joint_velocity_to_sim(target_vel)
```

### 2. Scene Configuration

**Location:** `src/mjlab/scene/scene.py`

The `Scene` manages multiple entities and terrain.

```python
@dataclass
class SceneCfg:
    num_envs: int = 1                        # Parallel environments
    env_spacing: float = 2.0                 # Space between envs
    terrain: TerrainImporterCfg | None = None
    entities: dict[str, EntityCfg] = {}      # Named entities
    extent: float | None = None              # Scene extent for rendering
```

**Scene Access:**

```python
# Get entity by name
robot = scene["robot"]

# Get terrain
terrain = scene["terrain"]

# Environment origins
env_origins = scene.env_origins  # (num_envs, 3)
```

### 3. Simulation Wrapper

**Location:** `src/mjlab/sim/sim.py`

Wraps MuJoCo Warp for GPU-accelerated simulation.

```python
@dataclass
class SimulationCfg:
    mujoco: MujocoCfg                  # MuJoCo physics params
    render: RenderCfg                  # Rendering config
    nconmax: int | None = None         # Max contacts
    njmax: int | None = None           # Max constraints
    ls_parallel: bool = True           # Parallel line search (faster)
```

**MuJoCo Physics Config:**

```python
@dataclass
class MujocoCfg:
    timestep: float = 0.002
    integrator: Literal["euler", "implicitfast"] = "implicitfast"
    cone: Literal["pyramidal", "elliptic"] = "pyramidal"
    jacobian: Literal["auto", "dense", "sparse"] = "auto"
    solver: Literal["newton", "cg", "pgs"] = "newton"
    iterations: int = 100
    tolerance: float = 1e-8
    gravity: tuple[float, float, float] = (0, 0, -9.81)
```

---

## Manager-Based Environment

### Manager Architecture

mjlab follows Isaac Lab's **manager-based pattern** where each aspect of the MDP is handled by a dedicated manager:

```python
class ManagerBasedRlEnv:
    observation_manager: ObservationManager   # Computes observations
    action_manager: ActionManager             # Processes actions
    reward_manager: RewardManager             # Computes rewards
    termination_manager: TerminationManager   # Checks episode termination
    command_manager: CommandManager           # Generates commands/goals
    event_manager: EventManager               # Applies randomization events
    curriculum_manager: CurriculumManager     # Curriculum learning
```

### Manager Configuration Pattern

All managers use a **dataclass-based term configuration** with the `term()` helper:

```python
from mjlab.managers import term, ObsTerm, RewTerm, ActionTerm

@dataclass
class ObservationsCfg:
    policy: ObsTerm = term(
        ObsTerm,
        func=mdp.base_lin_vel,
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )

@dataclass
class RewardsCfg:
    lin_vel_z: RewTerm = term(
        RewTerm,
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )
```

### Observation Manager

**Location:** `src/mjlab/managers/observation_manager.py`

Computes vectorized observations for RL policy.

**Key Features:**
- Multi-group observations (e.g., "policy", "critic")
- Automatic concatenation or dict output
- Per-term noise injection
- Clipping and normalization

**Example Configuration:**

```python
@dataclass
class ObservationsCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = term(PolicyCfg)
```

### Action Manager

**Location:** `src/mjlab/managers/action_manager.py`

Processes raw policy actions into simulation commands.

**Built-in Action Terms:**
- `JointAction`: PD control with position targets
- Support for custom action processing

**Example:**

```python
from mjlab.envs.mdp.actions import JointActionCfg

@dataclass
class ActionsCfg:
    joint_pos = JointActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,  # Action scaling
    )
```

### Reward Manager

**Location:** `src/mjlab/managers/reward_manager.py`

Computes weighted sum of reward terms.

**Built-in Reward Functions** (`src/mjlab/envs/mdp/rewards.py`):
- `track_lin_vel_xy_exp`: Velocity tracking (exponential)
- `track_ang_vel_z_exp`: Angular velocity tracking
- `lin_vel_z_l2`: Penalize vertical velocity
- `ang_vel_xy_l2`: Penalize angular velocity
- `joint_torques_l2`: Penalize joint torques
- `joint_vel_l2`: Penalize joint velocities
- `action_rate_l2`: Penalize action changes
- `flat_orientation_l2`: Keep body upright
- And many more...

**Example:**

```python
@dataclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"std": 0.5}
    )
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0
    )
```

### Command Manager

**Location:** `src/mjlab/managers/command_manager.py`

Generates goals/commands for the agent to track.

**Built-in Commands:**
- `UniformVelocityCommand`: Random velocity goals
- Motion tracking commands (from reference dataset)

### Termination Manager

**Location:** `src/mjlab/managers/termination_manager.py`

Determines when episodes end.

**Built-in Terminations:**
- `time_out`: Episode length limit
- `base_contact`: Body touches ground
- `bad_orientation`: Robot falls over
- `joint_pos_limit`: Joint limit violation

---

## Task Configuration

### Task Structure

Tasks are organized by type:

```
tasks/
├── velocity/          # Velocity command tracking
│   ├── velocity_env_cfg.py         # Base environment config
│   ├── mdp/                        # MDP functions
│   ├── config/                     # Robot-specific configs
│   │   ├── g1/
│   │   └── go1/
│   └── rl/                         # RL training configs
└── tracking/          # Motion imitation
    ├── tracking_env_cfg.py
    ├── mdp/
    ├── config/
    └── rl/
```

### Creating a Custom Task

**Step 1: Define MDP Components**

```python
from dataclasses import dataclass
from mjlab.managers import term, ObsTerm, RewTerm, ActionTerm
from mjlab.envs.mdp.actions import JointActionCfg

@dataclass
class MyObservationsCfg:
    @dataclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

    policy: PolicyCfg = term(PolicyCfg)

@dataclass
class MyRewardsCfg:
    task_reward = RewTerm(func=my_custom_reward, weight=1.0)

@dataclass
class MyActionsCfg:
    joint_pos = JointActionCfg(asset_name="robot", joint_names=[".*"])
```

**Step 2: Create Environment Config**

```python
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg

@dataclass
class MyTaskEnvCfg(ManagerBasedRlEnvCfg):
    def __post_init__(self):
        # Scene
        self.scene = SceneCfg(
            num_envs=4096,
            entities={"robot": MY_ROBOT_CFG}
        )

        # Simulation
        self.sim = SimulationCfg(...)

        # MDP
        self.observations = MyObservationsCfg()
        self.actions = MyActionsCfg()
        self.rewards = MyRewardsCfg()
        self.terminations = MyTerminationsCfg()

        # Episode config
        self.episode_length_s = 20.0
        self.decimation = 4  # 50Hz control (with 200Hz sim)
```

**Step 3: Register Task**

```python
# In src/mjlab/tasks/__init__.py
import gymnasium as gym

gym.register(
    id="MyTask-v0",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MyTaskEnvCfg,
    },
)
```

---

## Asset Zoo

### Current Robots

#### 1. Unitree G1 (Humanoid)
**Location:** `src/mjlab/asset_zoo/robots/unitree_g1/`

**Specifications:**
- **DOF**: 23 joints (legs: 12, arms: 8, waist: 3)
- **Height**: 1.27m
- **Mass**: ~35kg (estimated from model)
- **Actuators**:
  - 5020 motors: arms (elbow, shoulder, wrist_roll)
  - 7520-14 motors: hip pitch/yaw, waist yaw
  - 7520-22 motors: hip roll, knee
  - 4010 motors: wrist pitch/yaw
  - Dual 5020 motors: waist pitch/roll, ankles (4-bar linkage)

**Joint List:**
```
Legs (12):    left/right_hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
Arms (8):     left/right_shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_roll, wrist_yaw
Waist (3):    waist_pitch, waist_roll, waist_yaw
Hands (12):   left/right_hand_thumb_0/1/2, index_0/1, middle_0/1 (usually not actuated)
```

**Action Scaling:** Automatically computed as `0.25 * effort_limit / stiffness` per joint

#### 2. Unitree Go1 (Quadruped)
**Location:** `src/mjlab/asset_zoo/robots/unitree_go1/`

**Specifications:**
- **DOF**: 12 (4 legs × 3 joints)
- **Mass**: ~12kg
- **Joints**: FL/FR/RL/RR + hip/thigh/calf

### Adding Custom Robots

**Requirements:**
1. **MJCF format** (`.xml` file) - Convert from URDF using MuJoCo tools
2. **Mesh assets** (`.stl`, `.obj`)
3. **Actuator specifications** (effort limits, stiffness, damping)

**Example Robot Integration:**

```python
from pathlib import Path
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import ActuatorCfg

ROBOT_XML = Path("path/to/robot.xml")

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    # Load mesh assets
    return spec

# Define actuators
actuator_cfg = ActuatorCfg(
    joint_names_expr=[".*"],  # All joints
    effort_limit=100.0,
    stiffness=50.0,
    damping=5.0,
    armature=0.01,  # Reflected inertia
)

ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=(actuator_cfg,),
    ),
    init_state=EntityCfg.InitialStateCfg(
        pos=(0, 0, 1.0),
        joint_pos={".*": 0.0},
    ),
)
```

**Resources:**
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): 100+ robot models
- [MJCF Documentation](https://mujoco.readthedocs.io/en/stable/modeling.html)

---

## Physics & Simulation

### MuJoCo Warp Backend

mjlab uses **MuJoCo Warp** (GPU-accelerated MuJoCo) for parallel simulation:

- **Vectorized environments**: Simulate 1000s of environments in parallel
- **GPU-native**: All physics computation on GPU
- **Batched operations**: No CPU-GPU transfer overhead

### Key Physics Parameters

**Timestep & Integration:**
```python
mujoco_cfg = MujocoCfg(
    timestep=0.002,              # 500Hz physics (typical)
    integrator="implicitfast",   # Stable for stiff systems
)
```

**Contact & Friction:**
```python
mujoco_cfg = MujocoCfg(
    cone="pyramidal",            # Friction cone type
    impratio=1.0,                # Contact impedance ratio
)
```

**Solver:**
```python
mujoco_cfg = MujocoCfg(
    solver="newton",             # Newton, CG, or PGS
    iterations=100,              # Solver iterations
    tolerance=1e-8,              # Convergence tolerance
    ls_iterations=50,            # Line search iterations
)
```

### Domain Randomization

**Physics Randomization:**
```python
from mjlab.managers import EventTermCfg

physics_randomization = EventTermCfg(
    func=mdp.randomize_rigid_body_material,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "static_friction_range": (0.7, 1.3),
        "dynamic_friction_range": (0.7, 1.3),
    },
)
```

**Actuator Randomization:**
```python
actuator_randomization = EventTermCfg(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "stiffness_distribution_params": (0.75, 1.25),
        "damping_distribution_params": (0.75, 1.25),
    },
)
```

### Terrain Generation

**Built-in Terrains** (`src/mjlab/terrains/`):
- `plane`: Flat ground
- `random_rough`: Random height field
- `sloped_terrain`: Inclined plane
- `pyramid_stairs`: Stairs
- Custom terrains via heightfield

**Example:**
```python
from mjlab.terrains import TerrainImporterCfg, TerrainGeneratorCfg

terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        curriculum=True,
        terrain_proportions=[0.5, 0.5],  # Mix of terrains
    ),
)
```

---

## Training & Deployment

### Training Pipeline

**1. Define Task** (see [Task Configuration](#task-configuration))

**2. Create RL Config:**

```python
from dataclasses import dataclass
from mjlab.rl import RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg

@dataclass
class MyTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100

    algorithm = RslRlPpoActorCriticCfg(
        learning_rate=5e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
    )
```

**3. Train:**

```bash
MUJOCO_GL=egl uv run train MyTask-v0 \
    --env.scene.num-envs 4096 \
    --headless
```

**4. Evaluate:**

```bash
uv run play --task MyTask-v0-Play \
    --num-envs 1 \
    --wandb-run-path your-org/project/run-id
```

### WandB Integration

**Automatic Logging:**
- Episode returns
- Reward term breakdown
- Video rollouts (configurable)
- Policy checkpoints

**Motion Dataset Management:**
```bash
# Upload motion to WandB registry
MUJOCO_GL=egl uv run scripts/tracking/csv_to_npz.py \
    --input-file motion.csv \
    --output-name motion_name \
    --input-fps 30 \
    --output-fps 50
```

### Deployment

**Export Policy:**
```python
# Policies are saved as .pt files (JIT traced)
policy = torch.jit.load("policy.pt")
obs = get_observation()
action = policy(obs)
```

**Sim-to-Real:**
- mjlab provides same API for sim and real deployment
- Export trained policy to onboard controller
- Control frequency: typically 50Hz (decimation=4 from 200Hz sim)

---

## Migration from Isaac Lab

### API Similarities (95%+ compatible)

**What's the Same:**
- Manager-based architecture (Observation, Reward, Action, etc.)
- Scene configuration with entities
- MDP term definitions (rewards, observations, terminations)
- Curriculum learning support
- Event-based randomization

**Key Differences:**

| Aspect | Isaac Lab | mjlab |
|--------|-----------|-------|
| **Config** | `@configclass` | `@dataclass` + `term()` helper |
| **Physics** | PhysX/Newton via Isaac Sim | MuJoCo Warp |
| **Assets** | USD/URDF | MJCF (MuJoCo XML) |
| **Scene Graph** | USD scene with `prim_path` | Direct MjSpec attachment |
| **Installation** | Heavy (Isaac Sim) | Lightweight (`pip install mjlab`) |

### Migration Steps

**1. Convert Assets:**
- Convert USD/URDF → MJCF using MuJoCo tools
- Or use pre-converted assets from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

**2. Update Imports:**
```python
# Isaac Lab
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.configclass import configclass

# mjlab
from mjlab.envs import ManagerBasedRlEnv
from dataclasses import dataclass
from mjlab.managers import term
```

**3. Update Config Syntax:**
```python
# Isaac Lab
@configclass
class RewardsCfg:
    velocity_tracking = RewTerm(func=mdp.track_vel, weight=1.0)

# mjlab
@dataclass
class RewardsCfg:
    velocity_tracking: RewTerm = term(RewTerm, func=mdp.track_vel, weight=1.0)
```

**4. Update Scene Config:**
```python
# Isaac Lab (USD-based)
scene = InteractiveSceneCfg(
    robot=ArticulationCfg(prim_path="/World/Robot", ...)
)

# mjlab (MjSpec-based)
scene = SceneCfg(
    entities={"robot": EntityCfg(...)}
)
```

**5. No Need to Change:**
- Reward functions (mostly compatible)
- Observation functions (mostly compatible)
- Training loop (same RSL-RL)

**See:** `docs/migration_guide.md` for complete examples

---

## Performance & Best Practices

### Performance Tips

**1. Maximize Parallelization:**
```bash
# Use all available GPU compute
--env.scene.num-envs 8192  # RTX 4090
--env.scene.num-envs 16384 # H100
```

**2. Optimize Physics:**
```python
# Parallel line search (significant speedup)
SimulationCfg(ls_parallel=True)

# Appropriate solver iterations
MujocoCfg(iterations=50)  # Reduce if stable

# Sparse Jacobian for articulated systems
MujocoCfg(jacobian="sparse")
```

**3. Minimize Rendering:**
```bash
# Training: disable rendering
MUJOCO_GL=egl uv run train ... --headless

# Eval: render only when needed
uv run play ... --num-envs 1
```

**4. Profile Bottlenecks:**
```python
# Use torch profiler
with torch.profiler.profile() as prof:
    env.step(action)
print(prof.key_averages().table())
```

### Hardware Recommendations

| GPU | Envs | Use Case |
|-----|------|----------|
| RTX 3090 | 4096 | Development |
| RTX 4090 | 8192 | Training |
| L40s | 12288 | Production |
| H100 | 16384+ | Large-scale |

**CPU:** Minimal impact (GPU does physics), 8+ cores sufficient

**RAM:** 32GB+ (large batches need more)

### Best Practices

**1. Start Small, Scale Up:**
```bash
# Debug with few envs
--env.scene.num-envs 32

# Train at scale
--env.scene.num-envs 4096
```

**2. Use Curriculum Learning:**
```python
curriculum = CurriculumManagerCfg(
    terrain_levels=CurriculumTermCfg(
        func=mdp.terrain_levels_vel,
    )
)
```

**3. Domain Randomization:**
```python
# Randomize physics for sim-to-real
events = EventManagerCfg(
    physics_material=EventTermCfg(...),
    add_base_mass=EventTermCfg(...),
)
```

**4. Monitor Training:**
- Watch episode returns in WandB
- Check reward term breakdown (identify issues)
- Log videos periodically (verify behavior)

---

## Limitations & Roadmap

### Current Limitations (Beta)

**Missing Features:**
- ❌ Camera/pixel rendering (vision-based RL)
- ❌ Soft body / deformables
- ❌ Fluid simulation
- ❌ Built-in robot arm examples
- ❌ Windows support (untested)

**Known Issues:**
- Breaking API changes possible (beta status)
- Limited asset zoo (2 robots)
- No photorealistic rendering (MuJoCo only)

**See:** [Issue #100](https://github.com/mujocolab/mjlab/issues/100) for stable release checklist

### Workarounds

**Vision-based RL:**
- Wait for MuJoCo Warp camera support (in development)
- Or use CPU MuJoCo rendering (slow)

**Photorealistic Rendering:**
- Use Isaac Sim for final visualization
- Export policy and deploy in IsaacLab for rendering

**Robot Arms:**
- Use this guide to add your own (architecturally supported!)
- Check MuJoCo Menagerie for pre-converted arms

### Roadmap

**Stable Release (v1.0):**
- ✅ Core API stabilization
- ⏳ Camera rendering support
- ⏳ Expanded robot library (separate repo)
- ⏳ Windows support
- ⏳ Complete documentation

**Future:**
- Advanced visualization (Isaac Sim integration?)
- Multi-GPU training
- Imitation learning utilities
- More terrain types

---

## Quick Reference

### Installation

```bash
# From source (recommended)
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
uv run demo

# From PyPI (beta snapshot)
uv add mjlab "mujoco-warp @ git+https://github.com/..."
```

### Common Commands

```bash
# List available tasks
uv run python -m mjlab.scripts.list_envs

# Train
MUJOCO_GL=egl uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --env.scene.num-envs 4096

# Play
uv run play --task Mjlab-Velocity-Flat-Unitree-G1-Play \
    --wandb-run-path org/project/run-id

# Test
make test

# Format
make format
```

### Key Files to Read

1. **Entity System:** `src/mjlab/entity/entity.py`
2. **Environment Base:** `src/mjlab/envs/manager_based_rl_env.py`
3. **Managers:** `src/mjlab/managers/*.py`
4. **Example Task:** `src/mjlab/tasks/velocity/config/g1/flat_env_cfg.py`
5. **Rewards Library:** `src/mjlab/envs/mdp/rewards.py`

### Community

- **Issues:** https://github.com/mujocolab/mjlab/issues
- **Discussions:** https://github.com/mujocolab/mjlab/discussions
- **Docs:**  (if available)

---

## Conclusion

mjlab provides a **lean, fast, MuJoCo-native alternative** to Isaac Lab for RL robotics research. It's ideal if you:

- Want Isaac Lab's proven API without Omniverse overhead
- Need fast iteration and debugging (standard Python)
- Prefer MuJoCo's physics and ecosystem
- Are building rigid-body RL agents (locomotion, manipulation)

**Robot arms are architecturally supported** - you just need to bring your own MJCF model and configure it following this guide.

For questions, see the FAQ or open a discussion on GitHub!
