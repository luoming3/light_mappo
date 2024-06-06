import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.physx.scripts.utils as script_utils
import torch

from omni.isaac.cloner import GridCloner

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.kit.commands import execute

from pxr import UsdGeom, PhysxSchema, UsdPhysics

import os
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from envs.isaac_sim.utils.config import RobotCfg

cfg_cls = RobotCfg
cfg = cfg_cls()

rigid_props = cfg.rigid_props
articulation_props = cfg.articulation_props

_world = None


def get_world():
    global _world
    if World._world_initialized:
        return _world
    
    # GPU buffers
    sim_params = {
        "gpu_max_rigid_contact_count": 524288,
        "gpu_max_rigid_patch_count": 163840, # 33554432
        "gpu_found_lost_pairs_capacity": 4194304,
        "gpu_found_lost_aggregate_pairs_capacity": 33554432,
        "gpu_total_aggregate_pairs_capacity": 4194304,
        "gpu_max_soft_body_contacts": 1048576,
        "gpu_max_particle_contacts": 1048576,
        "gpu_heap_capacity": 67108864,
        "gpu_temp_buffer_capacity": 16777216,
        "gpu_max_num_partitions": 8,
    }

    _world = World(stage_units_in_meters=1.0, backend="torch", device="cuda:0", sim_params=sim_params)

    return _world

def set_up_new_scene(env_num=1, bot_num=4):
    world = get_world()
    assets_root_path = get_assets_root_path()
    assert assets_root_path is not None, "asserts root path is None"
    jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
    scene = world.scene
    scene.clear(True)

    base_env_path = "/World/envs"
    default_zero_env_path="/World/envs/env_0"
    prim_path = "/World/envs/env_0/car"

    if prim_utils.is_prim_path_valid(prim_path):
        raise RuntimeError(f"Duplicate prim at {prim_path}.")
    car = prim_utils.create_prim(
        prim_path,
        # translation=translation,
    )

    if bot_num == 4:
        payload_scale = (0.75, 0.5, 0.001)
    elif bot_num == 6:
        payload_scale = (1.0, 0.5, 0.001)
    payload = prim_utils.create_prim(
        prim_path=f"{prim_path}/payload",
        prim_type="Cube",
        translation=(0.0, 0.0, 0.4),
        scale=payload_scale,
    )

    script_utils.setRigidBody(payload, "convexHull", False)
    UsdPhysics.MassAPI.Apply(payload)
    payload.GetAttribute("physics:mass").Set(2.0)
    payload.GetAttribute("physics:collisionEnabled").Set(False)

    PhysxSchema.PhysxRigidBodyAPI(payload).GetLinearDampingAttr().Set(0.1)  # angular_damping=0.1,
    PhysxSchema.PhysxRigidBodyAPI(payload).GetAngularDampingAttr().Set(0.1)  # linear_damping=0.1

    # kit_utils.set_rigid_body_properties(
    #     payload.GetPath(),
    #     angular_damping=0.1,
    #     linear_damping=0.1
    # )

    if bot_num == 4:
        drone_translations = torch.tensor([
            [0.75, 0.5, 0],
            [0.75, -0.5, 0],
            [-0.75, -0.5, 0],
            [-0.75, 0.5, 0],
        ])
    elif bot_num == 6:
        drone_translations = torch.tensor([
            [1.0, 0.5, 0],
            [1.0, -0.5, 0],
            [0.0, 0.5, 0],
            [0.0, -0.5, 0],
            [-1.0, -0.5, 0],
            [-1.0, 0.5, 0],
        ])

    for i in range(bot_num):
        jetbot_prim_path = f"{prim_path}/jetbot_{i}"
        if prim_utils.is_prim_path_valid(jetbot_prim_path):
            raise RuntimeError(f"Duplicate prim at {jetbot_prim_path}.")
        # prim = self._create_prim(prim_path, translation, orientation)
        jetbot_prim = prim_utils.create_prim(
                prim_path=jetbot_prim_path,
                usd_path=jetbot_asset_path,
                translation=drone_translations[i],
            )
        # drone_prim = self.drone.spawn(
        #     translations=drone_translations[i],
        #     prim_paths=[f"{prim_path}/{self.drone.name.lower()}_{i}"],
        # )[0]
        execute(
            "UnapplyAPISchema",
            api=UsdPhysics.ArticulationRootAPI,
            prim=jetbot_prim,
        )
        execute(
            "UnapplyAPISchema",
            api=PhysxSchema.PhysxArticulationAPI,
            prim=jetbot_prim,
        )

        stage = stage_utils.get_current_stage()
        chassis = prim_utils.get_prim_at_path(f"{prim_path}/jetbot_{i}/chassis")
        joint = script_utils.createJoint(stage, "Revolute", payload, chassis)
        UsdPhysics.DriveAPI.Apply(joint, "linear")
        joint.GetAttribute("physics:axis").Set("Z")

    UsdPhysics.ArticulationRootAPI.Apply(car)
    PhysxSchema.PhysxArticulationAPI.Apply(car)

    physx_articulation_api = PhysxSchema.PhysxArticulationAPI(car)
    # set enable/disable rigid body API
    physx_articulation_api.GetEnabledSelfCollisionsAttr().Set(articulation_props.enable_self_collisions)
    # set solver position iteration
    physx_articulation_api.GetSolverPositionIterationCountAttr().Set(articulation_props.solver_position_iteration_count)
    # set solver velocity iteration
    physx_articulation_api.GetSolverVelocityIterationCountAttr().Set(articulation_props.solver_velocity_iteration_count)

    # create a GridCloner instance
    cloner = GridCloner(spacing=8)
    cloner.define_base_env(base_env_path)

    UsdGeom.Xform.Define(get_current_stage(), base_env_path)

    collision_filter_global_paths = []
    ground_plane_path = "/World/defaultGroundPlane"
    collision_filter_global_paths.append(ground_plane_path)

    scene.add_default_ground_plane(prim_path=ground_plane_path)

    # generate 4 paths that begin with "/World/Cube" - path will be appended with _{index}
    prim_paths = cloner.generate_paths("/World/envs/env", env_num)

    env_pos = cloner.clone(
        source_prim_path=default_zero_env_path, prim_paths=prim_paths, replicate_physics=True, copy_from_source=False
    )
    env_pos = torch.tensor(env_pos, dtype=torch.float32)

    cloner.filter_collisions(
        world.get_physics_context().prim_path,
        "/World/Collisions",
        prim_paths,
        collision_filter_global_paths
    )

    car_view = ArticulationView(
        prim_paths_expr="/World/envs/.*/car",
        name="car_view",
        positions=env_pos,
        reset_xform_properties=False)
    jetbot_view = RigidPrimView(
        prim_paths_expr="/World/envs/env_*/car/jetbot_.*/chassis",
        name="jetbot_chassis_view",
        reset_xform_properties=False)

    # add car ArticulationView for control
    scene.add(car_view)
    scene.add(jetbot_view)

    world.reset()

    return world



def set_up_scene(env_num=1):
    world = get_world()
    scene = world.scene
    scene.clear(True)

    add_reference_to_stage("C:/Users/wenze.li/AppData/Local/ov/pkg/isaac_sim-2023.1.1/light_mappo/envs/isaac_sim/utils/car_jetbot.usd", "/World/envs/env_0/car")

    # create our base environment with one cube
    base_env_path = "/World/envs"

    # create a GridCloner instance
    cloner = GridCloner(spacing=8)
    cloner.define_base_env(base_env_path)

    default_zero_env_path="/World/envs/env_0"
    UsdGeom.Xform.Define(get_current_stage(), base_env_path)

    collision_filter_global_paths = []
    ground_plane_path = "/World/defaultGroundPlane"
    collision_filter_global_paths.append(ground_plane_path)

    scene.add_default_ground_plane(prim_path=ground_plane_path)

    # generate 4 paths that begin with "/World/Cube" - path will be appended with _{index}
    prim_paths = cloner.generate_paths("/World/envs/env", env_num)

    env_pos = cloner.clone(
        source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=True, copy_from_source=False
    )
    env_pos = torch.tensor(env_pos, dtype=torch.float32)

    cloner.filter_collisions(
        world.get_physics_context().prim_path,
        "/World/Collisions",
        prim_paths,
        collision_filter_global_paths
    )

    cars = ArticulationView(
        prim_paths_expr="/World/envs/.*/car",
        name="car_view",
        positions=env_pos,
        reset_xform_properties=False
    )
    jetbot_view = RigidPrimView(
        prim_paths_expr="/World/envs/env_*/car/jetbot_.*/chassis",
        name="jetbot_chassis_view",
        reset_xform_properties=False
    )

    # add car ArticulationView for control
    scene.add(cars)
    scene.add(jetbot_view)

    # reset car env
    world.reset()

    return scene