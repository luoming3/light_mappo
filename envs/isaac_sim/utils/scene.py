from omni.isaac.cloner import GridCloner

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

from pxr import UsdGeom
import torch

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

def set_up_scene(env_num=1):
    world = get_world()
    scene = world.scene
    scene.clear(True)

    add_reference_to_stage("/home/user/Desktop/car_jetbot.usd", "/World/envs/env_0/car")

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