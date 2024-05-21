from omni.isaac.kit import SimulationApp

def init_simulation_app():
    # launch the simulator
    config = {"headless": True, "anti_aliasing": 1}
    simulation_app = SimulationApp(config)
    # simulation_app = SimulationApp(config)
    return simulation_app