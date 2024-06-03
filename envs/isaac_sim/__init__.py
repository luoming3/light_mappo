from omni.isaac.kit import SimulationApp


def init_simulation_app(headless=True):
    # launch the simulator
    config = {"headless": headless, "anti_aliasing": 1}
    simulation_app = SimulationApp(config)
    # simulation_app = SimulationApp(config)

    # wait two frames so that stage starts loading
    simulation_app.update()
    simulation_app.update()

    print("Loading stage...")
    from omni.isaac.core.utils.stage import is_stage_loading

    while is_stage_loading():
        simulation_app.update()
    print("Loading Complete")
    return simulation_app