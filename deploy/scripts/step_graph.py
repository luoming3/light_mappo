import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

log_path = Path(os.path.join(os.path.dirname(__file__), "../logs"))
sub_folder = ""
log_file_path_list = list(log_path.glob(f"**/{sub_folder}/**/step_record_*_*.npy"))

log_id_set = list(set(map(lambda x: x.stem.split("_")[-1], log_file_path_list)))
log_id_set.sort()

for log_id in log_id_set:
    log_id_path_list = list(log_path.glob(f"**/step_record_*_{log_id}.npy"))
    log_id_path_list.sort()

    car_center_list = []
    guide_point_list = []
    velocity_list = []
    orientation_list = []
    force_list = []

    for file_path in log_id_path_list:
        records = np.load(file_path, allow_pickle=True)
        car_center_list.append([[x, y] for x, y in records[:, 0]])
        guide_point_list.append([[x, y] for x, y in records[:, 1]])
        velocity_list.append([[x, y] for x, y in records[:, 2]])
        orientation_list.append([x for x in records[:, 3]])
        force_list.append([[x, y] for x, y in records[:, 4]])

    min_step_count = min(map(lambda x: len(x), car_center_list))
    car_center = np.array([each[:min_step_count] for each in car_center_list])
    guide_point = np.array([each[:min_step_count] for each in guide_point_list])
    velocity = np.array([each[:min_step_count] for each in velocity_list])
    orientation = np.array([each[:min_step_count] for each in orientation_list])
    force = np.array([each[:min_step_count] for each in force_list])
    step_list = list(range(1, min_step_count + 1))

    plt_legend = list(map(lambda x: x.stem.split("_")[2], log_id_path_list))
    plt_legend.sort()
    # car center
    plt.figure("car_center")
    plt.subplot(211)
    plt.plot(step_list, car_center[:, :, 0].T, ".", ms=1)
    plt.ylabel("car_center_x")
    plt.legend(plt_legend)

    plt.subplot(212)
    plt.plot(step_list, car_center[:, :, 1].T, ".", ms=1)
    plt.xlabel("step")
    plt.ylabel("car_center_y")
    plt.legend(plt_legend)

    # guide point
    plt.figure("guide_point")
    plt.subplot(211)
    plt.plot(step_list, guide_point[:, :, 0].T, ".", ms=1)
    plt.ylabel("guide_point_x")
    plt.legend(plt_legend)

    plt.subplot(212)
    plt.plot(step_list, guide_point[:, :, 1].T, ".", ms=1)
    plt.xlabel("step")
    plt.ylabel("guide_point_y")
    plt.legend(plt_legend)

    # velocity
    plt.figure("velocity")
    plt.subplot(211)
    plt.plot(step_list, velocity[:, :, 0].T, ".", ms=1)
    plt.ylabel("velocity_x")
    plt.legend(plt_legend)

    plt.subplot(212)
    plt.plot(step_list, velocity[:, :, 1].T, ".", ms=1)
    plt.xlabel("step")
    plt.ylabel("velocity_y")
    plt.legend(plt_legend)

    # orientation
    plt.figure("orientation")
    plt.plot(step_list, orientation.T, ".", ms=1)
    plt.ylabel("orientation")
    plt.legend(plt_legend)

    # force
    plt.figure("force")
    plt.subplot(211)
    plt.plot(step_list, force[:, :, 0].T, ".", ms=1)
    plt.ylabel("force_x")
    plt.legend(plt_legend)

    plt.subplot(212)
    plt.plot(step_list, force[:, :, 1].T, ".", ms=1)
    plt.xlabel("step")
    plt.ylabel("force_y")
    plt.legend(plt_legend)

    plt.show()
