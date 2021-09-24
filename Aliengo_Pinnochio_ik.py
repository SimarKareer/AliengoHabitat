# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# [setup]

import os

import magnum as mn
import matplotlib.pyplot as plt
import numpy as np

import habitat_sim

import pybullet as p

# import habitat_sim.utils.common as ut
import habitat_sim.utils.viz_utils as vut
from habitat_sim.physics import JointMotorSettings
from aliengo_ik_solver import AliengoIkSolver

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../habitat-sim/data/")
output_path = os.path.join(dir_path, "URDF_robotics_tutorial_output/")

def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.7, 1.0]
    # agent_state.position = [-0.15, -0.7, 1.0]
    # agent_state.position = [-0.15, -1.6, 1.0]

    agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = os.path.join(data_path, "scene_datasets/habitat-test-scenes/apartment_1.glb")
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [540, 720]
    sensors = {
        "rgba_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_specs.append(sensor_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    # while sim.get_world_time() < start_time + dt:
    #     sim.step_physics(1.0 / 60.0)
    #     if get_frames:
    #         observations.append(sim.get_sensor_observations())
    num_steps = int(dt // (1/60))
    if get_frames:
        for _ in range(num_steps):
            observations.append(sim.get_sensor_observations())

    return observations


def place_robot_from_agent(sim, robot_id, angle_correction=-1.56, local_base_pos=None):
    if local_base_pos is None:
        local_base_pos = np.array([0.0, -0.0, -2.0]) #-0.65
    # place the robot root state relative to the agent
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    # angle_correction=0
    base_transform = mn.Matrix4.rotation(
        mn.Rad(angle_correction), mn.Vector3(1.0, 0, 0)
    )
    base_transform.translation = agent_transform.transform_point(local_base_pos)
    robot_id.transformation = base_transform

urdf_files = {
    "aliengo": os.path.join(data_path, "URDF_demo_assets/aliengo/urdf/aliengo.urdf"),
}



# This is wrapped such that it can be added to a unit test
def main(make_video=True, show_video=True):

    # [initialize]
    ik = AliengoIkSolver(urdf_files["aliengo"])
    # print("IK Solved: ", ik.calc_ik(np.array([0, 0, 0])))

    # create the simulator
    cfg = make_configuration()
    with habitat_sim.Simulator(cfg) as sim:
        place_agent(sim)
        observations = []

        # load a URDF file
        robot_file = urdf_files["aliengo"]
        ao_mgr = sim.get_articulated_object_manager()
        robot_id = ao_mgr.add_articulated_object_from_urdf(
            robot_file, fixed_base=True
        )

        # sim.set_gravity([0., 0., 0.])
        # place the robot root state relative to the agent
        place_robot_from_agent(sim, robot_id)
        print("Start position: ", robot_id.rigid_state.translation)
        print("Start Rotation: ", robot_id.rigid_state.rotation)

        robot_id.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        # set a better initial joint state for the aliengo
        if robot_file == urdf_files["aliengo"]:
            pose = robot_id.joint_positions
            print("0 POSE: ", pose)
            calfDofs = [2, 5, 8, 11]
            for dof in calfDofs:
                pose[dof] = np.deg2rad(-75) #second joint
                pose[dof - 1] = np.deg2rad(45) #first joint
            # pose[6] = -1
            robot_id.joint_positions = pose
            print("joint_positions after moving stuff", robot_id.joint_positions)

            rest_positions = {
                k: np.array([0.2399, 0.134, -0.4]) * reflect
                for k, reflect in zip(
                    ['FL_foot', 'FR_foot', 'RR_foot', 'RL_foot'],
                    [
                        np.array([1.0, 1.0, 1.0]), np.array([1.0, -1.0, 1.0]),
                        np.array([-1.0, -1.0, 1.0]), np.array([-1.0, 1.0, 1.0]),
                    ],
                )
            }
            neg = 1
            for link_name, rest in rest_positions.items():
                for _ in range(4):
                    for _ in range(30):
                        rest -= np.array([0., 0., -0.01*neg])
                        # q = ik_solver.inverse_kinematics(q, link_name, rest)
                        robot_id.joint_positions = ik.inverse_kinematics(robot_id.joint_positions, link_name, rest)
                        observations += simulate(sim, dt=0.02, get_frames=make_video)
                    neg *= -1

        # print("POST SIM: ", robot_id.joint_positions)

        # reset the object state (sets dof positions/velocities/forces to 0, recomputes forward kinematics, udpate collision state)
        # robot_id.joint_positions = np.zeros(len(robot_id.joint_positions))
        # note: reset does not change the robot base state, do this manually
        # place_robot_from_agent(sim, robot_id)

        # disable gravity
        # sim.set_gravity([0., 0., 0.])
        robot_id.motion_type = habitat_sim.physics.MotionType.DYNAMIC

        # get rigid state of robot links and show proxy object at each link COM
        obj_mgr = sim.get_object_template_manager()
        cube_id = sim.add_object_by_handle(obj_mgr.get_template_handles("cube")[0])
        sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, cube_id)
        sim.set_object_is_collidable(False, cube_id)

        # robot_id._update_motor_settings_cache()

        sim.remove_object(cube_id)
        # for link_id in robot_id.link_object_ids.values():
        #     observations += simulate(sim, dt=0.5, get_frames=make_video)

        if make_video:
            vut.make_video(
                observations,
                "rgba_camera_1stperson",
                "color",
                output_path + "URDF_basics",
                open_vid=show_video,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    make_video = args.make_video

    if make_video and not os.path.exists(output_path):
        os.mkdir(output_path)

    main(make_video, show_video)