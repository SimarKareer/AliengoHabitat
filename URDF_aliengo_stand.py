# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# [setup]

import os

import magnum as mn
import matplotlib.pyplot as plt
import numpy as np

import habitat_sim

# import habitat_sim.utils.common as ut
import habitat_sim.utils.viz_utils as vut
from habitat_sim.physics import JointMotorSettings

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../../data")
output_path = os.path.join(dir_path, "URDF_robotics_tutorial_output/")

class IkHelper:
    def __init__(self, arm_start):
        self._arm_start = arm_start
        self._arm_len = 7

    def setup_sim(self):
        self.pc_id = p.connect(p.DIRECT)

        self.robo_id = p.loadURDF(
                "./orp/robots/opt_fetch/robots/fetch_onlyarm.urdf",
                basePosition=[0, 0, 0],
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=self.pc_id
                )

        p.setGravity(0, 0, -9.81, physicsClientId=self.pc_id)
        JOINT_DAMPING = 0.5
        self.pb_link_idx = 7

        for link_idx in range(15):
            p.changeDynamics(
                    self.robo_id,
                    link_idx,
                    linearDamping=0.0,
                    angularDamping=0.0,
                    jointDamping=JOINT_DAMPING,
                    physicsClientId=self.pc_id
                    )
            p.changeDynamics(self.robo_id, link_idx, maxJointVelocity=200,
                    physicsClientId=self.pc_id)

    def set_arm_state(self, joint_pos, joint_vel=None):
        if joint_vel is None:
            joint_vel = np.zeros((len(joint_pos),))
        for i in range(7):
            p.resetJointState(self.robo_id, i, joint_pos[i],
                    joint_vel[i], physicsClientId=self.pc_id)

    def calc_fk(self, js):
        self.set_arm_state(js, np.zeros(js.shape))
        ls = p.getLinkState(self.robo_id, self.pb_link_idx,
                computeForwardKinematics=1, physicsClientId=self.pc_id)
        world_ee = ls[4]
        return world_ee

    def get_joint_limits(self):
        lower = []
        upper = []
        for joint_i in range(self._arm_len):
            ret = p.getJointInfo(self.robo_id, joint_i, physicsClientId=self.pc_id)
            lower.append(ret[8])
            if ret[9] == -1:
                upper.append(2*np.pi)
            else:
                upper.append(ret[9])
        return np.array(lower), np.array(upper)

    def calc_ik(self, targ_ee):
        """
        targ_ee is in ROBOT COORDINATE FRAME NOT IN EE COORDINATE FRAME
        """
        js = p.calculateInverseKinematics(self.robo_id, self.pb_link_idx,
                targ_ee, physicsClientId=self.pc_id)
        return js[:self._arm_len]

def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.7, 1.0]
    # agent_state.position = [-0.15, -1.6, 1.0]
    agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
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
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())

    return observations


def place_robot_from_agent(sim, robot_id, angle_correction=-1.56, local_base_pos=None):
    if local_base_pos is None:
        local_base_pos = np.array([0.0, -0.0, -2.0]) #-0.65
    # place the robot root state relative to the agent
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
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
    # create the simulator
    cfg = make_configuration()
    with habitat_sim.Simulator(cfg) as sim:
        place_agent(sim)
        observations = []

        # load a URDF file
        robot_file = urdf_files["aliengo"]
        ao_mgr = sim.get_articulated_object_manager()
        robot_id = ao_mgr.add_articulated_object_from_urdf(
            robot_file, fixed_base=False
        )

        # place the robot root state relative to the agent
        place_robot_from_agent(sim, robot_id)

        # set a better initial joint state for the aliengo
        if robot_file == urdf_files["aliengo"]:
            pose = robot_id.joint_positions
            print(pose)
            calfDofs = [2, 5, 8, 11]
            for dof in calfDofs:
                pose[dof] = -2.3 #second joint
                pose[dof - 1] = 1.3 #first joint
                # also set a thigh
            robot_id.joint_positions = pose
        print(robot_id.joint_positions)

        # simulate
        observations += simulate(sim, dt=1.5, get_frames=make_video)
        print("POST SIM: ", robot_id.joint_positions)

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

        jms = JointMotorSettings(
            0.0,  # position_target
            0.01,  # position_gain
            0.0,  # velocity_target
            0.5,  # velocity_gain
            0.1,  # max_impulse
        )
        jmsBody1 = JointMotorSettings( #close body joint
            1.0,  # position_target -2.3
            0.01,  # position_gain
            0.0,  # velocity_target
            0.5,  # velocity_gain
            .1,  # max_impulse
        )
        jmsBody2 = JointMotorSettings( #far body joint
            1.0,  # position_target -2.3
            0.01,  # position_gain
            0.0,  # velocity_target
            0.5,  # velocity_gain
            0.1,  # max_impulse
        )

        jmsKnee1 = JointMotorSettings( #far knee joint
            -1.8,  # position_target 1.4
            0.02,  # position_gain 0.09
            0.0,  # velocity_target
            0.5,  # velocity_gain 0.02
            0.1, # max_impulse
        )

        jmsKnee2 = JointMotorSettings( #close knee joint
            -1.8,  # position_target 1.4
            0.03,  # position_gain 0.09
            0.0,  # velocity_target
            0.4,  # velocity_gain 0.02
            0.4, # max_impulse
        )
        robot_id.create_all_motors(jms)
        # lowerJoint = [a-1 for a in calfDofs]

        #0 back left rotator
        #1: na
        #2: back left knee
        #5: back right knee
        #6: weird front left leg thing
        #8: na
        #9: leg rotation
        #11: na
        #12: nope
        #13: seems like back left body joint
        #14: actually back left knee?
        #15: nah
        #16: back right body joint
        #17: knee
        #18: nah
        #20: knee

        for joint in [19, 22]:
            robot_id.update_joint_motor(joint, jmsBody1)

        for joint in [13, 16]:
            robot_id.update_joint_motor(joint, jmsBody2)

        for joint in [2, 5]:
            robot_id.update_joint_motor(joint, jmsKnee1)

        for joint in [20, 23]:
            robot_id.update_joint_motor(joint, jmsKnee2)
        # for joint in [7, 10]:
        #     robot_id.update_joint_motor(joint, jms4)
        
        # for joint in calfDofs + lowerJoint:
        #     print(join)
        # for joint not in calf
        #     robot_id.update_joint_motor(joint-1, jms3)
        # robot_id._update_motor_settings_cache()
        sim.remove_object(cube_id)
        for link_id in robot_id.link_object_ids.values():
            # sim.set_translation(
            #     robot_id.get_link_scene_node(
            #         link_id
            #     ).translation,
            #     cube_id,
            # )
            # sim.set_rotation(
            #     robot_id.get_link_scene_node(
            #         link_id
            #     ).rotation,
            #     cube_id,
            # )
            observations += simulate(sim, dt=0.5, get_frames=make_video)
        # sim.remove_object(cube_id)

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