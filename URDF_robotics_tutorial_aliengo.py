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

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "../habitat-sim/data/")
output_path = os.path.join(dir_path, "URDF_robotics_tutorial_output/")

class IkHelper:
    def __init__(self, arm_start):
        self._arm_start = arm_start
        self._arm_len = 7

    """
    This function will create a dictionary to map joint and link indices from sim space to bullet space
    """
    def simToBulletLink(self, sim_robot_id):
        simMap = {}
        for link_id in sim_robot_id.link_object_ids.values():
            simMap[sim_robot_id.get_link_name(link_id)] = link_id
        print("SimLinkMap: ", simMap)
        
        bulletMap = {p.getBodyInfo(self.robo_id)[0].decode('UTF-8'):-1,}
        for link_id in range(p.getNumJoints(self.robo_id)):
            _name = p.getJointInfo(self.robo_id, link_id)[12].decode('UTF-8')
            bulletMap[_name] = link_id
        
        return self.joinDicts(simMap, bulletMap)

    def simToBulletJoint(self, sim_robot_id):
        # Sim map here won't work
        simMap = {'RR_hip_joint': 0, 'RR_thigh_joint':1, 'RR_calf_joint': 2, 'RL_hip_joint': 3, 'RL_thigh_joint':4, 'RL_calf_joint': 5, 'FR_hip_joint': 6, 'FR_thigh_joint':7, 'FR_calf_joint': 8, 'FL_hip_joint': 9, 'FL_thigh_joint':10, 'FL_calf_joint': 11}
        
        bulletMap = {'FR_hip_joint': 0, 'FR_thigh_joint':1, 'FR_calf_joint': 2, 'FL_hip_joint': 3, 'FL_thigh_joint':4, 'FL_calf_joint': 5, 'RR_hip_joint': 6, 'RR_thigh_joint':7, 'RR_calf_joint': 8, 'RL_hip_joint': 9, 'RL_thigh_joint':10, 'RL_calf_joint': 11}# fr, fl, rr, rl
        # for joint_id in range(p.getNumJoints(self.robo_id)):
        #     _name = p.getJointInfo(self.robo_id, joint_id)[1].decode('UTF-8')
        #     bulletMap[_name] = joint_id

        return self.joinDicts(simMap, bulletMap)
    
    def joinDicts(self, simMap, bulletMap):
        simToBullet = {}
        for (k, v) in simMap.items():
            simToBullet[v] = bulletMap[k]
        
        return simToBullet
    
    def setupMappings(self, sim_robot_id):
        self.simToBulletLink = self.simToBulletLink(sim_robot_id)
        self.bulletToSimLink = {v: u for (u, v) in self.simToBulletLink.items()}

        self.simToBulletJoint = self.simToBulletJoint(sim_robot_id)
        self.bulletToSimJoint = {v: u for (u, v) in self.simToBulletJoint.items()}

    def setup_sim(self):
        self.pc_id = p.connect(p.DIRECT)

        self.robo_id = p.loadURDF(
            os.path.join(data_path, "URDF_demo_assets/aliengo/urdf/aliengo.urdf"),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.pc_id
        )

        _link_name_to_index = {p.getBodyInfo(self.robo_id)[0].decode('UTF-8'):-1,}
                
        for _id in range(p.getNumJoints(self.robo_id)):
            _name = p.getJointInfo(self.robo_id, _id)[12].decode('UTF-8')
            _link_name_to_index[_name] = _id
        
        print(_link_name_to_index)
        # print("Type of pybullet p", type(p))
        # for jointNum in range(p.getNumJoints(self.robo_id)):
        #     joint = p.getJointInfo(self.robo_id, jointNum)
        #     print("Joint Index: ", joint[0])
        #     print("Joint Name: ", joint[1])
        #     print("Link Name: ", joint[12])
        #     print("Parent Link Index: ", joint[16])
        #     print("-"*20)

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

    def calc_ik(self, link_index, targ_ee):
        """
        targ_ee is in ROBOT COORDINATE FRAME NOT IN EE COORDINATE FRAME
        """
        link_index = self.simToBulletLink[link_index]
        js = p.calculateInverseKinematics(self.robo_id, link_index,
                targ_ee, physicsClientId=self.pc_id)

        # print("IK in bullet joint: ", js)
        
        simBasedIndex = np.zeros(len(js))
        for i in range(len(js)):
            simBasedIndex[self.bulletToSimJoint[i]] = js[i]
        return simBasedIndex

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
    ik = IkHelper(0)
    ik.setup_sim()
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
        ik.setupMappings(robot_id)
        print("Sim To Bullet Link", ik.simToBulletLink)
        print("Bullet to Sim Link", ik.bulletToSimLink)
        print("Sim To Bullet Joint", ik.simToBulletJoint)
        print("Bullet To Sim Joint", ik.bulletToSimJoint)
        print("CALC IK: ", ik.calc_ik(4, [0.5, 0.5, 0.5]))

        sim.set_gravity([0., 0., 0.])
        # place the robot root state relative to the agent
        place_robot_from_agent(sim, robot_id)

        # set a better initial joint state for the aliengo
        if robot_file == urdf_files["aliengo"]:
            pose = robot_id.joint_positions
            print(pose)
            calfDofs = [2, 5, 8, 11]
            robot_id.joint_positions = ik.calc_ik(8, [0.0, 0.0, 0.0]) + ik.calc_ik(4, [0.0, 0.0, 0.0])
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

        # robot_id._update_motor_settings_cache()

        sim.remove_object(cube_id)
        for link_id in robot_id.link_object_ids.values():
            observations += simulate(sim, dt=0.5, get_frames=make_video)

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