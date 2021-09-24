from ik_solver import IkSolver
import numpy as np

class AliengoIkSolver(IkSolver):
    def joinDicts(self, simMap, bulletMap):
        simToBullet = {}
        for (k, v) in simMap.items():
            simToBullet[v] = bulletMap[k]
        
        return simToBullet
    
    def habitatToPinnochio(self, habitatJoints):
        pinnochioJoints = np.ones(16)
        for i in range(len(habitatJoints)):
            pinnochioJoints[self.habitatToPinnochioD[i]] = habitatJoints[i]

        return pinnochioJoints
    
    def pinnochioToHabitat(self, pinnochioJoints):
        habitatJoints = np.zeros(12)
        for i in range(len(pinnochioJoints)):
            if i in self.pinnochioToHabitatD:
                habitatJoints[self.pinnochioToHabitatD[i]] = pinnochioJoints[i]
        
        return habitatJoints

    def __init__(self, urdf_path):
        super().__init__(urdf_path)
        habitatJointMap = {
            'FL_hip_joint': 0, 'FL_thigh_joint':1, 'FL_calf_joint': 2, 
            'FR_hip_joint': 3, 'FR_thigh_joint':4, 'FR_calf_joint': 5, 
            'RL_hip_joint': 6, 'RL_thigh_joint':7, 'RL_calf_joint': 8, 
            'RR_hip_joint': 9, 'RR_thigh_joint':10, 'RR_calf_joint': 11
        }
        pinnochioJointMap = {
            "FL_hip_joint": 0, "FL_scale": 1, "FL_thigh_joint": 2, "FL_calf_joint": 3,
            "FR_hip_joint": 4, "FR_scale": 5, "FR_thigh_joint": 6, "FR_calf_joint": 7,
            "RL_hip_joint": 8, "RL_scale": 9, "RL_thigh_joint": 10, "RL_calf_joint": 11,
            "RR_hip_joint": 12, "RR_scale": 13, "RR_thigh_joint": 14, "RR_calf_joint": 15
        }

        self.habitatToPinnochioD = self.joinDicts(habitatJointMap, pinnochioJointMap)
        self.pinnochioToHabitatD = {v: u for (u, v) in self.habitatToPinnochioD.items()}

    
    def inverse_kinematics(self, joint_pos, link_name, ee_pos):
        joint_pos = self.habitatToPinnochio(joint_pos)
        # print("Got new joint_pos", joint_pos)
        pinnochioJoints = super().inverse_kinematics(joint_pos, link_name, ee_pos)
        # print("ran IK: ", pinnochioJoints)

        return self.pinnochioToHabitat(pinnochioJoints)


