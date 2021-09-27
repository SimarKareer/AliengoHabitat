import pinocchio
import numpy as np
from os import path as osp

EPS = 1e-2
IT_MAX = 10000
DT = 1e-2
DAMP = 1e-12

class IkSolver():

    def __init__(self, urdf_path):

        # Load the urdf model
        assert osp.isfile(urdf_path), (
            f'{urdf_path} does not exist; please pass in absolute '
            'path of your aliengo.urdf'
        )

        # Create data required by the algorithms
        self.model, self.collision_model, self.visual_model = (
            pinocchio.buildModelsFromUrdf(urdf_path, '')
        )
        self.data, self.collision_data, self.visual_data = (
            pinocchio.createDatas(self.model, self.collision_model, self.visual_model)
        )

    def inverse_kinematics(self, q, link_name, ee_pos):
        """

        :param q: Current joint configuration
        :param link_name: Name of end effector link (according to urdf)
        :param ee_pos: Desired 3D coordinate of end effector
        :return: New joint configuration
        """
        ee_frame_idx = self.model.getBodyId(link_name)
        oMdes = pinocchio.SE3(np.eye(3), ee_pos)  # rotation doesn't matter
        for _ in range(IT_MAX):
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacement(self.model, self.data, ee_frame_idx)
            dMi = oMdes.actInv(self.data.oMf[ee_frame_idx])
            err = pinocchio.log(dMi).vector[:3]

            if np.linalg.norm(err) < EPS:
                print("Pinnochio joints: ", q, "\n")
                return q

            J = pinocchio.computeFrameJacobian(
                self.model, self.data, q, ee_frame_idx, pinocchio.LOCAL
            )[:3, :]

            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + DAMP * np.eye(3), err))
            q = pinocchio.integrate(self.model, q, v * DT)

        print('IK WAS UNSUCCESSFUL!')
        return q

if __name__ == '__main__':
    from pinocchio.visualize import MeshcatVisualizer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('urdf_path', help='path to urdf of AlienGo')
    urdf_path = parser.parse_args().urdf_path

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

    ik_solver = IkSolver(urdf_path)

    # Bend knees
    q = pinocchio.neutral(ik_solver.model)
    for i in range(3, 16, 4):
        q[i] = np.deg2rad(-90)
    pinocchio.forwardKinematics(ik_solver.model, ik_solver.data, q)

    viz = MeshcatVisualizer(
        ik_solver.model, ik_solver.collision_model, ik_solver.visual_model
    )
    viz.initViewer()
    viz.loadViewerModel()
    viz.display(q)

    input("Refresh web page. Enter to continue.")

    # Raise and lower each foot one at a time, four times each
    neg = 1
    for link_name, rest in rest_positions.items():
        for _ in range(4):
            for _ in range(30):
                rest -= np.array([0., 0., -0.01*neg])
                q = ik_solver.inverse_kinematics(q, link_name, rest)
                viz.display(q)
            neg *= -1

    # Slide all feet horizontally simultaneously, four times
    for _ in range(4):
        for _ in range(30):
            for link_name, rest in rest_positions.items():
                rest -= np.array([-0.01*neg, 0.0, 0.0])

                # Feet need to be lifted 0.1m to allow for more horz. motion
                q = ik_solver.inverse_kinematics(
                    q, link_name, rest + np.array([0.0, 0.0, 0.1])
                )
            viz.display(q)
        neg *= -1