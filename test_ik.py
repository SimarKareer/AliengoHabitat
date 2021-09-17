import pinocchio
import numpy as np

from numpy.linalg import norm, solve

urdf_filename = "/Users/naokiyokoyama/gt/aliencat/habitat-sim/data/aliengo/urdf/aliengo.urdf"
# Load the urdf model
model = pinocchio.buildModelFromUrdf(urdf_filename)
print('model name: ' + model.name)

# Create data required by the algorithms
data = model.createData()

q = pinocchio.randomConfiguration(model)

JOINT_ID = 3
rot_mat = np.eye(3)
rot_mat[0,0] = 0.0
rot_mat[2,2] = 0.0
oMdes = pinocchio.SE3(rot_mat, np.array([0.24, 0.14, 0.]))

eps = 1e-2
IT_MAX = 10000
DT = 1e-3
damp = 1e-12

i = 0
while True:
    pinocchio.forwardKinematics(model, data, q)
    dMi = oMdes.actInv(data.oMi[JOINT_ID])
    err = pinocchio.log(dMi).vector
    err = err[:3]

    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break

    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
    J = J[:3, :]

    v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(3), err))
    q = pinocchio.integrate(model, q, v * DT)
    if not i % 10:
        print('%d: error = %s' % (i, norm(err.T)))
    i += 1

if success:
    print("Convergence achieved!")
else:
    print(
        "\nWarning: the iterative algorithm has not reached"
        "convergence to the desired precision"
    )

print('\nresult: %s' % q.flatten().tolist())
print('\nfinal error: %s' % err.T)