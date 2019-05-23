import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as la
from scipy.linalg import expm


def Rot3d(rpy):
    """
    Transform a roll, pitch, yaw into a rotation matrix
    ASSUMES body/relative rotations i.e. "rzyx"
    :param rpy: roll, pitch, yaw
    :return: rotation matrix
    """
    cx = np.cos(rpy[0])
    sx = np.sin(rpy[0])
    cy = np.cos(rpy[1])
    sy = np.sin(rpy[1])
    cz = np.cos(rpy[2])
    sz = np.sin(rpy[2])
    rot = np.array((
        [cx * cy,  -sx * cz + sy * sz * cx,   sx * sz + sy * cx * cz],
        [sx * cy,   cx * cz + sx * sy * sz,  -sz * cx + sx * sy * cz],
        [-sy,        sz * cy,                  cy * cz]))

    return rot


def plotFrame(ax, pts, alpha=.3, tip='k'):
    # plots a RGB frame thing
    colors = ['r', 'g', 'b']
    for pt, color in zip(pts.T, colors):
        ax.plot([0, pt[0]],
                [0, pt[1]],
                [0, pt[2]],
                color=color, alpha=alpha)
        ax.scatter(*pt, color=tip)  # put balls on the tips


class so3Grp(object):
    # a member, X, of the lie group so3
    def __init__(self, X):
        # make a member of the group
        self.X = X
        # asset we hold the group constraint
        # for numerics we have  || X'X -I || < 1e-4
        assert la.norm(np.dot(X.T, X) - np.eye(3)) < 1e-4

    def __repr__(self):
        # prints the group member
        return self.X.__repr__()

    def inv(self):
        # reurns the inverse group member z* such that z*z=I
        return so3Grp((self.X).T)  # cuz orthogonal matrix

    def mult(self, Z):
        # return a member from group_opp(self, z)
        Y = self.X.dot(Z.X)
        return so3Grp(Y)

    def step(self, v):
        # v is a member of the lie algebra (tanget space at the identy)
        # we want make a step of v at self
        v_local = v.adj(self)  # convert step in global frame to local frame
        v_group = v_local.exp()  # map to manifold
        return self.mult(v_group)

    def plot(self, ax, frame, alpha=.3, tip='k'):
        # plot a frame at the rotaiton
        frame_R = self.X.dot(frame)
        plotFrame(ax, frame_R, alpha, tip)


class so3Alg(object):
    # a member of the lie algabra of so3
    def __init__(self, w):
        # make the member
        # w is in R3 and is a preterbation about each axis
        self.w = w  # vector space

        # W is the member of the algebra and is a skew sym matrix
        self.W = np.array([[0.,   -w[2], w[1]],
                           [w[2],  0,  -w[0]],
                           [-w[1], w[0],  0]])

    def __repr__(self):
        # print the member
        return self.W.__repr__()

    def vee(self):
        # returns a memeber of the associated VECTOR space
        return self.w

    def exp(self):
        # map the lie algebra member to the corresponding lie group member
        X = expm(self.W)
        return so3Grp(X)

    def adj(self, Z):
        # Z is a member of the partent lie group
        # Z * self.W * Z.inv
        # sorry for the overload of matrix opps, blame numpy
        Z_inv = Z.inv()
        W_global = np.dot(Z_inv.X, self.W.dot(Z.X))
        # print(W_global)  # need to extract w
        w_global = [W_global[2, 1], W_global[0, 2], W_global[1, 0]]

        return so3Alg(w_global)


if __name__ == '__main__':

    Rtest1 = Rot3d([.1, .2, .2])  # about z, about y, about x
    Rtest2 = Rot3d([.0, .2, 0])
    frame = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])

    A = so3Grp(Rtest1)
    B = so3Grp(Rtest2)
    C = A.mult(B)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # c.plot(ax, frame, alpha=1)
    plotFrame(ax, frame, alpha=.5)

    v = so3Alg([0, .8, 0])  # algebra
    V = v.exp()  # group
    # print(vx)
    # print(v.W)
    AV_world = A.step(v)
    AV_body = A.mult(V)

    # should be the same as AV_world
    VA_world = V.mult(A)

    A.plot(ax, frame, tip='r')
    # V.plot(ax, frame, tip='b')

    # plot
    AV_body.plot(ax, frame, tip='g')
    AV_world.plot(ax, frame, tip='c')
    VA_world.plot(ax, frame, tip='m')

    print(AV_world)
    print(VA_world)


    # plotted for style

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
