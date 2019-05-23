import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


class complxGrp(object):
    # a member, z, of the lie group of unit complex numbers S1
    def __init__(self, x, y):
        # make a member of the group
        self.x = x
        self.y = y
        # asset we hold the group constraint
        # for numerics we have  | ||z|| -1 | < 1e-4
        assert abs(la.norm([self.x, self.y]) - 1) < 1e-4

    def __repr__(self):
        # prints the group member
        return '{:2.4f} + {:2.4f} i'.format(self.x, self.y)

    def inv(self):
        # reurns the inverse group member z* such that z*z=1
        return complxGrp(self.x, -self.y)  # cuz conjugate

    def mult(self, z):
        # return a complx from group_opp(self, z)
        new_x = self.x*z.x - self.y*z.y  # real part
        new_y = self.x*z.y + self.y*z.x  # im part
        return complxGrp(new_x, new_y)

    def step(self, v):
        # v is a member of the lie algebra (tanget space at the identy)
        # we want make a step of v at self
        v_local = v.adj(self)  # convert step in global frame to local frame
        v_group = v_local.exp()  # map to manifold
        return self.mult(v_group)

    def plot(self, ax, color='b'):
        # plots the complex unit number
        ax.scatter(self.x, self.y, color=color)


class complxAlg(object):
    # a member of the lie algabra of unit complex numbers S1
    def __init__(self, theta):
        # make the member
        self.theta = theta

    def __repr__(self):
        # print the member
        return "{:2.4f}i".format(self.theta)

    def vee(self):
        # returns a memeber of the associated VECTOR space
        return theta

    def exp(self):
        # map the lie algebra member to the corresponding lie group member
        return complxGrp(np.cos(self.theta), np.sin(self.theta))

    def adj(self, X):
        # X is a member of the partent lie group
        # (a+bi)(theta-i)(a-bi) = (a^2+b^2)theta-i
        # but for complex unit numbers (a^2+b^2)=1 so this is self adjoint
        return self

    def plot(self, pt):
        # plots an arrow depicting the "velocity" of self at pt
        pass


if __name__ == '__main__':

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    # plot a cool thing

    a = complxGrp(0, 1)  # 90
    v = complxAlg(np.pi/6)
    b = a.step(v)  # shift by 30 degs anticlockwise to 120

    a.plot(ax)
    b.plot(ax, 'r')

    print(a)
    print(b)

    # plotted for style
    circle = plt.Circle([0, 0], 1, color="k", clip_on=False, fill=False)
    ax.add_artist(circle)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    plt.show()
