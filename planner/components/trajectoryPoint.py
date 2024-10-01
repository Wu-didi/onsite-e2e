import math
import numpy as np
from planner.components.pathPoint import PathPoint


def NormalizeAngle(angle_rad):
    # to normalize an angle to [-pi, pi]
    a = math.fmod(angle_rad + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a = a + 2.0 * math.pi
    return a - math.pi
def LinearInterpolate(path_point_0, path_point_1, rs_inter):
    ''' path point interpolated linearly according to rs value
    path_point_0 should be prior to path_point_1'''

    def lerp(x0, x1, w):
        return x0 + w * (x1 - x0)

    def slerp(a0, a1, w):
        # angular, for theta
        a0_n = NormalizeAngle(a0)
        a1_n = NormalizeAngle(a1)
        d = a1_n - a0_n
        if d > math.pi:
            d = d - 2 * math.pi
        elif d < -math.pi:
            d = d + 2 * math.pi
        a = a0_n + w * d
        return NormalizeAngle(a)

    rs_0 = path_point_0.rs
    rs_1 = path_point_1.rs
    weight = (rs_inter - rs_0) / (rs_1 - rs_0)
    if weight < 0 or weight > 1:
        print("weight error, not in [0, 1]")
        quit()
    rx_inter = lerp(path_point_0.rx, path_point_1.rx, weight)
    ry_inter = lerp(path_point_0.ry, path_point_1.ry, weight)
    rtheta_inter = slerp(path_point_0.rtheta, path_point_1.rtheta, weight)
    rkappa_inter = lerp(path_point_0.rkappa, path_point_1.rkappa, weight)
    rdkappa_inter = lerp(path_point_0.rdkappa, path_point_1.rdkappa, weight)
    return PathPoint([rx_inter, ry_inter, rs_inter, rtheta_inter, rkappa_inter, rdkappa_inter])


class TrajPoint:
    def __init__(self, tp_list):
        # tp_list: from sensors, [x, y, v, a, theta, kappa]
        self.matched_point = None
        self.x = tp_list[0]
        self.y = tp_list[1]
        self.v = tp_list[2]
        self.a = tp_list[3]
        self.theta = NormalizeAngle(tp_list[4])
        self.kappa = tp_list[5]
        self.time = 0
        self.s = 0

    def MatchPath(self, path_points):
        '''
        find the closest/projected point on the reference path
        the deviation is not large; the curvature is not large
        '''

        def DistSquare(traj_point, path_point):
            dx = path_point.rx - traj_point.x
            dy = path_point.ry - traj_point.y
            return (dx ** 2 + dy ** 2)

        dist_all = []
        for path_point in path_points:
            dist_all.append(DistSquare(self, path_point))
        dist_min = DistSquare(self, path_points[0])
        index_min = 0
        # for index, path_point in enumerate(path_points):
        #     dist_temp = DistSquare(self, path_point)
        #     if dist_temp < dist_min:
        #         dist_min = dist_temp
        #         index_min = index
        # 上述代码又计算了一次不必要的最小距离
        for index in range(len(path_points)):
            if dist_all[index] < dist_min:
                dist_min = dist_all[index]
                index_min = index
        path_point_min = path_points[index_min]
        if index_min == 0 or index_min == len(path_points) - 1:
            self.matched_point = path_point_min
        else:
            path_point_next = path_points[index_min + 1]
            path_point_last = path_points[index_min - 1]
            vec_p2t = np.array([self.x - path_point_min.rx, self.y - path_point_min.ry])
            vec_p2p_next = np.array([path_point_next.rx - path_point_min.rx, path_point_next.ry - path_point_min.ry])
            vec_p2p_last = np.array([path_point_last.rx - path_point_min.rx, path_point_last.ry - path_point_min.ry])
            if np.dot(vec_p2t, vec_p2p_next) * np.dot(vec_p2t, vec_p2p_last) >= 0:
                self.matched_point = path_point_min
            else:
                if np.dot(vec_p2t, vec_p2p_next) >= 0:
                    rs_inter = path_point_min.rs + np.dot(vec_p2t, vec_p2p_next / np.linalg.norm(vec_p2p_next))
                    self.matched_point = LinearInterpolate(path_point_min, path_point_next, rs_inter)
                else:
                    rs_inter = path_point_min.rs - np.dot(vec_p2t, vec_p2p_last / np.linalg.norm(vec_p2p_last))
                    self.matched_point = LinearInterpolate(path_point_last, path_point_min, rs_inter)
        return self.matched_point

    def LimitTheta(self, theta_thr=math.pi / 6):
        # limit the deviation of traj_point.theta from the matched path_point.rtheta within theta_thr
        if self.theta - self.matched_point.rtheta > theta_thr:
            self.theta = NormalizeAngle(self.matched_point.rtheta + theta_thr)  # upper limit of theta
        elif self.theta - self.matched_point.rtheta < -theta_thr:
            self.theta = NormalizeAngle(self.matched_point.rtheta - theta_thr)  # lower limit of theta
        else:
            pass  # maintained, actual theta should not deviate from the path rtheta too much

    def IsOnPath(self, dist_thr=0.5):
        # whether the current traj_point is on the path
        dx = self.matched_point.rx - self.x
        dy = self.matched_point.ry - self.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist <= dist_thr:
            return True
        else:
            return False
