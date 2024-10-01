# import planner.Lattice.lattice_v1 as utils
from planner.utilsTools import *


class Obstacle:
    def __init__(self, obstacle_info):
        self.matched_point = None
        self.x = obstacle_info[0]
        self.y = obstacle_info[1]
        self.v = obstacle_info[2]
        self.length = obstacle_info[3]
        self.width = obstacle_info[4]
        self.heading = obstacle_info[5]  # 这里设定朝向是length的方向，也是v的方向
        self.type = obstacle_info[6]
        self.corner = self.GetCorner()

    def GetCorner(self):
        cos_o = math.cos(self.heading)
        sin_o = math.sin(self.heading)
        dx3 = cos_o * self.length / 2
        dy3 = sin_o * self.length / 2
        dx4 = sin_o * self.width / 2
        dy4 = -cos_o * self.width / 2
        return [self.x - (dx3 - dx4), self.y - (dy3 - dy4)]

    def MatchPath(self, path_points):
        '''
        find the closest/projected point on the reference path
        the deviation is not large; the curvature is not large
        '''

        def DistSquare(traj_point, path_point):
            dx = path_point.rx - traj_point.x
            dy = path_point.ry - traj_point.y
            return dx ** 2 + dy ** 2

        dist_all = []
        for path_point in path_points:
            dist_all.append(DistSquare(self, path_point))  # 求障碍物到reference line的各个点距
        dist_min = DistSquare(self, path_points[0])  # 与第一个参考点的距离
        index_min = 0
        for index, path_point in enumerate(path_points):  # 求最近的参考点
            dist_temp = DistSquare(self, path_point)
            if dist_temp < dist_min:
                dist_min = dist_temp
                index_min = index
        path_point_min = path_points[index_min]  # 得到障碍物到reference line的最短距离
        if index_min == 0 or index_min == len(path_points) - 1:
            self.matched_point = path_point_min
        else:
            path_point_next = path_points[index_min + 1]  # 上一时刻参考点和下一时刻参考点
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
