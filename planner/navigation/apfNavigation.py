import numpy as np
from typing import List
import math

class ApfNavigation:
    def __init__(self, road_width: float, v_width: float, v_length: float, start_point,
                 end_point: np.array, obstacles: List[np.array]) -> None:
        """
        road_width: 道路宽度
        v_width: 车辆宽度
        v_length：车辆长度
        startPoint：出发起始点,[x,y,vx,vy]
        endPoint：出发终止点,[x,y,vx,vy]
        obstacles：场景中的障碍物, List[[x,y,vx,vy]]
        """
        self.road_width = road_width
        self.v_width = v_width
        self.v_length = v_length
        self.startPoint = np.array([start_point.x, start_point.y, start_point.v * math.cos(start_point.theta), 
                           start_point.v * math.cos(start_point.theta)])
        self.endPoint = np.array([end_point[0], end_point[1], 0, 0])
        self.obstacles = obstacles
        # 计算引力的增益系数
        self.eta_att = 5
        # 计算斥力的增益系数
        self.eta_rep_ob = 15
        # 计算边界斥力的增益系数
        self.eta_rep_edge = 50
        # 障碍物影响范围
        self.r = 20
        self.object = self.obstacles + self.endPoint
        self.object_num = len(self.object)
        # todo：步长
        self.step_length = 0.5
        # 最大迭代次数
        self.num_iter = 200
        self.unit_vector = None
        self.dist = None

    def navigation(self) -> List:
        """
        Main interface for ego vehicle navigation.
        """
        navigation_path = []
        current_ego_position: np.array = self.startPoint
        unit_vector_Fsum = []
        i = 0
        while self.europeanDistance(current_ego_position[:2], self.endPoint[:2]) > 0.5:

            self.unit_vector = self.ego2Object(current_ego_position)
            F_rep_ob = self.calRepulsiveForce()
            F_rep_edge = self.calEdgeRepulsiveFroce(current_ego_position)
            # 所有障碍物的合斥力矢量
            F_req = [np.sum(F_rep_ob[:, 0] + F_rep_edge[0, 0], np.sum(F_rep_ob[:, 1] + F_rep_edge[0, 1]))]
            F_req = np.array(F_req)
            # 引力矢量
            F_att = [self.eta_att * self.dist[self.object_num - 1, 0], self.eta_att * self.dist[self.object_num - 1, 0]
                     * self.unit_vector[self.object_num - 1, 1]]
            F_att = np.array(F_att)
            # 总合力矢量
            F_sum = [F_req[0, 0] + F_att[0, 0], F_req[0, 1] + F_att[0, 1]]
            F_sum = np.array(F_sum)
            # 总合力单位向量
            unit_vector_Fsum.append(1 / np.linalg.norm(F_sum) * F_sum)
            unit_vector_Fsum = np.array(unit_vector_Fsum)
            i += 1
            # 计算车辆下一步位置
            current_ego_position = current_ego_position + self.step_length * unit_vector_Fsum
            navigation_path.append(current_ego_position)
        return navigation_path

    def ego2Object(self, current_position) -> np.array:
        """
        calculate the unit vector between ego vehicle and other object(obstacles, end point).
        return: the unit vector of ego to obstacle and end point to ego.
        """
        delta = np.zeros((self.object_num, 2))
        self.dist = np.zeros((self.object_num, 1))
        unit_vector = np.zeros((self.object_num, 2))
        # ego 2 obstacle
        for j in range(self.object_num - 1):
            # 用车辆点-障碍点表达斥力
            delta[j, :] = current_position[0, 0:2] - self.object[j, 0:2]
            # 车辆当前位置与障碍物的距离
            self.dist[j, 0] = np.linalg.norm(delta[j, :])
            # 防止除以零错误
            if self.dist[j, 0] != 0:
                # 斥力的单位方向向量
                unit_vector[j, :] = delta[j, :] / self.dist[j, 0]
            else:
                # 如果距离为0，则无法定义单位方向向量，可以设置为零向量或进行其他错误处理
                unit_vector[j, :] = [0, 0]
        # end 2 ego
        delta[self.object_num, :] = self.object[self.object_num, 0:2] - current_position[0, 0:2]
        self.dist[self.object_num, 0] = np.linalg.norm(delta[self.object_num, :])
        unit_vector[self.object_num, :] = delta[self.object_num, :] / self.dist[self.object_num, 0]
        return unit_vector

    def europeanDistance(self, prior_point, post_point) -> float:
        """
        calculate the european distance between two points.
        """
        delta_point = prior_point - post_point
        distance = np.sum(delta_point ** 2)
        return distance[0]

    def calRepulsiveForce(self) -> np.array:
        """
        calculate total repulsive force of obstacles.
        """
        F_rep_ob = np.zeros((self.object_num - 1, 2))
        for j in range(self.object_num - 1):
            if self.dist[j, 0] >= self.r:
                F_rep_ob[j, 0] = 0
                F_rep_ob[j, 1] = 0
            else:
                # 障碍物的斥力1，方向由障碍物指向车辆
                F_rep_ob1_abs = (self.eta_rep_ob * (1 / self.dist[j, 0] - 1 / self.r) *
                                 self.dist[self.object_num - 1, 0] / self.dist[j, 0] ** 2)
                F_rep_ob1 = [F_rep_ob1_abs * self.unit_vector[j, 0], F_rep_ob1_abs * self.unit_vector[j, 1]]

                # 障碍物的斥力2，方向可能由车辆指向目标点（这里假设目标点是dist的最后一个点）
                F_rep_ob2_abs = 0.5 * self.eta_rep_ob * (1 / self.dist[j, 0] - 1 / self.r) ** 2
                F_rep_ob2 = [F_rep_ob2_abs * self.unit_vector[self.object_num - 1, 0], F_rep_ob2_abs *
                             self.unit_vector[self.object_num - 1, 1]]

                # 改进后的障碍物合斥力计算
                F_rep_ob[j, :] = np.array(F_rep_ob1) + np.array(F_rep_ob2)
        return F_rep_ob

    def calEdgeRepulsiveFroce(self, current_position) -> np.array:
        """
        Add the boundary repulsion potential field, and select the corresponding repulsion function according
        to the current position of the vehicle.
        """
        # 下道路边界区域力场，方向指向y轴正向
        if -self.road_width + self.v_width / 2 < current_position[0, 1] <= -self.road_width / 2:
            F_rep_edge = [0, self.eta_rep_edge * np.linalg.norm(current_position[:, 2:]) *
                          (np.exp(-self.r / 2 - current_position[0, 1]))]
        # 下道路分界线区域力场，方向指向y轴负向
        elif -self.road_width / 2 < current_position[0, 1] <= - self.v_width:
            F_rep_edge = [0, (1 / 3) * self.eta_rep_edge * (current_position[0, 1] ** 2)]
        # 上道路分界线区域力场，方向指向y轴正向
        elif self.v_width / 2 < current_position[0, 1] < self.road_width / 2:
            F_rep_edge = [0, -(1 / 3) * self.eta_rep_edge * (current_position[0, 1] ** 2)]
        # 上道路边界区域力场，方向指向y轴负向
        elif self.road_width / 2 < current_position[0, 1] <= self.road_width - self.v_width / 2:
            F_rep_edge = [0, self.eta_rep_edge * np.linalg.norm(current_position[:, 2:]) *
                          (np.exp(current_position[0, 1] - self.r / 2))]
        else:
            F_rep_edge = [0, 0]
        return np.array(F_rep_edge)
    
    def extractObstacleInfo(self, obstacles):
        obstacles_info = []
        for obstacle in obstacles:
            

