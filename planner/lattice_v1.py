#! /usr/bin/env python3
# _*_ coding: utf-8 _*_
# from utils.logger_lattice import lattice_logger
import csv
import os
import time
import math
import numpy as np
from lxml import etree
from scipy.interpolate import splrep, splev
import scipy.signal
import warnings
from planner.utilsTools import NormalizeAngle
import pandas as pd
from typing import List
from planner.components.SampleBase import SampleBasis
from planner.components.dCondition import D_conditon
from planner.components.knotState import KnotState
from planner.components.pathPoint import PathPoint
from planner.components.trajectoryPoint import TrajPoint
from planner.components.obstacle import Obstacle
from planner.localPlanner import LocalPlanner
from planner.navigation.elements.openDrive import OpenDrive
from planner.plannerBase import PlannerBase

CSVLOG = True
delta_s = 0
delta_time = 0
current_time = 0
previous_traj = []
first_run = True
y_ego_lc = 0
x_ego_lc = 0

kdis = 10
b_use_sticher = True

transformed_path = []

near_destination = False

M_PI = 3.141593

# 车辆属性, global const. and local var. check!
VEH_L = 4.5  # length
VEH_W = 2.6  # width
MAX_V = 30
MIN_V = -2

MAX_OBJ_DIS = 10000

MAX_A = 10
MIN_A = -20
MAX_LAT_A = 100  # 参考apollo，横向约束应该是给到向心加速度，而不是角速度

# cost权重
SPEED_COST_WEIGHT = 1  # 速度和目标速度差距，暂时不用
DIST_TRAVEL_COST_WEIGHT = 1  # 实际轨迹长度，暂时不用
LAT_COMFORT_COST_WEIGHT = 1  # 横向舒适度
LAT_OFFSET_COST_WEIGHT = 1  # 横向偏移量

# 前四个是中间计算时用到的权重，后三个是最终合并时用到的
LON_OBJECTIVE_COST_WEIGHT = 1  # 纵向目标cost，暂时不用
LAT_COST_WEIGHT = 1  # 横向约束，包括舒适度和偏移量
LON_COLLISION_COST_WEIGHT = 1  # 碰撞cost
DESTINATION_WEIGHT = 1

# Constant
delta_t = 0.04 * 1  # fixed time between two consecutive trajectory points, sec
v_tgt = 5  # fixed target speed, m/s
sight_range = 100  # 判断有无障碍物的视野距离
# ttcs = [3, 4, 5]  # static ascending time-to-collision, sec
ttcs = [-1, -0.5, 0, 0.5]  # change this u should also change line 682 605 878 now means acc
theta_thr = M_PI / 6  # delta theta threshold, deviation from matched path


def detail_xy(xy):  # 将原车道中心线上少量的点加密为0.1m间隔的点
    [direct, add_length] = get_lane_feature(xy)
    dist_interval = 1
    new_xy = [[], []]
    new_direct = []
    new_add_len = [0]
    temp_length = dist_interval
    for k in range(0, len(xy[0]) - 1):
        new_xy[0].append(xy[0][k])
        new_xy[1].append(xy[1][k])
        new_add_len.append(temp_length)
        new_direct.append(direct[k])
        while temp_length < add_length[k + 1]:
            temp_length += dist_interval
            new_xy[0].append(new_xy[0][-1] + dist_interval * math.cos(direct[k]))
            new_xy[1].append(new_xy[1][-1] + dist_interval * math.sin(direct[k]))
            new_add_len.append(temp_length)
            new_direct.append(direct[k])
    return [new_xy, new_direct, new_add_len]


def find_same_value_index(object_list):
    """
    找到列表中相同值的索引
    """
    index_dict = {}
    for index, item in enumerate(object_list):
        if item in index_dict:
            index_dict[item].append(index)
        else:
            index_dict[item] = [index]
    duplicates = {value: indexes for value, indexes in index_dict.items() if len(indexes) > 1}
    return duplicates.values()


def get_lane_feature(xy):
    xy = np.array(xy)
    # n为中心点个数，2为x,y坐标值
    x_prior = xy[0][:-1]
    y_prior = xy[1][:-1]
    x_post = xy[0][1:]
    y_post = xy[1][1:]
    # 根据前后中心点坐标计算【行驶方向】
    dx = x_post - x_prior
    dy = y_post - y_prior

    direction = list(map(lambda d: d > 0 and d or d + 2 * np.pi, np.arctan2(dy, dx)))

    length = np.sqrt(dx ** 2 + dy ** 2)
    length = length.tolist()
    for i in range(len(length) - 1):
        length[i + 1] += length[i]
    length.insert(0, 0)
    return direction, length


def smooth_cv(cv_init, point_num=1000):
    cv = cv_init
    list_x = cv[:, 0]
    list_y = cv[:, 1]
    if type(cv) is not np.ndarray:
        cv = np.array(cv)
    delta_cv = cv[1:, ] - cv[:-1, ]
    s_cv = np.linalg.norm(delta_cv, axis=1)

    s_cv = np.array([0] + list(s_cv))
    s_cv = np.cumsum(s_cv)

    bspl_x = splrep(s_cv, list_x, s=0.1, k=1)
    bspl_y = splrep(s_cv, list_y, s=0.1, k=1)
    # values for the x axis
    s_smooth = np.linspace(0, max(s_cv), point_num)
    # get y values from interpolated curve
    x_smooth = splev(s_smooth, bspl_x)
    y_smooth = splev(s_smooth, bspl_y)
    new_cv = np.array([x_smooth, y_smooth]).T

    delta_new_cv = new_cv[1:, ] - new_cv[:-1, ]
    s_accumulated = np.cumsum(np.linalg.norm(delta_new_cv, axis=1))
    s_accumulated = np.concatenate(([0], s_accumulated), axis=0)
    return new_cv, s_accumulated


def RefLine(origion, destination):
    origion_knotstate = KnotState()
    origion_knotstate.x, origion_knotstate.y, origion_knotstate.h = origion
    destination_knotstate = KnotState()
    destination_knotstate.x, destination_knotstate.y, destination_knotstate.h = destination
    if abs(origion_knotstate.y - destination_knotstate.y) < 1:  # 该段为直线段
        # print('该段为直线段')
        new_xy, new_direct, new_add_len = detail_xy([[origion[0], destination[0]], [origion[1], destination[1]]])
        return new_xy[0], new_xy[1]
    else:
        # planned_curve = CurvePlaner(origion_knotstate, destination_knotstate)
        planned_curve, _ = smooth_cv(np.array([origion, destination]))
        return planned_curve[:, 0], planned_curve[:, 1]


def AStarRefLine(map_path, start_point, end_point):
    """
    解析地图文件，使用A*算法找寻全局最优路径
    """
    if os.path.exists(map_path):
        doc = etree.parse(map_path)
        opendrive = OpenDrive(doc.getroot())
        opendrive.buildMap()
        opendrive.calLaneLineAndCenterLine()
        # 如果return false则未找到最优解
        return opendrive.findPath(start_point[0], start_point[1],
                                  end_point[0], end_point[1], False, False)
    else:
        print("no found map file.")


def FrenetToCartesian(path_point, s_condition, d_condition):
    ''' from Frenet to Cartesian coordinate
    copy Apollo cartesian_frenet_conversion.cpp'''
    rx, ry, rs, rtheta, rkappa, rdkappa = path_point.rx, path_point.ry, path_point.rs, \
        path_point.rtheta, path_point.rkappa, path_point.rdkappa
    # check value
    if abs(rkappa) > 10e5:
        rkappa = 0
    if math.fabs(rs - s_condition[0]) >= 1.0e-6:
        pass
        # print("the reference point s and s_condition[0] don't match")

    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)

    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]

    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    if pd.isnull(one_minus_kappa_r_d):
        one_minus_kappa_r_d = 1.0
    if abs(one_minus_kappa_r_d) > 10e50:
        one_minus_kappa_r_d = 1.0
    tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
    delta_theta = math.atan2(d_condition[1], one_minus_kappa_r_d)
    cos_delta_theta = math.cos(delta_theta)
    theta = NormalizeAngle(delta_theta + rtheta)
    # check
    if abs(rdkappa) > 5e50:
        rdkappa = 0
    if abs(d_condition[0]) > 5e50:
        d_condition[0] = 0
    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
    with warnings.catch_warnings(record=True) as w:
        kappa = ((d_condition[2] + kappa_r_d_prime * tan_delta_theta) * cos_delta_theta ** 2 /
                 one_minus_kappa_r_d + rkappa) * cos_delta_theta / one_minus_kappa_r_d
        if any(issubclass(warning.category, RuntimeWarning) for warning in w):
            print("d_condition[2], kappa_r_d_prime, tan_delta_theta, cos_delta_theta, rkappa",
                  d_condition[2], kappa_r_d_prime, tan_delta_theta, cos_delta_theta, rkappa)

    d_dot = d_condition[1] * s_condition[1]
    # check Nan
    if pd.isnull(d_dot):
        d_dot = 0
    # check Large value
    if abs(d_dot) > 10e50:
        d_dot = 0
    # ======================================
    with warnings.catch_warnings(record=True) as w:
        v = math.sqrt((one_minus_kappa_r_d * s_condition[1]) ** 2 + d_dot ** 2)
        if any(issubclass(warning.category, RuntimeWarning) for warning in w):
            pass
            # lattice_logger.info("v % s, one_minus_kappa_r_d % s,
            # s_condition[1] % s, d_dot % s", v,one_minus_kappa_r_d,
            #                     s_condition[1], d_dot)
        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
        a = s_condition[2] * one_minus_kappa_r_d / cos_delta_theta + s_condition[1] ** 2 / cos_delta_theta * (
                d_condition[1] * delta_theta_prime - kappa_r_d_prime)
        if any(issubclass(warning.category, RuntimeWarning) for warning in w):
            pass
            # lattice_logger.info(" d_condition[1] % s, delta_theta_prime % s, kappa_r_d_prime % s",
            #                     d_condition[1], delta_theta_prime, kappa_r_d_prime)
    tp_list = [x, y, v, a, theta, kappa]
    return TrajPoint(tp_list)


def CalcRefLine(cts_points):  # 输入参考轨迹的x y 计算rs/rtheta/rkappa/rdkappa 此时是笛卡尔坐标系 rs为已走路程 rtheta为角度
    '''
    deal with reference path points 2d-array
    to calculate rs/rtheta/rkappa/rdkappa according to cartesian points
    '''
    rx = cts_points[0]  # the x value
    ry = cts_points[1]  # the y value
    # 剔除坐标中的相同值
    if find_same_value_index(rx):
        same_index = list(find_same_value_index(rx))
        rx = np.delete(rx, same_index[0][1])
        ry = np.delete(ry, same_index[0][1])
    rs = np.zeros_like(rx)
    rtheta = np.zeros_like(rx)
    rkappa = np.zeros_like(rx)
    rdkappa = np.zeros_like(rx)
    for i, x_i in enumerate(rx):
        # y_i = ry[i]
        if i != 0:
            dx = rx[i] - rx[i - 1]
            dy = ry[i] - ry[i - 1]
            rs[i] = rs[i - 1] + math.sqrt(dx ** 2 + dy ** 2)
        if i < len(ry) - 1:
            dx = rx[i + 1] - rx[i]
            dy = ry[i + 1] - ry[i]
            ds = math.sqrt(dx ** 2 + dy ** 2)
            rtheta[i] = math.copysign(math.acos(dx / ds),
                                      dy)  # acos求角度 copysign功能为返回第一个输入的值和第二个输入的符号(即dy>0在0-pi dy<0在-pi-0)
            # if i == 0:
            #     rs[i] = 0
            #     rtheta[i] = 0
            # elif 0 < i < len(ry) - 1:
            #     dx = rx[i + 1] - rx[i]
            #     dy = ry[i + 1] - ry[i]
            #     ds = math.sqrt(dx ** 2 + dy ** 2)
            #     with warnings.catch_warnings(record=True) as w:
            #         rtheta[i] = math.copysign(math.acos(dx / ds),
            #                                   dy)  # acos求角度 copysign功能为返回第一个输入的值和第二个输入的符号(即dy>0在0-pi dy<0在-pi-0)
            #         if any(issubclass(warning.category, RuntimeWarning) for warning in w):
            #             print("dx", "ds", "dy", "rx", dx, ds, dy)
            #             print(" rx[i + 1]", "rx[i]", rx[i + 1],  rx[i])
            #             print("rx=", rx)
            #             print("ry=", ry)
            rs[i] = rs[i - 1] + ds
    if len(rtheta) > 2:
        rtheta[-1] = rtheta[-2]  # 最后一个时刻的角度没法求就直接等于倒数第二个时刻
        rkappa[:-1] = np.diff(rtheta) / (np.diff(rs) + 0.000001)  # 角度变化量/路程变化量
        rdkappa[:-1] = np.diff(rkappa) / (np.diff(rs) + 0.000001)
        rkappa[-1] = rkappa[-2]
        rdkappa[-1] = rdkappa[-3]
        rdkappa[-2] = rdkappa[-3]
    if len(rkappa) > 333:
        window_length = 333
    elif len(rkappa) % 2 == 0:
        window_length = len(rkappa) - 1
    else:
        window_length = len(rkappa)
    polyorder = 5
    if window_length <= polyorder:
        polyorder = window_length - 1
    rkappa = scipy.signal.savgol_filter(rkappa, window_length, polyorder)  # 平滑
    rdkappa = scipy.signal.savgol_filter(rdkappa, window_length, polyorder)
    path_points = []
    for i in range(len(rx)):
        path_points.append(PathPoint([rx[i], ry[i], rs[i], rtheta[i], rkappa[i], rdkappa[i]]))  # 生成笛卡尔坐标系下的参考轨迹点
    return path_points


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

    if weight < 0:
        weight = 0
    if weight < 0 or weight > 1:
        print("rs_0", "rs_1", rs_0, rs_1)
        print("weight error, not in [0, 1]")
        # quit()
    rx_inter = lerp(path_point_0.rx, path_point_1.rx, weight)
    ry_inter = lerp(path_point_0.ry, path_point_1.ry, weight)
    rtheta_inter = slerp(path_point_0.rtheta, path_point_1.rtheta, weight)
    rkappa_inter = lerp(path_point_0.rkappa, path_point_1.rkappa, weight)
    rdkappa_inter = lerp(path_point_0.rdkappa, path_point_1.rdkappa, weight)
    return PathPoint([rx_inter, ry_inter, rs_inter, rtheta_inter, rkappa_inter, rdkappa_inter])


class LATTICE(PlannerBase):
    def __init__(self):
        super().__init__()
        self.first_run = True
        self.way_points = None
        self.map_path = None
        self.scenario_type = None
        self.rx = []
        self.ry = []
        # 假设主车初始状态为加速度0，前轮转角0
        self.acc_target_previous = 0
        self.wheel_target_previous = 0
        self.transformed_path = []
        self.previous_d_condition: List[D_conditon] = []
        self.delta_t = 0.1
        self.mixed = False
        self.intersection = False

    def init(self, scenario_dict):
        print("----------------------------LATTICE INIT----------------------------")
        global origion, destination
        origion = np.array([scenario_dict['task_info']['startPos'][0], scenario_dict['task_info']['startPos'][1], 0])
        destination_x = (scenario_dict['task_info']['targetPos'][1][0] +
                         scenario_dict['task_info']['targetPos'][0][0]) / 2
        destination_y = (scenario_dict['task_info']['targetPos'][1][1] +
                         scenario_dict['task_info']['targetPos'][0][1]) / 2
        destination = np.array([destination_x, destination_y, 0])
        self.scenario_type = scenario_dict['type']
        self.map_path = scenario_dict['source_file']['xodr']
        self.way_points = scenario_dict['task_info']['waypoints']
        if "follow" in self.map_path or "highway_merge" in self.map_path or "cutin" in self.map_path:
            self.rx, self.ry = RefLine(origion, destination)
        elif AStarRefLine(self.map_path, origion, destination):
            self.rx, self.ry = AStarRefLine(self.map_path, origion, destination)
        else:
            self.rx, self.ry = RefLine(origion, destination)


    def act(self, observation):
        global current_time, first_run, delta_time
        if not first_run:
            delta_time = time.time() - current_time
            # lattice_logger.info("delta time is %f", delta_time)
        current_time = time.time()
        self.transformed_path.clear()
        control = self.alg(observation.ego_info, observation.object_info, current_time)
        return control

    def transfrom(self, x_ori, y_ori, x_ego, y_ego, ego_heading):
        x_scaled = x_ori - x_ego
        y_scaled = y_ori - y_ego
        ego_heading = ego_heading
        # Apply rotation
        x_translated = x_scaled * math.cos(ego_heading) + y_scaled * math.sin(ego_heading)
        y_translated = -x_scaled * math.sin(ego_heading) + y_scaled * math.cos(ego_heading)
        return [x_translated, y_translated, x_ori, y_ori]

    def alg(self, ego, obs, current_time):
        # 主车和障碍物位置信息
        ego_info = ego
        x_ego, y_ego, v_ego, heading_ego = ego_info.x, ego_info.y, ego_info.v, ego_info.yaw
        global first_run, y_ego_lc, x_ego_lc

        tp_list = [x_ego, y_ego, v_ego, 0, heading_ego, 0]  # from sensor actually, an example here
        traj_point = TrajPoint(tp_list)  # [x, y, v, a, theta, kappa]

        heading_ego_cal = NormalizeAngle(math.atan2(y_ego - y_ego_lc, x_ego - x_ego_lc))
        # lattice_logger.info("check ego heading %f ,cal value %f", NormalizeAngle(heading_ego), heading_ego_cal)

        y_ego_lc = y_ego
        x_ego_lc = x_ego

        # 测试
        cts_points = np.array([self.rx, self.ry])
        path_points = CalcRefLine(cts_points)

        for path_point in path_points:  # cts_points  path_points
            transformed_point = self.transfrom(path_point.rx, path_point.ry, x_ego, y_ego, heading_ego)
            self.transformed_path.append(transformed_point)

        static_obstacles = []
        for key in obs:
            ego_obstacles = obs[key]
            for id in ego_obstacles:
                obstacles = ego_obstacles[id]

                static_obstacles.append(Obstacle(
                    [obstacles.x, obstacles.y, obstacles.v, obstacles.length, obstacles.width, obstacles.yaw,
                     'static']))

        for obstacle in static_obstacles:
            obstacle.MatchPath(path_points)  # 同样match障碍物与参考轨迹

        traj_point.MatchPath(path_points)  # matching once is enough 将traj_point(单点)与path_points(序列)中最近的点匹配
        samp_basis = SampleBasis(traj_point, theta_thr, ttcs, v_tgt)  ### 采样区间(类动作空间)
        local_planner = LocalPlanner(traj_point, path_points, static_obstacles,
                                     samp_basis, self.transformed_path, delta_time, current_time, v_tgt, self.first_run,
                                     self.previous_d_condition, destination, self.scenario_type, self.delta_t)
        ### 规划器 输入为目前位置 参考轨迹点 障碍物位置 采样空间
        traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, static_obstacles, samp_basis)

        if not traj_points_opt:
            # lattice_logger.info("紧急停车")
            local_planner = LocalPlanner(traj_point, path_points, static_obstacles, samp_basis, self.transformed_path
                                         , delta_time, current_time, v_tgt, self.first_run, self.previous_d_condition,
                                         destination, self.scenario_type, self.delta_t)
            local_planner.status = 'brake'
            traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, static_obstacles, samp_basis)
        else:  # 正常情况下在正常采样空间内如果有opt 就将opt规划出的点作为下一时刻的traj
            traj_points = []
            for tp_opt in traj_points_opt:
                traj_points.append([tp_opt.x, tp_opt.y, tp_opt.v, tp_opt.a, tp_opt.theta, tp_opt.kappa])

        acc_target = 0  # initial value
        wheel_target = 0

        # lattice_logger.info("check last point x %f", transformed_path[-1][0])

        try:
            target_point: TrajPoint = traj_points_opt[0]
            angle = NormalizeAngle(heading_ego)
            # lattice_logger.info("check ego later heading %f",angle)

            transformed_traj = []
            for point in traj_points_opt:
                transformed_point = self.transfrom(point.x, point.y, x_ego, y_ego, angle)
                transformed_point.append(point.a)
                # lattice_logger.info("traj ori x %f ,y %f , transformed x %f, transformed % f ,ego x %f, ego y %f",
                # point.x, point.y, transformed_point[0],transformed_point[1],x_ego,y_ego)
                transformed_traj.append(transformed_point)

                # target_point.x = transformed_point[0]
                # target_point.y = transformed_point[1]
                # target_point.a = transformed_point[4]

            found_pos_match = False
            control_dis = 20
            global near_destination
            if near_destination:
                control_dis = min(self.transformed_path[-1][0], 10)

            # lattice_logger.info("control dis %f", control_dis)

            for tran_point in transformed_traj:  # #use path transformed_traj transformed_path
                # lattice_logger.info("path point x %f, point y %f, a %f",tran_point[0] ,tran_point[1],tran_point[4])
                # print(len(transformed_traj))
                with warnings.catch_warnings(record=True) as w:
                    if abs(tran_point[0]) > 10e50:
                        tran_point[0] = 0
                    if abs(tran_point[1]) > 10e50:
                        tran_point[1] = 0
                    if tran_point[0] > control_dis and math.sqrt(tran_point[0] ** 2 + tran_point[1] ** 2) > 5:
                        target_point.x = tran_point[0]
                        target_point.y = tran_point[1]
                        target_point.a = tran_point[4]
                        if any(issubclass(warning.category, RuntimeWarning) for warning in w):
                            pass
                            # lattice_logger.info("tran_point[0] % s,  tran_point[1] % d", tran_point[0], tran_point[1])
                        found_pos_match = True
                        break
            if not found_pos_match:
                # lattice_logger.info("target control point not found then found a near point")
                last_point = transformed_traj[-1]
                target_point.x = last_point[0]
                target_point.y = last_point[1]

            # trajectory log
            # for traj_point in traj_points_opt:
            #     lattice_logger.info("final time %f, acc %f ,v %f ,s %f", traj_point.time,
            #     traj_point.a, traj_point.v ,traj_point.s)

            for traj_point in traj_points_opt:
                # lattice_logger.info("final time %f, acc %f", traj_point.time, traj_point.a)
                if traj_point.time > 0.2:
                    acc_target = traj_point.a
                    # print("found  acc_target ",acc_target)
                    break

                # lattice_logger.info("last point as target  x %f, point y %f",target_point.x ,target_point.y)
            # else:
            # lattice_logger.info("target found x %f, target y %f",target_point.x , target_point.y)

            x_translated = target_point.x
            y_translated = target_point.y
            dis = math.sqrt(x_translated ** 2 + y_translated ** 2)
            diff_heading = math.atan(y_translated / x_translated)
            wheel_target_ori = math.atan(2 * 4.55 * y_translated / (dis ** 2))  # 4.55
            wheel_target = wheel_target_ori

            # lattice_logger.info("qhz to check dis %f, y dis f %f,x dis %f, wheel target %f ,target a  %f,
            # ego angle %f", dis, y_translated,x_translated, wheel_target_ori,acc_target, angle)

            if (CSVLOG):
                with open('wheeltarget.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([wheel_target, y_translated])

        except:
            # lattice_logger.info("no wheeltarget")
            wheel_target = 0

        # lattice_logger.info("final acc target %f, wheel_target %f ,ego_angle %f ",acc_target,wheel_target,heading_ego)
        if (CSVLOG):
            # with open('wheeltarget.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow([wheel_target])
            with open('ego_point.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # lattice_logger.info("check ego x %f, y %f", traj_point.x,traj_point.y)
                ego_point = [x_ego, y_ego, heading_ego]
                writer.writerow(ego_point)

        # check wheel_target, +-0.7
        if wheel_target > 0.7:
            wheel_target = 0.7
        if wheel_target < -0.7:
            wheel_target = -0.7
        # check acc_target, +-3
        if acc_target < -9:
            acc_target = -9
        if acc_target > 9:
            acc_target = 9
        if self.delta_t == 0.1:
            JERK_LIMIT = 48
            ROT_LIMIT = 1.2
        else:
            JERK_LIMIT = 125
            ROT_LIMIT = 3

        # check jerk, +-5
        jerk_target = (acc_target - self.acc_target_previous) / self.delta_t
        if jerk_target < -JERK_LIMIT:
            acc_target_modify = -JERK_LIMIT - jerk_target
            acc_target = acc_target + acc_target_modify * self.delta_t
        if jerk_target > JERK_LIMIT:
            acc_target_modify = jerk_target - JERK_LIMIT
            acc_target = acc_target - acc_target_modify * self.delta_t
        # check angle_rate, +-0.15
        angle_rate = (wheel_target - self.wheel_target_previous) / self.delta_t
        if angle_rate < -ROT_LIMIT:  # 减少的太多了
            wheel_target_modify = -angle_rate - ROT_LIMIT
            wheel_target = wheel_target + wheel_target_modify * self.delta_t
        if angle_rate > ROT_LIMIT:  # 增加的太多了
            wheel_target_modify = angle_rate - ROT_LIMIT
            wheel_target = wheel_target - wheel_target_modify * self.delta_t
        # 减速度不可能比车速还大
        if acc_target < 0 and abs(acc_target) > ego.v:
            acc_target = -ego.v

        # check acc_target, +-3
        # if acc_target < -3:
        #     acc_target = -3
        # if acc_target > 3:
        #     acc_target = 3
        #     # check wheel_target, +-0.7
        # if wheel_target > 0.698:
        #     wheel_target = 0.698
        # if wheel_target < -0.698:
        #     wheel_target = -0.698

        drivecontrol = [acc_target, wheel_target]  # 最终返回控制信息
        self.previous_d_condition = local_planner.poly_traj.GetPreviousDCondition()
        if self.first_run:
            self.first_run = False
            y_ego_lc = y_ego
            x_ego_lc = x_ego
        self.acc_target_previous = acc_target
        self.wheel_target_previous = wheel_target
        return drivecontrol
