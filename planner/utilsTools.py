import math
import numpy as np
from typing import List
import scipy.signal
import pandas as pd
import warnings
from planner.components.trajectoryPoint import TrajPoint
from planner.components.pathPoint import PathPoint

VEH_L = 4.5
VEH_W = 2.2

def NormalizeAngle(angle_rad):
    # to normalize an angle to [-pi, pi]
    a = math.fmod(angle_rad + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a = a + 2.0 * math.pi
    return a - math.pi


def Dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def trajectorySticher(trajtorypoints: List, time):
    global previous_traj
    updated_previous_traj = []
    for point in trajtorypoints:
        if point.time >= time:
            updated_previous_traj.append(point)
    previous_traj = updated_previous_traj


def find_last_d_condition(previous_d_condition, compared_time):  #:List[D_conditon]
    for point in previous_d_condition:
        if point.time >= compared_time:
            return point


def find_last_d_condition_by_s(previous_d_condition: List, delta_s):  #
    for point in previous_d_condition:
        if point.s >= delta_s:  # if (point[4] > delta_s):
            return point
    # lattice_logger.info("need to check delta s %f",delta_s)


def opt_d_condition_by_time(previous_d_condition, min_compared_time, max_time):
    opt_d_contion = []
    for point in previous_d_condition:
        if min_compared_time <= point[3] <= max_time:
            opt_d_contion.append(point)
    return opt_d_contion


def opt_d_condition_by_s(previous_d_condition, deltas):
    opt_d_contion = []
    for point in previous_d_condition:
        if point[4] >= deltas:
            point[4] = point[4] - deltas
            opt_d_contion.append(point)
    return opt_d_contion


def CartesianToFrenet(path_point, traj_point):
    ''' from Cartesian to Frenet coordinate, to the matched path point
    copy Apollo cartesian_frenet_conversion.cpp'''
    rx, ry, rs, rtheta, rkappa, rdkappa = path_point.rx, path_point.ry, path_point.rs, \
        path_point.rtheta, path_point.rkappa, path_point.rdkappa
    x, y, v, a, theta, kappa = traj_point.x, traj_point.y, traj_point.v, \
        traj_point.a, traj_point.theta, traj_point.kappa

    s_condition = np.zeros(3)
    d_condition = np.zeros(3)

    dx = x - rx
    dy = y - ry

    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)

    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = math.copysign(math.sqrt(dx ** 2 + dy ** 2), cross_rd_nd)

    delta_theta = theta - rtheta
    tan_delta_theta = math.tan(delta_theta)
    cos_delta_theta = math.cos(delta_theta)

    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta

    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]

    d_condition[2] = -kappa_r_d_prime * tan_delta_theta + one_minus_kappa_r_d / (cos_delta_theta ** 2) * \
                     (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa)

    s_condition[0] = rs
    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d

    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    s_condition[2] = (a * cos_delta_theta - s_condition[1] ** 2 *
                      (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d

    return s_condition, d_condition


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
    if len(rtheta) > 2:
        rtheta[-1] = rtheta[-2]  # 最后一个时刻的角度没法求就直接等于倒数第二个时刻
        rkappa[:-1] = np.diff(rtheta) / np.diff(rs)  # 角度变化量/路程变化量
        rdkappa[:-1] = np.diff(rkappa) / np.diff(rs)
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
    if weight < 0 or weight > 1:
        print("weight error, not in [0, 1]")
        quit()
    rx_inter = lerp(path_point_0.rx, path_point_1.rx, weight)
    ry_inter = lerp(path_point_0.ry, path_point_1.ry, weight)
    rtheta_inter = slerp(path_point_0.rtheta, path_point_1.rtheta, weight)
    rkappa_inter = lerp(path_point_0.rkappa, path_point_1.rkappa, weight)
    rdkappa_inter = lerp(path_point_0.rdkappa, path_point_1.rdkappa, weight)
    return PathPoint([rx_inter, ry_inter, rs_inter, rtheta_inter, rkappa_inter, rdkappa_inter])


def TrajObsFree(xoy_traj, obstacle, delta_t):  ### 输入为路径点 障碍物类 帧长
    dis_sum = 0
    for point in xoy_traj:
        if isinstance(point, PathPoint):  # 如果是原来路径点，就只按圆形计算。因为每点的车辆方向难以获得
            ### isinstance 当参数1和参数2是同一类型时返回True
            if ColliTestRough(point, obstacle) > 0:  ### 返回point与obstacle的距离
                continue
            return 0, False
        else:
            dis = ColliTestRough(point,
                                 obstacle)  ### isinstance执行时是路径点与障碍物的距离(是否碰撞) 而else里point不再是路径点(与PathPoint不同类)因此是车辆的位置
            dis_sum += dis
            if dis > 0:
                continue
            if ColliTest(point, obstacle):  ### 对于车辆与障碍物是否碰撞 ColliTestRough不足以(将两者视为圆形) 要用更准确的ColliTest检测是否碰撞
                # print("不满足实际碰撞检测")
                return 0, False
    if len(xoy_traj) != 0:
        dis_mean = dis_sum / len(xoy_traj)
    else:
        return 0, False
    # print("满足实际碰撞检测")
    return dis_mean, True


# 粗略的碰撞检测(视作圆形)  如果此时不碰撞，就无需按矩形检测。返回的距离作为该点车到障碍物的大致距离（无碰撞时也可能为负）
def ColliTestRough(point, obs):
    if isinstance(point, PathPoint):
        dis = math.sqrt((point.rx - obs.x) ** 2 + (point.ry - obs.y) ** 2)
    else:
        dis = math.sqrt((point.x - obs.x) ** 2 + (point.y - obs.y) ** 2)
    max_veh = max(VEH_L, VEH_W)
    max_obs = max(obs.length, obs.width)
    return dis - (max_veh + max_obs) / 2


# 碰撞检测 (这部分参考apollo代码)
def ColliTest(point, obs):
    shift_x = obs.x - point.x
    shift_y = obs.y - point.y

    cos_v = math.cos(point.theta)
    sin_v = math.sin(point.theta)
    cos_o = math.cos(obs.heading)
    sin_o = math.sin(obs.heading)
    half_l_v = VEH_L / 2
    half_w_v = VEH_W / 2
    half_l_o = obs.length / 2
    half_w_o = obs.width / 2

    dx1 = cos_v * VEH_L / 2
    dy1 = sin_v * VEH_L / 2
    dx2 = sin_v * VEH_W / 2
    dy2 = -cos_v * VEH_W / 2
    dx3 = cos_o * obs.length / 2
    dy3 = sin_o * obs.length / 2
    dx4 = sin_o * obs.width / 2
    dy4 = -cos_o * obs.width / 2

    # 使用分离轴定理进行碰撞检测
    return ((abs(shift_x * cos_v + shift_y * sin_v) <=
             abs(dx3 * cos_v + dy3 * sin_v) + abs(dx4 * cos_v + dy4 * sin_v) + half_l_v)
            and (abs(shift_x * sin_v - shift_y * cos_v) <=
                 abs(dx3 * sin_v - dy3 * cos_v) + abs(dx4 * sin_v - dy4 * cos_v) + half_w_v)
            and (abs(shift_x * cos_o + shift_y * sin_o) <=
                 abs(dx1 * cos_o + dy1 * sin_o) + abs(dx2 * cos_o + dy2 * sin_o) + half_l_o)
            and (abs(shift_x * sin_o - shift_y * cos_o) <=
                 abs(dx1 * sin_o - dy1 * cos_o) + abs(dx2 * sin_o - dy2 * cos_o) + half_w_o))


# 对符合碰撞和约束限制的轨迹对进行cost排序，目前只保留了碰撞和横向两个cost ### 取cost min作为opt traj
def CostSorting(traj_pairs, destination):
    cost_dict = {}
    num = 0
    LAT_COST_WEIGHT, LON_COLLISION_COST_WEIGHT, DESTINATION_WEIGHT = 1, 1, 1
    for i in traj_pairs:  # traj_pairs [0]是poly_traj [1]是到障碍物的dis_mean
        traj = i[0]
        lat_cost = traj.lat_cost  # 横向偏移和横向加速度
        lon_collision_cost = -i[1]  # 碰撞风险：apollo的较复杂，这里直接用轨迹上各点到障碍物圆的平均距离表示 这里
        x = np.mean([tp.x for tp in traj.tp_all])
        y = np.mean([tp.y for tp in traj.tp_all])
        lon_destination_cost = math.sqrt((x - destination[0]) ** 2 + (y - destination[1]) ** 2)
        cost_dict[
            num] = lat_cost * LAT_COST_WEIGHT + lon_collision_cost * LON_COLLISION_COST_WEIGHT + lon_destination_cost * DESTINATION_WEIGHT
        num += 1
    cost_list_sorted = sorted(cost_dict.items(), key=lambda d: d[1], reverse=False)
    return cost_list_sorted  # [0]表示原编号num [1]表示该traj的损失函数值