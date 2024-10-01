from planner.utilsTools import *
from planner.components.dCondition import D_conditon
from typing import List

class PolyTraj:
    def __init__(self, s_cond_init, d_cond_init, total_t, current_time):
        self.tp_all = None
        self.lat_cost = 0
        self.lat_coef = None
        self.long_coef = None
        self.s_cond_init = s_cond_init
        self.d_cond_init = d_cond_init
        self.total_t = total_t  # to plan how long in seconds
        self.delta_s = 0
        self.max_v = 36
        self.min_v = 0
        self.max_a = 3
        self.min_a = -3
        self.LAT_COMFORT_COST_WEIGHT = 1
        self.LAT_OFFSET_COST_WEIGHT = 1
        self.previous_d_condition: List[D_conditon] = []
        self.current_time = current_time

    def __QuinticPolyCurve(self, y_cond_init, y_cond_end, x_dur):  ### 五次多项式拟合
        '''
        form the quintic polynomial curve: y(x) = a0 + a1 * delta_x + ... + a5 * delta_x ** 5, x_dur = x_end - x_init
        y_cond = np.array([y, y', y'']), output the coefficients a = np.array([a0, ..., a5])
        '''
        a0 = y_cond_init[0]
        a1 = y_cond_init[1]
        a2 = 1.0 / 2 * y_cond_init[2]
        T = x_dur
        if T != 0:
            h = y_cond_end[0] - y_cond_init[0]
            v0 = y_cond_init[1]
            v1 = y_cond_end[1]
            acc0 = y_cond_init[2]
            acc1 = y_cond_end[2]
            # print(x_dur)
            with warnings.catch_warnings(record=True) as w:
                a3 = 1.0 / (2 * T ** 3) * (20 * h - (8 * v1 + 12 * v0) * T - (3 * acc0 - acc1) * T ** 2)
                if any(issubclass(warning.category, RuntimeWarning) for warning in w):
                    pass
                    # lattice_logger.info("T % s, h % s, v1 % s, v0 % s, acc0 % s, acc1 % s",
                    #                     T, h, v1, acc0, acc1)
                a4 = 1.0 / (2 * T ** 4) * (-30 * h + (14 * v1 + 16 * v0) * T + (3 * acc0 - 2 * acc1) * T ** 2)
                a5 = 1.0 / (2 * T ** 5) * (12 * h - 6 * (v1 + v0) * T + (acc1 - acc0) * T ** 2)
        else:  # 有时由于delta_s(采样距离=0) 导致total_t = delta_s/v_tgt 使T=0
            a3 = 0
            a4 = 0
            a5 = 0
        return np.array([a0, a1, a2, a3, a4, a5])

    def GenLongTraj(self, s_cond_end):
        self.long_coef = self.__QuinticPolyCurve(self.s_cond_init, s_cond_end,
                                                 self.total_t)  ### self.long_coef为五次多项式的参数
        self.delta_s = self.long_coef[1] * self.total_t + self.long_coef[2] * self.total_t ** 2 + \
                       self.long_coef[3] * self.total_t ** 3 + self.long_coef[4] * self.total_t ** 4 + \
                       self.long_coef[5] * self.total_t ** 5
        # return self.long_coef

    def GenLatTraj(self, d_cond_end):
        # GenLatTraj should be posterior to GenLongTraj
        self.lat_coef = self.__QuinticPolyCurve(self.d_cond_init, d_cond_end, self.delta_s)  # s_desired
        # return self.lat_coef

    # 求各阶导数
    def Evaluate(self, coef, order, t):
        if order == 0:
            return ((((coef[5] * t + coef[4]) * t + coef[3]) * t
                     + coef[2]) * t + coef[1]) * t + coef[0]
        if order == 1:
            return (((5 * coef[5] * t + 4 * coef[4]) * t + 3 *
                     coef[3]) * t + 2 * coef[2]) * t + coef[1]
        if order == 2:
            return (((20 * coef[5] * t + 12 * coef[4]) * t)
                    + 6 * coef[3]) * t + 2 * coef[2]
        if order == 3:
            return (60 * coef[5] * t + 24 * coef[4]) * t + 6 * coef[3]
        if order == 4:
            return 120 * coef[5] * t + 24 * coef[4]
        if order == 5:
            return 120 * coef[5]

    # 纵向速度&加速度约束
    def LongConsFree(self, delta_t):
        size = int(self.total_t / delta_t)
        for i in range(size):
            v = self.Evaluate(self.long_coef, 1, i * delta_t)
            # print(v)
            cnt = 0
            if v > self.max_v or v < self.min_v:
                if i * delta_t < 1.0:  # 1s 之内check 速度的限制 后面进行优化
                    # print(v, "纵向速度超出约束", " time is ", i* delta_t)
                    cnt += 1
                    self.total_t += 0.5
                    # lattice_logger.info("纵向速度超出约束v %f, time %f",v, i*delta_t)
                    if cnt == 3:
                        return False
            # 纵向加速度约束
            cnt = 0
            a = self.Evaluate(self.long_coef, 2, i * delta_t)
            if a > self.max_a or a < self.min_a:
                # print("纵向加速度超出约束")
                cnt += 1
                self.total_t += 0.5
                # lattice_logger.info("纵向加速度超出约束a %f, time %f", a, i*delta_t)
                if cnt == 3:
                    return False
        return True

    # 横向加速度约束，参考apollo。这里把横向的cost一块算了
    # 横向偏移量和横向加速度cost同样参考apollo，数学上做了一些简化，如省略了偏移量绝对值，只计算平方；忽略和起点之间的偏移量关系等
    def LatConsFree(self, delta_t):
        size = int(self.total_t / delta_t)
        lat_offset_cost = 0
        lat_comfort_cost = 0
        global LAT_COMFORT_COST_WEIGHT, LAT_OFFSET_COST_WEIGHT
        for i in range(size):
            s = self.Evaluate(self.long_coef, 0, i * delta_t)
            d = self.Evaluate(self.lat_coef, 0, s)
            dd_ds = self.Evaluate(self.lat_coef, 1, s)
            ds_dt = self.Evaluate(self.long_coef, 1, i * delta_t)
            d2d_ds2 = self.Evaluate(self.lat_coef, 2, s)
            d2s_dt2 = self.Evaluate(self.long_coef, 2, i * delta_t)

            lat_a = d2d_ds2 * ds_dt * ds_dt + dd_ds * d2s_dt2
            '''
            向心加速度暂时删去
            if abs(lat_a) > MAX_LAT_A:
                print(lat_a, "不满足横向约束")
                return False
            '''
            lat_comfort_cost += lat_a * lat_a
            lat_offset_cost += d * d

        self.lat_cost = lat_comfort_cost * self.LAT_COMFORT_COST_WEIGHT + lat_offset_cost * self.LAT_OFFSET_COST_WEIGHT
        # print("满足横向约束")
        return True

    def GenCombinedTraj(self, path_points, delta_t):
        '''
        combine long and lat traj together
        F2C function is used to output future traj points in a list to follow
        '''
        a0_s, a1_s, a2_s, a3_s, a4_s, a5_s = self.long_coef[0], self.long_coef[1], self.long_coef[2], \
            self.long_coef[3], self.long_coef[4], self.long_coef[5]
        a0_d, a1_d, a2_d, a3_d, a4_d, a5_d = self.lat_coef[0], self.lat_coef[1], self.lat_coef[2], \
            self.lat_coef[3], self.lat_coef[4], self.lat_coef[5]

        rs_pp_all = []  # the rs value of all the path points
        for path_point in path_points:
            rs_pp_all.append(path_point.rs)
        rs_pp_all = np.array(rs_pp_all)
        num_points = math.floor(self.total_t / delta_t)  ### 规划时长/帧长 = 规划点数
        s_cond_all = []  # possibly useless
        d_cond_all = []  # possibly useless
        pp_inter = []  # possibly useless
        tp_all = []  # all the future traj points in a list
        t, s = 0, 0  # initialize variables, s(t), d(s) or l(s)
        self.previous_d_condition.clear()
        if len(self.previous_d_condition) > 0:
            t_start = self.previous_d_condition[-1][3]
        else:
            t_start = self.current_time
        for i in range(int(num_points)):
            s_cond = np.zeros(3)
            d_cond = np.zeros(3)

            t = t + delta_t
            s_cond[0] = a0_s + a1_s * t + a2_s * t ** 2 + a3_s * t ** 3 + a4_s * t ** 4 + a5_s * t ** 5  # 路程
            s_cond[1] = a1_s + 2 * a2_s * t + 3 * a3_s * t ** 2 + 4 * a4_s * t ** 3 + 5 * a5_s * t ** 4  # 速度(d路程/dt)
            s_cond[2] = 2 * a2_s + 6 * a3_s * t + 12 * a4_s * t ** 2 + 20 * a5_s * t ** 3  # a
            # lattice_logger.info("check t %f, s %f, v %f, a %f",  t,s_cond[0], s_cond[1], s_cond[2])
            s_cond_all.append(s_cond)

            s = s_cond[0] - a0_s
            with warnings.catch_warnings(record=True) as w:
                d_cond[0] = a0_d + a1_d * s + a2_d * s ** 2 + a3_d * s ** 3 + a4_d * s ** 4 + a5_d * s ** 5
                if d_cond[0] > 10e50:
                    d_cond[0] = 0
                if any(issubclass(warning.category, RuntimeWarning) for warning in w):
                    pass
                    # lattice_logger.info("ao_d % s, a1_d % s, s % s, a3_d % s, a4_d % s, a5_d % s",
                    #                     a0_d, a1_d, s, a3_d, a4_d, a5_d)
                d_cond[1] = a1_d + 2 * a2_d * s + 3 * a3_d * s ** 2 + 4 * a4_d * s ** 3 + 5 * a5_d * s ** 4
                if d_cond[1] > 10e50:
                    d_cond[1] = 0
                d_cond[2] = 2 * a2_d + 6 * a3_d * s + 12 * a4_d * s ** 2 + 20 * a5_d * s ** 3
                if d_cond[2] > 10e50:
                    d_cond[2] = 0
                d_cond_all.append(d_cond)

            d_point = D_conditon()
            d_point.l = d_cond[0]
            d_point.dl = d_cond[1]
            d_point.ddl = d_cond[2]
            d_point.time = t + t_start
            d_point.s = s

            # lattice_logger.info("d0 %f, d1 %f, d2 %f, t%f , s %f", d_cond[0], d_cond[1],d_cond[2],t+t_start,s)
            self.previous_d_condition.append(d_point)
            index_min = np.argmin(np.abs(rs_pp_all - s_cond[0]))
            path_point_min = path_points[index_min]  # 现在到哪个位置了
            if index_min == 0 or index_min == len(path_points) - 1:
                path_point_inter = path_point_min
            else:
                if s_cond[0] >= path_point_min.rs:
                    path_point_next = path_points[index_min + 1]
                    path_point_inter = LinearInterpolate(path_point_min, path_point_next, s_cond[0])
                else:
                    path_point_last = path_points[index_min - 1]
                    path_point_inter = LinearInterpolate(path_point_last, path_point_min, s_cond[0])
            pp_inter.append(path_point_inter)
            traj_point = FrenetToCartesian(path_point_inter, s_cond, d_cond)
            traj_point.time = t
            traj_point.s = s_cond[0] - a0_s
            # traj_point.v = v_tgt
            tp_all.append(traj_point)
        self.tp_all = tp_all
        # print("++++++", self.previous_d_condition[0].l)
        return tp_all

    def GetPreviousDCondition(self):
        return self.previous_d_condition
