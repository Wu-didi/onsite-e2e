from planner.utilsTools import *
import math
import numpy as np
from planner.components.polyTraj import PolyTraj
from planner.components.obstacle import Obstacle
import csv
from planner.navigation.apfNavigation import ApfNavigation

# from utils.logger_lattice import lattice_logger


class LocalPlanner:
    def __init__(self, traj_point, path_points, obstacles, samp_basis, transformed_path, delta_time, current_time,
                 v_tgt, first_run, previous_d_condition, destination, scenario_type, delta_t):
        self.traj_point_theta = traj_point.theta  # record the current heading
        self.traj_point: TrajPoint = traj_point
        self.path_points = path_points
        self.obstacles = obstacles
        self.transformed_path = transformed_path
        self.theta_samp = samp_basis.theta_samp
        self.dist_samp = samp_basis.dist_samp
        self.d_end_samp = samp_basis.d_end_samp
        self.total_t_samp = samp_basis.total_t_samp
        self.v_end = samp_basis.v_end
        self.polytrajs = []
        self.nearest_obstacle_found = False
        self.backward_obstacle_found = False
        self.nearest_obstacles = []
        self.neareset_obs_x_in_ego_coor = 10000
        self.neareset_obs_y_in_ego_coor = 0
        self.nearest_back_obs_x_in_ego_coor = -10000
        self.neareset_back_obs_y_in_ego_coor = 0
        self.MAX_OBJ_DIS = 10000
        self.first_run = first_run
        self.b_use_sticher = True
        self.bug_plan = False
        self.lat_range = 3
        self.CSVLOG = True
        self.delta_t = delta_t
        self.sight_range = 100
        self.delta_time = delta_time
        self.current_time = current_time
        self.v_tgt = v_tgt
        self.poly_traj = None
        self.collision_free = None
        self.collision = False
        self.scenario_type = scenario_type
        self.previous_d = previous_d_condition
        self.nearest_obstacle = Obstacle(
            [self.MAX_OBJ_DIS, 0, 0, 0, 0, 0, 'static'])
        self.destination = destination
        self.__JudgeStatus(traj_point, path_points, obstacles, samp_basis)
        self.lat_range = 2
        # self.apf_ref = ApfNavigation(3.5, 4.5, traj_point, destination, obstacles)

    def __JudgeStatus(self, traj_point: TrajPoint, path_points, obstacles,
                      samp_basis):  # 赋值self.status和self.dist_prvw和self.to_Stop# 分别表示车辆的位置关系 最小的采样距离
        self.to_stop = False  # cruising
        self.dist_prvw = samp_basis.dist_prvw
        # todo: 根据场景划分to_stop距离
        if Dist(self.destination[0], self.destination[1], traj_point.x, traj_point.y) < 150.0:
            self.to_stop = True
        for obstacle in self.obstacles:
            # if obstacle.matched_point.rs < self.traj_point.matched_point.rs - 2:  ### 障碍物的match_point小于车辆当前match_point
            #     continue
            # if Dist(obstacle.x, obstacle.y, self.traj_point.x, self.traj_point.y) > sight_range:  ### 距离大于可视距离
            #     # 只看眼前一段距离
            #     continue
            angle = NormalizeAngle(self.traj_point.theta)  # angle 似乎不准确，需要检查

            x_scaled = obstacle.x - self.traj_point.x
            y_scaled = obstacle.y - self.traj_point.y

            # Apply rotation
            x_translated = x_scaled * math.cos(angle) + y_scaled * math.sin(angle)
            y_translated = -x_scaled * math.sin(angle) + y_scaled * math.cos(angle)

            # 判断前车
            if (100 > x_translated > 0 and abs(y_translated) < self.lat_range and
                    x_translated < self.neareset_obs_x_in_ego_coor):
                # and (obstacle.matched_point.rs - self.traj_point.matched_point.rs) > 0)
                self.nearest_obstacle: Obstacle = obstacle
                self.nearest_obstacle_found = True
                self.neareset_obs_y_in_ego_coor = y_translated
                self.neareset_obs_x_in_ego_coor = x_translated
                # self.nearest_obstacle = Obstacle()
                # print("self.neareset_obs_x_in_ego_coor==", self.neareset_obs_x_in_ego_coor)
            # 判断后车
            if (0 > x_translated > -100 and abs(y_translated) < 2 and
                    x_translated > self.nearest_back_obs_x_in_ego_coor):
                self.backward_obstacle_found = True
                self.backward_obstacle: Obstacle = obstacle
                self.nearest_back_obs_y_in_ego_coor = y_translated
                self.nearest_back_obs_x_in_ego_coor = x_translated
            if (abs(self.nearest_back_obs_x_in_ego_coor) < self.neareset_obs_x_in_ego_coor and
                    self.scenario_type == "REPLAY"):
                # print("有后车", self.nearest_back_obs_x_in_ego_coor, self.neareset_obs_x_in_ego_coor)
                self.nearest_obstacle_found = False
            # 用于计算邻近车辆速度
            if 100 > abs(x_translated):
                self.nearest_obstacles.append(obstacle)
        if self.nearest_obstacle_found:
            self.status = 'planning_out'  # 检测到障碍物
        if not self.nearest_obstacle_found:
            self.status = 'following_path'

        return

    def __LatticePlanner(self, traj_point: TrajPoint, path_points, obstacles, samp_basis):

        colli_free_traj_pairs = []  # PolyTraj object with corresponding trajectory's cost
        traj_pairs = []
        s_cond_init, d_cond_init = CartesianToFrenet(self.traj_point.matched_point,
                                                     self.traj_point)  ### 转化坐标系 s d分别速度方向和垂直于速度方向
        # lattice_logger.info("check inital d %f dl %f, ddl %f,traj point x %f, y %f, theta %f,
        # point x %f, y %f, theta %f", d_cond_init[0],\
        # d_cond_init[1],d_cond_init[2],self.traj_point.x, self.traj_point.y, self.traj_point.theta,
        # self.traj_point.matched_point.rx,self.traj_point.matched_point.ry, self.traj_point.matched_point.rtheta)
        if self.first_run:
            s_cond_init[2], d_cond_init[2] = 0, 0  # [0]为该坐标系下路程 [1]为速度
        if not self.first_run and self.b_use_sticher:
            # lattice_logger.info("check delta time %f, v %f, delta s %f", delta_time,
            # self.traj_point.v,delta_time*self.traj_point.v)
            if self.traj_point.v > 0:
                # todo
                last_point = find_last_d_condition_by_s(self.previous_d,
                                                        self.delta_time * self.traj_point.v)
                if last_point is not None:
                    d_cond_init[0] = last_point.l
                    d_cond_init[1] = last_point.dl
                    d_cond_init[2] = last_point.ddl
                else:
                    d_cond_init[0] = 0
                    d_cond_init[1] = 0
                    d_cond_init[2] = 0

        if self.nearest_obstacle_found:
            total_t = 3.5
            # lattice_logger.info("此时发现最近的障碍物，total_t设置为6s")
            # lattice_logger.info("此时障碍物距离为%f", self.nearest_obstacle.matched_point.rs - s_cond_init[0])
            # total_t = 3.5
            v_target = self.nearest_obstacle.v
            safe_dis = 20
            # 距离非常近，小于10m且在自车前方
            # if 0 < self.nearest_obstacle.matched_point.rs - s_cond_init[0] < 10:
            #     total_t = 2
            delta_s = v_target * total_t
            # todo
            obj_dis = traj_point.matched_point.rs + self.neareset_obs_x_in_ego_coor - safe_dis + delta_s  # 恒速度模型预测
            # # 最近障碍物在主车后方
            # if self.nearest_obstacle.matched_point.rs - s_cond_init[0] < 0:
            #     print("++++++++++")
            #     v_target = self.nearest_obstacle.v + 1.5
            #     obj_dis = traj_point.matched_point.rs - self.neareset_obs_x_in_ego_coor - safe_dis +
            #     delta_s # 恒速度模型预测
            s_desired = max(obj_dis, traj_point.matched_point.rs + 10)
            s_cond_end = np.array([s_desired, v_target, 0])  # v_end = v_tgt
            # lattice_logger.info("obj_dis % d,s_desired % d, obstacle_v % d", obj_dis, s_desired,
            #                     self.nearest_obstacle.v)
            # lattice_logger.info("s_desired %f, check front dis %f,front obj v %f,ego speed %f",s_desired ,
            # self.nearest_obstacle.matched_point.rs - s_cond_init[0],self.nearest_obstacle.v,traj_point.v)

            if self.to_stop:
                # lattice_logger.info("stop point s %f, x %f",path_points[-1].rs - s_cond_init[0],
                # transformed_path[-1][0])
                final_stop_s = max(traj_point.matched_point.rs, path_points[-1].rs) - 20
                s_cond_end = np.array([final_stop_s, 0, 0])
                # print("final s ",final_stop_s - traj_point.matched_point.rs, "ego speed",traj_point.v,
                # " a ",s_cond_init[2])
                total_t = 3
            self.poly_traj = PolyTraj(s_cond_init, d_cond_init, total_t, self.current_time)
            self.poly_traj.GenLongTraj(s_cond_end)  # GenLongTraj GenLatTraj分别得到五次多项式的系数
            if not self.poly_traj.LongConsFree(
                    self.delta_t):  # 先看纵向轨迹s是否满足纵向运动约束 not poly_traj.LongConsFree(delta_t) #后续检查
                pass
            else:
                d_end = 0
                d_cond_end = np.array([d_end, 0, 0])
                self.poly_traj.GenLatTraj(d_cond_end)
                # 生成规划轨迹 tp_all是笛卡尔坐标系 完成由frenet到笛卡尔转换
                tp_all = self.poly_traj.GenCombinedTraj(self.path_points,
                                                        self.delta_t)
                self.polytrajs.append(self.poly_traj)
                dis_to_obs = 0
                for obstacle in obstacles:
                    if obstacle.matched_point.rs < self.traj_point.matched_point.rs - 2:
                        continue
                    if Dist(obstacle.x, obstacle.y, traj_point.x, traj_point.y) > self.sight_range:
                        # 只看眼前一段距离
                        continue
                    self.collision_free = TrajObsFree(tp_all, obstacle, self.delta_t)
                    if not self.collision_free[1]:
                        self.collision = True
                        break
                    dis_to_obs += self.collision_free[0]
                if not self.collision:
                    colli_free_traj_pairs.append([self.poly_traj, dis_to_obs])
                traj_pairs.append([self.poly_traj, dis_to_obs])
                # todo: 若该轨迹有碰撞，重规划靠边停车轨迹
                if self.collision and self.bug_plan:
                    tp_all = self.__PlanningBug(traj_point, s_cond_init, d_cond_init)
                return tp_all
        # todo: 在replay场景下考虑
        elif self.scenario_type == "REPLAY" and self.backward_obstacle_found:
            total_t = 3.5
            v_target = self.backward_obstacle.v + 1
            safe_dis = 20
            # if abs(self.backward_obstacle.matched_point.rs - s_cond_init[0]) < 10:
            #     total_t = 2
            delta_s = v_target * total_t
            obj_dis = (traj_point.matched_point.rs - self.neareset_obs_x_in_ego_coor - safe_dis +
                       delta_s)  # 恒速度模型预测
            s_desired = max(obj_dis, traj_point.matched_point.rs + 10)
            s_cond_end = np.array([s_desired, v_target, 0])
            self.poly_traj = PolyTraj(s_cond_init, d_cond_init, total_t, self.current_time)
            self.poly_traj.GenLongTraj(s_cond_end)  # GenLongTraj GenLatTraj分别得到五次多项式的系数
            if not self.poly_traj.LongConsFree(
                    self.delta_t):  # 先看纵向轨迹s是否满足纵向运动约束 not poly_traj.LongConsFree(delta_t) #后续检查
                pass
            else:
                d_end = 0
                d_cond_end = np.array([d_end, 0, 0])
                self.poly_traj.GenLatTraj(d_cond_end)
                # 生成规划轨迹 tp_all是笛卡尔坐标系 完成由frenet到笛卡尔转换
                tp_all = self.poly_traj.GenCombinedTraj(self.path_points,
                                                        self.delta_t)
                self.polytrajs.append(self.poly_traj)
                for obstacle in obstacles:
                    if obstacle.matched_point.rs < self.traj_point.matched_point.rs - 2:
                        continue
                    if Dist(obstacle.x, obstacle.y, traj_point.x, traj_point.y) > self.sight_range:
                        # 只看眼前一段距离
                        continue
                    self.collision_free = TrajObsFree(tp_all, obstacle, self.delta_t)
                    if not self.collision_free[1]:
                        self.collision = True
                        break
                # todo: 若该轨迹有碰撞，重规划靠边停车轨迹
                if self.collision and self.bug_plan:
                    if self.bug_plan:
                        tp_all = self.__PlanningBug(traj_point, s_cond_init, d_cond_init)
                return tp_all
        else:  # no object
            # lattice_logger.info("没有发现障碍物")
            total_t = 4.0
            # todo: 目标车速应该为周车平均速度，若可观察范围内没有障碍物，那么设置为：if highway: v_tgt = 25; else: 15
            v_target = self.GetObastaclesVelocity()  # max(traj_point.v - total_t * 2 ,16)
            s_desired = v_target * total_t + s_cond_init[0]
            s_cond_end = np.array([s_desired, v_target,
                                   0])  ### traj_point.matched_point.rs + (v_tgt + traj_point.v) / 2.0 * total_t v_end = v_tgt
            if self.to_stop:
                # lattice_logger.info("stop point s %f",path_points[-1].rs - s_cond_init[0])
                final_stop_s = max(traj_point.matched_point.rs, path_points[-1].rs) - 20
                s_cond_end = np.array([final_stop_s, 0, 0])
                total_t = 3

            self.poly_traj = PolyTraj(s_cond_init, d_cond_init, total_t, self.current_time)  ##### 报错由于total+t =0
            # lattice_logger.info("no front obj ,  ego speed %f, total t %f",traj_point.v, total_t)
            # print("no front obj ego speed ",traj_point.v ," total t ", total_t)
            self.poly_traj.GenLongTraj(s_cond_end)  ### GenLongTraj GenLatTraj分别得到五次多项式的系数
            if not self.poly_traj.LongConsFree(
                    self.delta_t):  # 先看纵向轨迹s是否满足纵向运动约束 not poly_traj.LongConsFree(delta_t)
                pass
            else:
                d_end = 0
                d_cond_end = np.array([d_end, 0, 0])
                self.poly_traj.GenLatTraj(d_cond_end)
                tp_all = self.poly_traj.GenCombinedTraj(self.path_points,
                                                        self.delta_t)  ### 生成规划轨迹 tp_all是笛卡尔坐标系 完成由frenet到笛卡尔转换
                for obstacle in obstacles:
                    if obstacle.matched_point.rs < self.traj_point.matched_point.rs - 2:
                        continue
                    if Dist(obstacle.x, obstacle.y, traj_point.x, traj_point.y) > self.sight_range:
                        # 只看眼前一段距离
                        continue
                    self.collision_free = TrajObsFree(tp_all, obstacle, self.delta_t)
                    if not self.collision_free[1]:
                        self.collision = True
                        break
                # todo: 若该轨迹有碰撞，重规划靠边停车轨迹
                # print("self.collision: ", self.collision)
                if self.collision:
                    if self.bug_plan:
                        tp_all = self.__PlanningBug(traj_point, s_cond_init, d_cond_init)
                return tp_all

    def __PathFollower(self, traj_point, path_points, obstacles, samp_basis):  # 无障碍物且在原轨迹上时的循迹,认为从matched_point开始
        global delta_t
        # print(f'v_end:{self.v_end},traj_point.v:{self.traj_point.v},dist_prvw:{self.dist_prvw},delta_t:{delta_t}')
        acc = ((self.v_end ** 2 - self.traj_point.v ** 2) / (
                2 * self.dist_prvw) + 10e-10)  ### 以此加速度向v_end靠拢(在to_stop=False时=v_tgt)
        if self.dist_prvw < 2:  ### 到2m还没停下 管不了舒适减速度了直接刹死
            acc = -3 * self.traj_point.v
        total_t = 2 * self.dist_prvw / (self.v_end + self.traj_point.v)
        num_points = math.floor(total_t / self.delta_t)
        tp_all = []  # all the future traj points in a list
        rs_pp_all = []  # the rs value of all the path points
        tp_x = []
        tp_y = []
        for path_point in path_points:
            rs_pp_all.append(path_point.rs)
        rs_pp_all = np.array(rs_pp_all)
        for i in range(int(num_points)):
            s_cond = np.zeros(3)
            d_cond = np.zeros(3)
            s_cond[0] = (self.traj_point.matched_point.rs
                         + self.traj_point.v * i * self.delta_t + (1 / 2) * acc * ((i * self.delta_t) ** 2))
            s_cond[1] = self.traj_point.v + acc * i * self.delta_t
            s_cond[2] = acc  ### 此时的路程 速度 加速度 由于沿着参考轨迹行驶 所以在frenet下没有横向数据
            index_min = np.argmin(np.abs(rs_pp_all - s_cond[0]))
            path_point_min = path_points[index_min]
            if index_min == 0 or index_min == len(path_points) - 1:
                path_point_inter = path_point_min
            else:
                if s_cond[0] >= path_point_min.rs:
                    path_point_next = path_points[index_min + 1]
                    path_point_inter = LinearInterpolate(path_point_min, path_point_next, s_cond[0])
                else:
                    path_point_last = path_points[index_min - 1]
                    path_point_inter = LinearInterpolate(path_point_last, path_point_min, s_cond[0])

            traj_point = FrenetToCartesian(path_point_inter, s_cond, d_cond)
            tp_all.append(traj_point)
            tp_x.append(traj_point.x)
            tp_y.append(traj_point.y)
        return tp_all  ### 同样转化回笛卡尔坐标系

    def __FollowingPath(self, traj_point, path_points, obstacles, samp_basis):
        if traj_point.v < 0.5:  ### 如果处于刚起步状态 特别是v几乎等于0时 采样dis_sample很容易为[0,....] 走不动道了就 所以跟wait一样给赋一个dis_sample 渡过起步的难关 这样ego初速度也能为0了 不然之前ego初速度不能设为0.05以下
            self.dist_samp = [0.2 * self.v_tgt, 0.5 * self.v_tgt, self.v_tgt, 2 * self.v_tgt]
        if self.to_stop:
            self.v_end = 0  ### else v_end = v_tgt
        return self.__LatticePlanner(traj_point, path_points, obstacles, samp_basis)

    def __PlanningOut(self, traj_point, path_points, obstacles, samp_basis):
        if self.to_stop:  # stopping
            # self.dist_samp = [self.dist_prvw]
            self.v_end = 0
        return self.__LatticePlanner(traj_point, path_points, obstacles, samp_basis)  ### 规划出损失最小的轨迹

    def __PlanningBack(self, traj_point, path_points, obstacles, samp_basis):
        self.theta_samp = [self.traj_point_theta]  # just use the current heading, is it necessary?
        self.dist_samp = [self.dist_prvw]  # come back asap, is it necessary?
        self.d_end_samp = [0]
        if self.to_stop:  # stopping
            self.v_end = 0
        return self.__LatticePlanner(traj_point, path_points, obstacles, samp_basis)

    def __PlanningStop(self, traj_point, path_points, obstacles, samp_basis):
        self.dist_samp = [0, 0.5 * self.dist_prvw, self.dist_prvw]
        self.v_end = 0
        return self.__LatticePlanner(traj_point, path_points, obstacles, samp_basis)

    def LocalPlanning(self, traj_point, path_points, obstacles, samp_basis):
        # lattice_logger.info("status info:"+self.status)
        if self.status == "following_path":  ### 在参考轨迹上且无碰撞的风险
            return self.__FollowingPath(traj_point, path_points, obstacles, samp_basis)
        elif self.status == "planning_out":  ### 与障碍物有冲突 需要离开参考轨迹
            return self.__PlanningOut(traj_point, path_points, obstacles, samp_basis)
        elif self.status == "planning_back":  ### 无冲突且不在参考轨迹上 要回到参考轨迹
            return self.__PlanningBack(traj_point, path_points, obstacles, samp_basis)
        elif self.status == 'wait':
            return self.__PlanningStop(traj_point, path_points, obstacles, samp_basis)
        elif self.status == 'brake':  # 除brake外都是lattice规划器
            self.v_end = 0
            return self.__PathFollower(traj_point, path_points, obstacles, samp_basis)
        else:
            quit()

    def GetObastaclesVelocity(self):
        """
        计算邻近车辆的平均速度作为主车目标速度
        """
        v_list = []
        if len(self.nearest_obstacles) >= 1:
            for obs in self.nearest_obstacles:
                v_list.append(obs.v)
            mean_v = sum(v_list) / (len(self.nearest_obstacles) + 1)
        else:
            if self.scenario_type == "REPLAY":
                mean_v = 25
            else:
                mean_v = 35
        return mean_v

    # def __PlanningBug(self, traj_point, s_init_condition, d_init_condition, ):

    def __PlanningBug(self, traj_point: TrajPoint, s_cond_init, d_cond_init):
        total_t = 3
        # 如果主车靠近道路左边界，靠左停车
        desired_s = 20
        s_cond_end = np.array([desired_s, 0, 0])
        desired_d = -4
        d_cond_end = np.array([desired_d, 0, 0])
        self.poly_traj = PolyTraj(s_cond_init, d_cond_init, total_t, self.current_time)
        self.poly_traj.GenLongTraj(s_cond_end)  # GenLongTraj GenLatTraj分别得到五次多项式的系数
        self.poly_traj.GenLatTraj(d_cond_end)
        tp_all = self.poly_traj.GenCombinedTraj(self.path_points,
                                                self.delta_t)
        return tp_all
