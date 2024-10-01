from planner.utilsTools import *

class SampleBasis:
    # the basis of sampling: theta, dist, d_end (, v_end); normally for the planning_out cruising case
    def __init__(self, traj_point, theta_thr, ttcs, v_tgt):
        self.v_tgt = v_tgt
        traj_point.LimitTheta(theta_thr)
        self.theta_samp = [NormalizeAngle(traj_point.theta - theta_thr),
                           NormalizeAngle(traj_point.theta - theta_thr / 2),
                           traj_point.theta, NormalizeAngle(traj_point.theta + theta_thr / 2),
                           NormalizeAngle(traj_point.theta + theta_thr)]
        self.theta_samp = [traj_point.theta]
        planning_horizon = 2  # 2s
        # NormalizeAngle将角度转化为[-pi,pi] 角度的采样区间为原轨迹点theta下[-theta_thr,-theta_thr/2,0,
        # theta_thr/2,theta_thr]即最大转向角为theta_thr
        self.dist_samp = [self.v_tgt * ttc for ttc in ttcs]
        # self.dist_samp = [(traj_point.v + 0.5 * ttc) * planning_horizon for ttc in
        #                   ttcs]  ### 距离的采样区间为目标速度*ttcs区间(这里为3s 4s 5s)
        # for i in range(len(self.dist_samp)):
        #     if self.dist_samp[i] <= 0:
        #         self.dist_samp[i] = 0
        # self.dist_samp = [v_tgt*ttc for ttc in ttcs]
        self.dist_prvw = 5  # self.dist_samp[0]  # 最小的距离采样
        self.d_end_samp = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        self.total_t_samp = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        self.v_end = self.v_tgt  # v_tgt  # for cruising
