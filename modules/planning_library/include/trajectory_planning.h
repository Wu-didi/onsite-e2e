#pragma once
#include "header.h"

namespace vts {
namespace planner {
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化轨迹规划器
 *
 * 这个函数用于初始化轨迹规划器，设置地图数据和延迟参数。
 *
 * @param map_path 地图文件的路径
 * @param latency 延迟参数，单位为秒
 *
 * @return 初始化是否成功，成功返回 true，否则返回 false
 */
bool VTS_Planner_Init(const std::string &map_path, double latency);

/**
 * @brief 处理局部路径规划的主要函数
 *
 * @param obstacles 障碍物集合，表示为Box2d类型
 * @param robot 无人车的当前状态，表示为Box2d类型
 * @param vehicle_feedback_state 车辆的反馈状态，包括传感器信息和车辆状态
 * @param control_command 规划算法输出给底盘的控制命令
 * @return bool 操作是否成功，返回true表示成功，false表示出现错误或规划失败
 */
bool VTS_Planner_Process(
    const std::vector<localPlanner::Box2d> &obstacles,
    const localPlanner::Box2d &robot,
    const localPlanner::VehicleFeedbackState &vehicle_feedback_state,
    localPlanner::ControlCommand &control_command);
/**
 * @brief 更新测试案例的目标点
 *
 * @param goal 目标点的姿态信息，包括位置和朝向
 */
void VTS_Planner_UpdateGoal(const localPlanner::Pose &goal);

/**
 * @brief 更新VTS 规划器的地图数据
 *
 * @param map_path 地图文件的路径
 * @return bool 更新是否成功，返回true表示成功，false表示更新失败或出现错误
 */
bool VTS_Planner_UpdataMapData(const std::string &map_path);

#ifdef __cplusplus
}
#endif

} // namespace planner
} // namespace vts
