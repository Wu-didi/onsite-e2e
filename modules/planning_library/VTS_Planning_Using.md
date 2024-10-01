# VTS_Planning module
VTS 规控算法模块获取底盘的反馈信息，输出底盘的控制指令。  

# 整体结构
```
planning_library/
├── include
│   ├── header.h
│   ├── map.h
│   └── trajectory_planning.h
└── lib
    ├── libbehavior_planner.so
    ├── libcollision_detector.so
    ├── libcubic_bezier_curve.so
    ├── libinfo_preprocessor.so
    ├── libloncontroller.so
    ├── libtrajectory_evaluator.so
    ├── libtrajectory_generator.so
    ├── libtrajectory_planner.so
    ├── libvtslog.so
    └── libVTSMapInterfaceCPP.so
```
## 结构内容说明
### 头文件
- trajectory_planning.h 算法接口
- header.h 相关结构体说明   
- map.h 地图接口
### 动态库
- libtrajectory_planner.so 算法提供的动态库
- libvtslog.so 日志库
- libVTSMapInterfaceCPP.so 地图库 
- 算法依赖的其他动态库
# 接口说明
## 1 初始化接口
```
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
bool VTS_Planner_Init(const std::string &map_path,
                      double latency);
```
## 2 更新目标点接口
```
/**
 * @brief 更新测试案例的目标点
 *
 * @param goal 目标点的姿态信息，包括位置和朝向
 */
void VTS_Planner_UpdateGoal(const localPlanner::Pose &goal);
```
## 3 更新地图数据接口
```
/**
 * @brief 更新VTS 规划器的地图数据
 *
 * @param map_path 地图文件的路径
 * @return bool 更新是否成功，返回true表示成功，false表示更新失败或出现错误
 */
bool VTS_Planner_UpdataMapData(const std::string &map_path);
```
## 4 算法处理接口
```
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
```
# 使用方式
```
1 首次调用初始化接口
2 每次案例开始调用更新地图接口
2 然后调用更新目标点接口
3 测试中每帧调用Process处理接口
```
# 注意事项
```
1 初始化接口中的latency是循环周期，单位为s，建议周期为0.05(50ms），或0.1(100ms）
2 接口使用的接口体参考header.h
3 **更新地图接口调用在更新目标地接口之前**
```
# 数据结构的对应关系

## 反馈信号对应关系

| 规划器数据结构                                  | 仿真器数据结构                                               |
| ----------------------------------------------- | ------------------------------------------------------------ |
| VehicleFeedbackState.steering_wheel_angle       | protoVehicleFB.steering_feedback().steering_wheel_angle()    |
| VehicleFeedbackState.brake_pedal_position       | protoVehicleFB.brake_feedback().brake_pedal_position()       |
| VehicleFeedbackState.accelerator_pedal_position | protoVehicleFB.driving_feedback().accelerator_pedal_position() |
|                                                 |                                                              |
|                                                 |                                                              |

## 底盘输出信号对应关系

| 规划器数据结构          | 仿真器数据结构                                               |
| ----------------------- | ------------------------------------------------------------ |
| ControlCommand.steer    | protoVehicleControl.steering_control().target_steering_wheel_angle() |
| ControlCommand.brake    | protoVehicleControl.brake_control().target_brake_pedal_position() |
| ControlCommand.acc      | protoVehicleControl.acceleration()                           |
| ControlCommand.speed    | protoVehicleControl.speed()                                  |
| ControlCommand.throttle | protoVehicleControl.driving_control().target_accelerator_pedal_position() |

## 障碍物

```
参考 main.cpp  GetNewPubRole()
```

