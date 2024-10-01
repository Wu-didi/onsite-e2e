#pragma once
#include "map.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <sys/timeb.h>
#include <vector>

namespace localPlanner {

// 自定义数据结构
//需要输入的底盘反馈信息
struct VehicleFeedbackState {
  double steering_wheel_angle;       //底盘反馈的轮胎角度
  double brake_pedal_position;       // 刹车值
  double accelerator_pedal_position; //油门值
};
// 输出给底盘的控制指令
struct ControlCommand {
  double steer;    // 方向盘转角
  double speed;    // 速度
  double acc;      // 加速度
  double throttle; // 油门
  double brake;    // 刹车
};

struct Pointxy {
  double x;
  double y;
};

struct Pointsl {
  double s;
  double l;
};

struct Pose {
  double x;
  double y;
  double theta;
};

struct CubicBezierTrajectory {
  Pointxy p0;
  Pointxy p1;
  Pointxy p2;
  Pointxy p3;

  std::vector<Pose> poses;

  double maxk;
  double k0;
};

enum RoleType { VEHICLE, PEDESTRIAN, UNKNOWN };

struct TracePoint {
  zjlmap::SLZ rlsl;
  double x;
  double y;
  double z;
  double theta;
  double kappa;
  double speed;
  double acc;
};

struct Box2d {
  std::string id;
  RoleType roleType; //  -1:others; 0： 机动车； 1：行人； 2：非机动车；
                     //  3：动物； 4： 树； 5： 信号灯.

  bool is_static;
  bool is_virtual;

  double correct_speed_l;

  std::vector<TracePoint> prediction_points;

  zjlmap::SLZ rlsl;
  double x;
  double y;
  double z;
  double theta;
  double kappa;
  double speed;
  double acc;

  double width;
  double length;
};

struct Agent {
  Box2d box2d;
  size_t past_max_size;
  std::vector<TracePoint>
      past_points; // past infomation for intension and prediction
  size_t predict_max_size;
  std::vector<TracePoint> predict_points;
};

enum class LaneType { RIGHT, MIDDLE, LEFT };
enum class TurnType {
  RIGHT_TURN,
  STRAIGHT,
  LEFT_TURN,
  UTURN,
  STRAIGHT_LEFT,
  STRAIGHT_RIGHT,
  LEFT_UTURN,
  NO_TURN
};
enum class TargetType {
  LEFTCHANGING,
  LANEKEEPING,
  RIGHTCHANGING,
  LEFTPASSING,
  RIGHTPASSING,
  PULLOVER,
  PARKING
}; // for multi scenes
struct TargetNode {
  Pose pose;
  Pointsl sl; // s: length of target, l: offset base on lane center line
  double desired_speed;
  double lookaheaddist; // 需要做碰撞检测的轨迹长度，从起点开始计算
  zjlmap::SLZ rlsl;
  double driving_prob;
  LaneType lane_type;
  TargetType target_type;
};

enum class TrajectoryType { CANDIDATE, EXECUTABLE, OPTIMAL };

struct EvaTrajectory {
  size_t id;
  TargetNode target;
  CubicBezierTrajectory trajectory;
  double cost;
  double smoothcost;
  double loncost;
  double latcost;
  double collisioncost;
  bool isfeasible;
  TrajectoryType type;
};

struct CostCFG {
  // double w_safe;
  // double w_consist;
  double w_smooth;
  double w_lon;
  double w_lat;
  double w_collsion;
};

struct Trajectory {
  TargetNode target;
  CubicBezierTrajectory trajectory;
};

struct MiddleTarget {
  zjlmap::SLZ mt_sl;
  double s; // trajectory length
};

struct LMRLaneProb {
  double l;
  double m;
  double r;
}; // probability for driving

struct LaneNode {
  bool is_active;             // is lane useful?
  zjlmap::LaneInfo lane_info; // lane info
  TurnType turn_type;
  double ds;
  std::vector<zjlmap::TracePoint>
      lane_points_info; // start_s --- end_s  ds = 0.3
};

// ahead and back obstacles on a lane
struct ABObstacles {
  bool is_ahead_exit;
  Box2d ahead_obs;
  double ahead_dist;
  bool is_ahead_block_lane;

  bool is_back_exit;
  Box2d back_obs;
  double back_dist;
};

struct HLMRLaneGraph {
  // is lane useful?
  bool is_in_junction;
  Pointxy junction_xy;
  LaneNode L;
  LaneNode M;
  LaneNode R;
  LaneNode HL;
  LaneNode HM;
  LaneNode HR;
  LaneNode LC; // 左逆向车道
  std::vector<LaneNode> Ramp;
  Box2d robot;
  std::vector<Box2d> obstacles;
  LaneType next_lane_type;
  int change_num; // 未来需要连续变换车道的个数
                  // note: HR and HR aren't used yet
};

struct WayPoint {
  Pose pose;
  zjlmap::SLZ slz;
};

enum class SingleTargetStateCode { REACHING = 0, REACHED, UNREACHEABLE };

struct TaskStatus {
  int target_index;
  WayPoint current_target;
  SingleTargetStateCode state;
};

enum class PassengersStateCode { INCAR, OUTCAR };

struct PassengersStatus {
  PassengersStateCode passengers_state;
};

enum class EventType { ROLLED, STICKED, SLIPED };

struct HistoryInfo {
  zjlmap::SLZ event_slz;
  double prob;
  double speed_pass;
  double speed_limit;
  int event_type; // 0 ROLLED, 1 STICKED, 2 SLIPED, 3 BUMPED
  bool is_meet;
  bool is_active;
  bool is_pass;
};

// struct LocalView {
//   std::shared_ptr<Box2d> ugv;
//   std::shared_ptr<std::vector<Box2d> > obstacles;
//   std::shared_ptr<std::vector<localPlanner::TargetNode> > targets;
//   std::shared_ptr<std::vector<localPlanner::Trajectory> >
//   candidate_trajectories;
//   std::shared_ptr<std::vector<localPlanner::EvaTrajectory> >
//   evaluate_trajectories; std::shared_ptr<LocalView> pre_localview;
// }

struct ChassisState {
  double speed;
  double acc;
  double delta;
  double throttle;
  double brake;
  double cmd;
};

enum class RGYLightColorStatus : int {
  TLCS_BLACK = 0,         //信号灯灭
  TLCS_RED = 1,           //红灯
  TLCS_RED_FLASH = 2,     //红灯闪烁
  TLCS_GREEN = 4,         //绿灯
  TLCS_GREEN_FLASH = 8,   //绿灯闪烁
  TLCS_YELLOW = 16,       //黄灯
  TLCS_YELLOW_FLASH = 32, //黄灯闪烁
  TLCS_BROKEN = 63,       //所有灯都损坏
};

struct RGYLight {
  bool is_exist;
  RGYLightColorStatus color;
  int remain_time;
};

struct TrafficLight {
  std::string id;
  zjlmap::LaneId lane_id;
  zjlmap::ObjectId stopline_id;
  RGYLight left_light;
  RGYLight straight_light;
  RGYLight right_light;
  RGYLight uturn_light;
};

typedef struct _MulticastChannel {
  std::string config_center_addr; //配置中心
  std::string field_id;           //场地ID
  std::string net_interface_name; //网卡名称
  std::string net_client_name;    //客户端名称
  std::string net_ip;             //本地ip
} MulticastChannel;

} // namespace localPlanner
