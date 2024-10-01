## 1.主函数
```
from planner.lattice_v1 import LATTICE
# 实例化planner
planner = LATTICE()
# 初始化planner
planner.init(scenario_dict)
```

## 2.scenario_dict
构造：<br>
scenario_dict = 
{'num': 0, <br>
'name': '7_28_1_89', <br> 
'type': 'FRAGMENT', <br> 
'source_file': {'xodr': 'E:\\dev_code\\onsite\\scenario\\fragment\\7_28_1_89\\7_28_1_89.xodr', 'xosc': 'E:\\dev_code\\onsite\\scenario\\fragment\\7_28_1_89\\7_28_1_89_exam.xosc', 'json': '', 'tess': 'E:\\dev_code\\onsite\\scenario\\fragment\\7_28_1_89\\7_28_1_89.tess'}, <br> 
'output_path': 'E:\\dev_code\\onsite\\outputs\\FRAGMENT_0_7_28_1_89_result.csv', <br> 
'task_info': <br>
    {'startPos': [-19.998, 11.663], <br>
    'targetPos': [[49.501, 12.296], [59.501, 14.296]], <br> 
    'waypoints': [], 'dt': 0.1}} <br>
其中，num字段设置为0，name字段设置为空字符，type字段设置为空字符，source_file字段将xorc字段后的路径设置为该场景对应的地图文件即可，
output字段设置为空字符串，task_info中startPos是起点，targetPos是终点，waypoints若没有则设置为空列表，dt是单步时间帧（默认为10Hz）.

## 3. 调用planner主接口
`
[acc_target, wheel_target] = planner.act(observation)
`# 输入当前状态，输出纵向车速与前轮转角
### 3.1 observation
每一时间步需要更新感知到的观察量，具体描述如下：<br>
`
new_observation = Observation() # 实例化障碍物类
`<br>
ObjectStatus()： 障碍物状态类(new_observation.object_info)<br>
[x, y, v, a, yaw, width, length]分别是障碍物全局x坐标、y坐标、速度、加速度、宽度、长度；
提供了障碍物状态更新接口，使用方法如下：
```
new_observation.update_object_info(
                        'vehicle', # 障碍物类型
                        name, # 障碍物名称
                        length=obstacle_length, # 以m为单位
                        width=obstacle_width,  # 以m为单位
                        x=obstacle_x,
                        y=obstacle_y,
                        v=obstacle_v, # 障碍物速度，以m/s为单位
                        a=obstacle_a, # 障碍物加速度，以m/s^2为单位
                        yaw=obstacle_heading, # 障碍物航向角， 以rad为单位
                    )

```
EgoStatus： 主车状态类(new_observation.ego_info)<br>
[x, y, v, a, yaw, rot, length, width]分别是主车全局x坐标、y坐标、速度、加速度、前轮转角、长度、宽度；
提供了主车的状态更新接口，使用方法如下:
```
new_observation.update_ego_info(
            x=x_ego,
            y=y_ego,
            v=v_ego,
            a=a_ego,
            yaw=yaw_ego,
            rot=rot_ego,
            length=length_ego,
            width=width_ego
        )
```

