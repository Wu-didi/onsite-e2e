import math
import os
import sys
import time

from mmdet3d.apis import inference_detector, init_model
import numpy as np

pwd = sys.path[0]
sys.path.insert(0, os.path.dirname(pwd))
from modules.planning_library.planner import (Box2d, ControlCommand, Pose,
                                              RoleType, VehicleFeedbackState,
                                              VTS_Planner_Init,
                                              VTS_Planner_Process,
                                              VTS_Planner_UpdataMapData,
                                              VTS_Planner_UpdateGoal)
from modules.tracking.tracker import AB3DMOT, BBox2DataFrame

from ultralytics import YOLO
import cv2
import logging

logger = logging.getLogger(__name__)

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def GetYawfromQuaternion(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# 相机焦距
foc  = 735.059326
# 预设置行人高度inch
real_hight_person_inch = 66.9
# 预设置车辆高度inch
real_hight_car_inch = 57.08
# traffic-cone
real_hight_cone_inch = 10
# bus
real_hight_bus_inch = 144
# truck
real_hight_truck_inch = 144


# 单目测量距离，通过相似三角形
def get_distance(real_hight_inch, h):
    '''返回真的distance'''
    dis_inch = real_hight_inch * foc / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def pixel_to_camera_coords(x, y, z):  
    """  
    将图像坐标系下的像素坐标（x, y）和深度z（单位为m）转换为相机坐标系下的坐标（Xc, Yc, Zc）。  
  
    参数:  
    x, y (int): 图像坐标系下的像素坐标。  
    z (float): 深度值，即相机坐标系下的Z坐标（单位为m）。  
    fx, fy (float): 相机内参矩阵的焦距。  
    cx, cy (float): 相机内参矩阵的主点坐标。  
  
    返回:  
    Xc, Yc, Zc (float): 相机坐标系下的三维坐标。  
    """  
    
    # 相机内参  
    fx = 733.614441  
    fy = 735.059326  
    cx = 604.237122  
    cy = 339.834991  
    
    
    # 将像素坐标转换为归一化相机坐标  
    xc = (x - cx) / fx  
    yc = (y - cy) / fy  
    # 归一化相机坐标的Z分量就是输入的深度z  
    zc = z  
    # 输出相机坐标系下的三维坐标  
    Xc = xc * zc  
    Yc = yc * zc  
    Zc = zc  
    return Xc, Yc, Zc  

# 定义一个绘制矩形框的函数，并将坐标值和类别值传入，写在image上
def draw_rectangle(image, x, y, x2, y2, class_label):
    cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return image

# 定义一个显示图像的函数，可以根据传入的图像显示出来，显示视频的时候可以用到
def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(1)



# only for lidar, 实现了从Lidar坐标系到车身坐标系的平移转换，没有实现旋转转换
def transform(xyz):
    """
    lidar坐标系转换为车身坐标系
    """
    # lidar extrinsics，可以是其他值，参考simulator的传感器配置
    transform_coeff = {
        "x": 0.1,
        "y": 0.013,
        "z": 1.755,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
    }
    xyz[0] -= transform_coeff["x"]
    xyz[1] -= transform_coeff["y"]
    xyz[2] -= transform_coeff["z"]
    return xyz

class Predictor_image:
    def __init__(self):
        # Load a model
        self.YOLO_model = YOLO('/home/wudi/python_files/onsite/0519update/e2e/perception_model/yolov8/onsite_v1.0/best.pt')  # load an official model~
        # self.tracker = AB3DMOT(ID_init=1)
        self.default_map = "AITownReconstructed_V0103_200518.xodr"
        default_map_path = os.path.join(os.path.dirname(__file__), "maps", self.default_map)
        if not VTS_Planner_Init(default_map_path, 0.1):
            raise RuntimeError
        self.ego = None
        self.vehicle_feedback = VehicleFeedbackState()
        self.vehicle_feedback.steering_wheel_angle = 0.0
        self.vehicle_feedback.brake_pedal_position = 0.0
        self.vehicle_feedback.accelerator_pedal_position = 0.0
        self.id2class = {0: 'person', 1: 'car', 2: 'bus', 3: 'truck', 4: 'traffic-cone'}

    def process_image_msg(self, images):
        # print(len(images))
        # for image in images: # three images 
            # 
        img = images[0].data.astype(np.uint8).reshape(720, 1280, 3)
        t0 = time.time()
        result = self.YOLO_model.predict(img, conf=0.3, iou=0.5)
        
        # logging.info("after remove low score box3ds: %s", box3ds)
        t1 = time.time()
        # print("inference time: ", t1 - t0)

        result = self.process_yolo_result(result)
        obstacles = self.process_pubrole_image(result)
        # print("obstacles: ", obstacles)
        return obstacles

    def process_yolo_result(self, results):
        result_boxes = []
        for r in results:
            
            if r.boxes is None:
                continue
            
            # 这里的results是以图像维度，表示长度，每一个r表示一个image的
            for single_box in r.boxes:
                coords_carm = None
                
                # Check if the center of the box is within the specified rectangular region
                if 100 <= single_box.xywh[0][0].cpu().numpy() <= 1200 and \
                    630 <= single_box.xywh[0][1].cpu().numpy()  <= 700:
                    continue  # Ignore the box if its center is within the specified region
                if single_box.cls == 0:
                    # print('detect is person')
                    # 获取person pix high
                    x = single_box.xywh[0][0].cpu().numpy()
                    y = single_box.xywh[0][1].cpu().numpy()
                    h = single_box.xywh[0][-1].cpu().numpy()
                    # 
                    real_dist = get_distance(real_hight_person_inch, h)
                    coords_carm = pixel_to_camera_coords(x, y, real_dist)
                    # print("real distance is ", real_dist)
                    result_boxes.append([coords_carm[0], coords_carm[1], coords_carm[2],str(self.id2class[int(single_box.cls.cpu().numpy())])])
                    # r.orig_img = draw_rectangle(r.orig_img, int(single_box.xyxy[0][0].cpu().numpy()),
                    #                     int(single_box.xyxy[0][1].cpu().numpy()),
                    #                     int(single_box.xyxy[0][2].cpu().numpy()),
                    #                     int(single_box.xyxy[0][3].cpu().numpy()),
                    #                     str(self.id2class[int(single_box.cls.cpu().numpy())]) +str(coords_carm))
                    
                elif single_box.cls == 1:
                    # print(single_box.xyxy)
                    # print('detect is car')
                    # 获取person pix high
                    x = single_box.xywh[0][0].cpu().numpy()
                    y = single_box.xywh[0][1].cpu().numpy()
                    h = single_box.xywh[0][-1].cpu().numpy()

                    real_dist = get_distance(real_hight_car_inch, h)
                    # print("real distance is ", real_dist)
                    coords_carm = pixel_to_camera_coords(x, y, real_dist)
                    # print("coords_carm is ", coords_carm)
                    result_boxes.append([coords_carm[0], coords_carm[1], coords_carm[2],str(self.id2class[int(single_box.cls.cpu().numpy())])])
                    # r.orig_img = draw_rectangle(r.orig_img, int(single_box.xyxy[0][0].cpu().numpy()),
                    #                     int(single_box.xyxy[0][1].cpu().numpy()),
                    #                     int(single_box.xyxy[0][2].cpu().numpy()),
                    #                     int(single_box.xyxy[0][3].cpu().numpy()),
                    #                     str(self.id2class[int(single_box.cls.cpu().numpy())]) +str(coords_carm))
                # bus
                elif single_box.cls == 2:
                    # print(single_box.xyxy)
                    # print('detect is car')
                    # 获取person pix high
                    x = single_box.xywh[0][0].cpu().numpy()
                    y = single_box.xywh[0][1].cpu().numpy()
                    h = single_box.xywh[0][-1].cpu().numpy()

                    real_dist = get_distance(real_hight_bus_inch, h)
                    # print("real distance is ", real_dist)
                    coords_carm = pixel_to_camera_coords(x, y, real_dist)
                    # print("coords_carm is ", coords_carm)
                    result_boxes.append([coords_carm[0], coords_carm[1], coords_carm[2],str(self.id2class[int(single_box.cls.cpu().numpy())])])
                    # r.orig_img = draw_rectangle(r.orig_img, int(single_box.xyxy[0][0].cpu().numpy()),
                    #                     int(single_box.xyxy[0][1].cpu().numpy()),
                    #                     int(single_box.xyxy[0][2].cpu().numpy()),
                    #                     int(single_box.xyxy[0][3].cpu().numpy()),
                    #                     str(self.id2class[int(single_box.cls.cpu().numpy())]) +str(coords_carm))
                
                #truck
                elif single_box.cls == 3:
                    # print(single_box.xyxy)
                    # print('detect is car')
                    # 获取person pix high
                    x = single_box.xywh[0][0].cpu().numpy()
                    y = single_box.xywh[0][1].cpu().numpy()
                    h = single_box.xywh[0][-1].cpu().numpy()

                    real_dist = get_distance(real_hight_truck_inch, h)
                    # print("real distance is ", real_dist)
                    coords_carm = pixel_to_camera_coords(x, y, real_dist)
                    # print("coords_carm is ", coords_carm)
                    result_boxes.append([coords_carm[0], coords_carm[1], coords_carm[2],str(self.id2class[int(single_box.cls.cpu().numpy())])])
                    # r.orig_img = draw_rectangle(r.orig_img, int(single_box.xyxy[0][0].cpu().numpy()),
                    #                     int(single_box.xyxy[0][1].cpu().numpy()),
                    #                     int(single_box.xyxy[0][2].cpu().numpy()),
                    #                     int(single_box.xyxy[0][3].cpu().numpy()),
                    #                     str(self.id2class[int(single_box.cls.cpu().numpy())]) +str(coords_carm))
                    
                #traffic-cone
                elif single_box.cls == 4:
                    # print(single_box.xyxy)
                    # print('detect is car')
                    # 获取person pix high
                    x = single_box.xywh[0][0].cpu().numpy()
                    y = single_box.xywh[0][1].cpu().numpy()
                    h = single_box.xywh[0][-1].cpu().numpy()

                    real_dist = get_distance(real_hight_cone_inch, h)
                    # print("real distance is ", real_dist)
                    coords_carm = pixel_to_camera_coords(x, y, real_dist)
                    # print("coords_carm is ", coords_carm)
                    result_boxes.append([coords_carm[0], coords_carm[1], coords_carm[2],str(self.id2class[int(single_box.cls.cpu().numpy())])])
                    # r.orig_img = draw_rectangle(r.orig_img, int(single_box.xyxy[0][0].cpu().numpy()),
                    #                     int(single_box.xyxy[0][1].cpu().numpy()),
                    #                     int(single_box.xyxy[0][2].cpu().numpy()),
                    #                     int(single_box.xyxy[0][3].cpu().numpy()),
                    #                     str(self.id2class[int(single_box.cls.cpu().numpy())]) +str(coords_carm))
                
                
                else:
                    print('detect is others')
                
            # show_image(r.orig_img)
            # cv2.imwrite(f'result.jpg', r.orig_img)
            # idx += 1

            # r.plot()  # plot predictions
            # # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

            # # Show results to screen (in supported environments)
            # r.show()
        return result_boxes
    
    def process_pubrole_image(self, bboxes):
        obstacles = []
        for bbox in bboxes:
        # bbox = [1, 1, 1]
        # for i in range(1):
            ob = Box2d()
            ob.id = str(1)
            xx, yy, zz, class_ = bbox
            ob.x = xx + self.ego.x
            ob.y = -zz + self.ego.y
            
            logger.info("class_:%s, ob.x: %s, ob.y: %s, xx:%s, yy:%s, zz:%s", class_, ob.x, ob.y, xx, yy, zz)
            
            ob.length = 50
            ob.width = 5
            ob.theta = 0
            ob.roleType = RoleType.VEHICLE
            vx = 0
            vy = 0
            vz = 0
            ob.speed = -5
            if ob.speed < 0.01:
                ob.is_static = True
            else:
                ob.is_static = False
            ob.is_virtual = False
            obstacles.append(ob)
            
        # 新增几个个虚拟障碍物
        # obx = [784745.045715332, 784677.8129433006, 784692]
        # oby = [3352900.5697021484, 3352876.871958466, 3352874]
        # for i in range(3):
        # # bbox = [1, 1, 1]
        # # for i in range(1):
        #     ob = Box2d()
        #     ob.id = str(1)
            
        #     ob.x = obx[i]
        #     ob.y = oby[i]
        #     ob.x = 0 + self.ego.x
        #     ob.y = -5 + self.ego.y
        #     ob.length = 5
        #     ob.width = 2
        #     ob.theta = 0
        #     ob.roleType = RoleType.VEHICLE
        #     vx = 0
        #     vy = 0
        #     vz = 0
        #     ob.speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        #     if ob.speed < 0.01:
        #         ob.is_static = True
        #     else:
        #         ob.is_static = False
        #     ob.is_virtual = False
        #     obstacles.append(ob)
        return obstacles

    def process_pubrole(self, bboxes):
        obstacles = []
        for bbox in bboxes:
            ob = Box2d()
            ob.id = str(bbox.node_id)
            xx, yy, zz = transform([bbox.x, bbox.y, bbox.z])
            # print("xx: ", xx)
            # print("yy: ", yy)
            
            ob.x = xx + self.ego.x
            ob.y = yy + self.ego.y
            # print("self.ego.x: ", self.ego.x)
            # print("self.ego.y: ", self.ego.y)
            # print("ob.x: ", ob.x)
            # print("ob.y: ", ob.y)
            # xx:  9.988031029379489
            # yy:  -13.343526885557441
            # self.ego.x:  784829.0141296387
            # self.ego.y:  3352923.7061920166
            # ob.x:  784839.002160668
            # ob.y:  3352910.362665131

            ob.length = bbox.length
            ob.width = bbox.width
            ob.theta = bbox.heading
            ob.roleType = RoleType.VEHICLE
            vx = bbox.vx
            vy = bbox.vy
            vz = bbox.vz
            ob.speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            if ob.speed < 0.01:
                ob.is_static = True
            else:
                ob.is_static = False
            ob.is_virtual = False
            obstacles.append(ob)
        return obstacles

    def change_map(self, new_map):
        if new_map == self.default_map:
            return
        new_map_path = os.path.join(os.path.dirname(__file__), "maps", new_map)
        if not os.path.exists(new_map_path):
            raise PathNotFoundError
        self.default_map = new_map
        VTS_Planner_UpdataMapData(new_map_path)

    def set_destination(self, x, y, theta):
        goal = Pose()
        goal.x = x
        goal.y = y
        goal.theta = theta
        self.goalx = x
        self.goaly = y
        self.goaltheta = theta
        print("====================goal==============================")
        print("goal: ", goal.x, goal.y, goal.theta)
        VTS_Planner_UpdateGoal(goal)

    def update_ego(self, ins):
        self.ego = Box2d()
        self.ego.id = "testee1"
        self.ego.width = ins.veh_size.y
        self.ego.length = ins.veh_size.x
        self.ego.x = ins.position.x
        self.ego.y = ins.position.y

        self.ego.theta = ins.heading

        vx = ins.linear_velocity.x
        vy = ins.linear_velocity.y
        vz = ins.linear_velocity.z
        self.ego.speed = math.sqrt(vx * vx + vy * vy + vz * vz)

        ax = ins.linear_acceleration.x
        ay = ins.linear_acceleration.y
        az = ins.linear_acceleration.z
        self.ego.acc = math.sqrt(ax * ax + ay * ay + az * az)

        self.ego.is_static = False
        self.ego.is_virtual = False

    def update_vehicle_feedback(self, vehicle_feedback):
        if hasattr(vehicle_feedback.steering_feedback, "target_steering_wheel_angle"):
            self.vehicle_feedback.steering_wheel_angle = (
                vehicle_feedback.steering_feedback.target_steering_wheel_angle
            )
        if hasattr(vehicle_feedback.brake_feedback, "brake_pedal_position"):
            self.vehicle_feedback.brake_pedal_position = (
                vehicle_feedback.brake_feedback.brake_pedal_position
            )
        if hasattr(vehicle_feedback.driving_feedback, "accelerator_pedal_position"):
            self.vehicle_feedback.accelerator_pedal_position = (
                vehicle_feedback.driving_feedback.accelerator_pedal_position
            )

    def infer(self, images):
        if self.ego is None or self.vehicle_feedback is None:
            return
        if len(images) == 0:
            print("==========================no image=================================")
            return None
            # obstacles = []
        else:
            obstacles = self.process_image_msg(images)
        control_cmd = ControlCommand()
        VTS_Planner_Process(obstacles, self.ego, self.vehicle_feedback, control_cmd)
        return control_cmd

    # 计算一下当前的目标点和车辆的距离
    def get_destination(self, ins):
        egox = ins.position.x
        egoy = ins.position.y
        
        goalx = self.goalx
        goaly = self.goaly
        
        distance = math.sqrt((goalx - egox) ** 2 + (goaly - egoy) ** 2)
        return distance
    
        # 计算一下当前的车辆的速度
    def get_ego_speed(self, ins):
        # 获取车辆速度信息
        vx = ins.linear_velocity.x
        vy = ins.linear_velocity.y
        vz = ins.linear_velocity.z
        ego_speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        logger.info("ego_speed: {}".format(ego_speed))
        return ego_speed