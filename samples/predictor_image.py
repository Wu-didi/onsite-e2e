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
# import logging

# logger = logging.getLogger(__name__)

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

# # 调用函数并打印结果  
# Xc, Yc, Zc = pixel_to_camera_coords(x_pixel, y_pixel, z_depth, fx, fy, cx, cy)  
# print(f"相机坐标系下的坐标: ({Xc:.2f}, {Yc:.2f}, {Zc:.2f})")


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

def transform_list(input_list):  
    # 创建一个新列表来存储替换后的值  
    result_list = []  

    # 遍历输入列表并替换元素  
    for num in input_list:  
        if num in [8, 9]:  
            result_list.append(-1)  
        elif num in [0, 1, 2, 3, 4, 6]:  
            result_list.append(0)  
        elif num == 5:  
            result_list.append(2)  
        elif num == 7:  
            result_list.append(1)  
        # 如果num不在上述任何一组中，则保持原样（但在这个特定问题中，我们不需要这个else分支）  
        
    # 返回替换后的列表  
    return result_list

class Predictor_image:
    def __init__(self):
        # Load a model
        self.YOLO_model = YOLO('yolov8s.pt')  # load an official model~
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
        
        self.id2class = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', \
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', \
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', \
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', \
                41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',\
                    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', \
                        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', \
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    def process_pointcloud_msg(self, pointclouds):
        for pointcloud in pointclouds:
            # print(
            #     pointcloud.sequence_num,
            #     pointcloud.timestamp_sec,
            #     pointcloud.lidar_timestamp,
            #     pointcloud.points_num,
            # )
            # print(pointcloud.points.shape) # (58383, 4)
            # print(pointcloud.points[0])
            t0 = time.time()
            result, data = inference_detector(self.model, pointcloud.points)
            t1 = time.time()
            box3ds = (
                result._pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            )  # x y z l w h yaw
            # print("box3ds: ", box3ds)
            labels = result._pred_instances_3d.labels_3d.cpu().numpy()
            # print("labels: ", labels)
            # logging.info("labels: %s", labels)
            scores_3d = result._pred_instances_3d.scores_3d.cpu().numpy()
            # logging.info("scores_3d: %s", scores_3d)
            # print("scores_3d: ", scores_3d)
            # logging.info("box3ds: %s", box3ds)
            # print("box3ds: ", box3ds)
            threshold = 0.5
            box3ds = box3ds[scores_3d > threshold]
            # labels and scores_3d should be filtered by scores_3d > 0.2
            labels = labels[scores_3d > threshold]
            scores_3d = scores_3d[scores_3d > threshold]
            print("box3ds: ", box3ds)
            # logging.info("after remove low score box3ds: %s", box3ds)

            # transform nuscenes label to onsite label
            labels = transform_list(labels)
            print(labels)
            print("inference time: ", t1 - t0)

            # box format : [ x, y, z, xdim(l), ydim(w), zdim(h), orientation] + label score
            # dets format : hwlxyzo + class
            
            #-======================================================================================================
            
            dets = box3ds[:, [5, 4, 3, 0, 1, 2, 6]] # hwlxyzo
            info_data = []
            dic_dets = {}
            info_data = np.stack((labels, scores_3d), axis=1)

            dic_dets = {
                "dets": dets,
                "info": info_data,
                "timestamp": pointcloud.lidar_timestamp,
                "frameid": pointcloud.sequence_num,
            }
            results = self.tracker.track_detections(dic_dets)
            print("results: ", results)
            obstacles = self.process_pubrole(results)
            
            #=======================================================================================================
            
            # obstacles = self.process_pubrole_notrack( labels, box3ds)
            print("obstacles: ", obstacles)
            # logging.info("result obstacles: %s", obstacles)
            return obstacles
        
    def process_image_msg(self, images):
        # print(len(images))
        # for image in images: # three images 
            # 
        img = images[0].data.astype(np.uint8).reshape(720, 1280, 3)
        t0 = time.time()
        result = self.YOLO_model.predict(img, conf=0.5, iou=0.5)
        
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
                if single_box.cls == 0 or single_box.cls == 1:
                    # print('detect is person')
                    # 获取person pix high
                    x = single_box.xywh[0][0].cpu().numpy()
                    y = single_box.xywh[0][1].cpu().numpy()
                    h = single_box.xywh[0][-1].cpu().numpy()
                    # 
                    real_dist = get_distance(real_hight_person_inch, h)
                    coords_carm = pixel_to_camera_coords(x, y, real_dist)
                    # print("real distance is ", real_dist)
                    result_boxes.append([coords_carm[0], coords_carm[1], coords_carm[2]])
                    r.orig_img = draw_rectangle(r.orig_img, int(single_box.xyxy[0][0].cpu().numpy()),
                                        int(single_box.xyxy[0][1].cpu().numpy()),
                                        int(single_box.xyxy[0][2].cpu().numpy()),
                                        int(single_box.xyxy[0][3].cpu().numpy()),
                                        str(self.id2class[single_box.cls] +coords_carm))
                elif single_box.cls == 2 or single_box.cls == 3 or single_box.cls == 4 or single_box.cls == 5 \
                    or single_box.cls == 6 or single_box.cls == 8:
                    print(single_box.xyxy)
                    # print('detect is car')
                    # 获取person pix high
                    x = single_box.xywh[0][0].cpu().numpy()
                    y = single_box.xywh[0][1].cpu().numpy()
                    h = single_box.xywh[0][-1].cpu().numpy()

                    real_dist = get_distance(real_hight_car_inch, h)
                    # print("real distance is ", real_dist)
                    coords_carm = pixel_to_camera_coords(x, y, real_dist)
                    # print("coords_carm is ", coords_carm)
                    result_boxes.append([coords_carm[0], coords_carm[1], coords_carm[2]])
                    r.orig_img = draw_rectangle(r.orig_img, int(single_box.xyxy[0][0].cpu().numpy()),
                                        int(single_box.xyxy[0][1].cpu().numpy()),
                                        int(single_box.xyxy[0][2].cpu().numpy()),
                                        int(single_box.xyxy[0][3].cpu().numpy()),
                                        str(self.id2class[single_box.cls] +coords_carm))
                else:
                    print('detect is others')
                
            # show_image(r.orig_img)
            cv2.imwrite(f'result.jpg', r.orig_img)
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
            xx, yy, zz = bbox
            # print("xx: ", xx)
            # print("yy: ", yy)
            # print("zz: ", zz)
            ob.x = xx + self.ego.x
            ob.y = -zz + self.ego.y
            # ob.x = 784745.045715332
            # ob.y = 3352900.5697021484

            # print("self.ego.x: ", self.ego.x)
            # print("self.ego.y: ", self.ego.y)
            # print("ob.x: ", ob.x)
            # print("ob.y: ", ob.y)

            
            ob.length = 5
            ob.width = 3
            ob.theta = 3
            ob.roleType = RoleType.VEHICLE
            vx = 0
            vy = 0
            vz = 0
            ob.speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            if ob.speed < 0.01:
                ob.is_static = True
            else:
                ob.is_static = False
            ob.is_virtual = False
            obstacles.append(ob)
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
            obstacles = []
        else:
            obstacles = self.process_image_msg(images)
        control_cmd = ControlCommand()
        VTS_Planner_Process(obstacles, self.ego, self.vehicle_feedback, control_cmd)
        return control_cmd
