import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import libMulticastNetwork

from chassis.proto.chassis_enums_pb2 import VEHICLE_FEEDBACK, VEHICLE_CONTROL
from chassis.proto.chassis_messages_pb2 import VehicleFeedback, VehicleControl
from main.proto.messages_pb2 import PubRole, Notify, ActorPrepare, ActorPrepareResult
from main.proto.enums_pb2 import (
    MT_PUBROLE,
    MT_NOTIFY,
    NT_ABORT_TEST,
    NT_START_TEST,
    NT_FINISH_TEST,
    NT_DESTROY_ROLE,
    MT_ACTOR_PREPARE,
    MT_ACTOR_PREPARE_RESULT,
)
from get_ip import get_ip_address
from predictor import Predictor
from predictor_image import Predictor_image

def prepare():
    print("send prepare result")
    send_prepare_result = ActorPrepareResult()
    send_prepare_result.session_id = session_id
    send_prepare_result.actor_id = actor_id
    send_prepare_result.result = True
    data = send_prepare_result.SerializeToString()
    length = len(data)
    ret = prepare_channel.put(MT_ACTOR_PREPARE_RESULT, length, data)
    if ret != 0:
        print("send prepare msg error")


def get_prepare():
    global recv_prepare
    global session_id
    global actor_id

    ret, msg = prepare_channel.get()
    if msg is None:
        return
    if ret >= 0 and msg.type() == MT_ACTOR_PREPARE:
        recv_prepare = True
        data = libMulticastNetwork.getMessageData(msg)
        prepare_msg = ActorPrepare()
        prepare_msg.ParseFromString(data) # 解析prepare消息
        session_id = prepare_msg.session_id # 获取session_id

        brief_data = json.loads(prepare_msg.archive_info.brief_data) # 获取brief_data
        model.change_map(brief_data["zjl_odv_file"])
        target_state = brief_data["testees"][0]["target_state"]
        model.set_destination(  # 设置目的地
            target_state["x"],
            target_state["y"],
            np.deg2rad(target_state["orientation_z"]),
        )


def get_pointcloud_msg():
    # 获取点云信息
    msg = pointcloud_channel.get_pointcloud()
    return model.infer(msg) # 通过点云信息获取控制命令


def get_image_msg():
    msg = image_channel.get_image()
    return model.infer(msg)
    

def process_image_msg(images):
    global img_id
    if save_results:
        save_dir = "results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for i, image in enumerate(images):
        # print(image.timestamp_sec, image.width, image.height)
        # print(image.data.shape)
        img = image.data.astype(np.uint8).reshape(720, 1280, 3)
        if save_results:
            cv2.imwrite(os.path.join(save_dir, f"cam_{i}_" + str(img_id) + ".jpg"), img)
    img_id += 1


def get_image():
    msg = image_channel.get_image()
    if len(msg) == 0:
        return
    process_image_msg(msg)


def process_notify():
    global start_test
    global recv_prepare
    ret, msg = notify_channel.get()
    if msg is None:
        return
    if ret >= 0 and msg.type() == MT_NOTIFY:
        notify = Notify()
        data = libMulticastNetwork.getMessageData(msg)
        notify.ParseFromString(data)
        if notify.type == NT_ABORT_TEST or notify.type == NT_FINISH_TEST:
            print("finish session")
            start_test = False
            recv_prepare = False
        elif notify.type == NT_START_TEST:
            print("start session")
            start_test = True
        elif notify.type == NT_DESTROY_ROLE:
            pass
        else:
            print(
                "session id: {}, event type: {}, time:{}".format(
                    notify.session_id, notify.type, notify.header.sim_ts
                )
            )


def send_control_cmd(target_acc, target_speed, target_steer):
    # print("control cmd: acc {}, speed {}, steer {}".format(target_acc, target_speed, target_steer))
    cmd = VehicleControl()
    # 目标加速度
    cmd.acceleration = target_acc
    # 目标速度
    cmd.speed = target_speed
    # 目标方向盘转角
    cmd.steering_control.target_steering_wheel_angle = target_steer
    data = cmd.SerializeToString()
    length = len(data)
    ret = cmd_channel.put(VEHICLE_CONTROL, length, data)
    if ret != 0:
        print("send cmd error")

# 获取车辆的反馈信息，包括速度、加速度、方向盘转角等
def get_vehicle_feedback():
    ret, msg = cmd_channel.get()
    if msg is None or ret < 0:
        return
    if msg.type() == VEHICLE_FEEDBACK:
        feedback = VehicleFeedback()
        data = libMulticastNetwork.getMessageData(msg)
        feedback.ParseFromString(data)
        model.update_vehicle_feedback(feedback)

# 获取车辆的信息，包括位置、速度、方向等
def get_vehicle_pose(): 
    ins = ins_channel.get_ins()
    if ins.sequence_num == 0:
        return
    model.update_ego(ins)


def main():
    while 1:
        process_notify()
        if not recv_prepare:
            get_prepare()
            time.sleep(0.1)
            continue
        if recv_prepare and not start_test:
            time.sleep(10)
            prepare()
            time.sleep(1)
            continue
        get_vehicle_pose() # 获取车辆的信息
        # get_image()
        # cmd = get_pointcloud_msg() # 通过点云信息获取控制命令
        cmd = get_image_msg()
        if cmd is not None:
            send_control_cmd(cmd.acc, cmd.speed, cmd.steer) # 发送控制命令
        get_vehicle_feedback() # 获取车辆的反馈信息


if __name__ == "__main__":
    
    # 将当前时间作为log文件名
    # import time
    # import logging
    # logging.basicConfig(filename=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)
    
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--config_center", type=str, default="10.11.17.88:52009")
    arg_parser.add_argument("--config_center", type=str, default="101.132.140.171:50009")
    arg_parser.add_argument("--field_id", type=str, default="888888")
    arg_parser.add_argument("--net_interface", type=str, default="wlo1")
    args = arg_parser.parse_args()
    param = libMulticastNetwork.CreateChannelsParam()

    local_ip = get_ip_address(args.net_interface)

    #######################################################
    ###################### 需要修改 ########################
    param.config_center_addr = args.config_center  # 组播配置中心的ip
    param.local_ip = local_ip  # 本机ip
    param.net_interface_name = args.net_interface  # 本机网卡
    param.field_id = (
        args.field_id
    )  # 唯一的场地id，可以任意字符串，需要和daemon和simulator一致
    #######################################################

    param.log_level = 1  # 1-info, 2-warning, 3-error， 设置不同的组播日志等级
    param.client_name = "apollo_testee"
    param.recv_self_msg = False
    session_id = ""

    channels = libMulticastNetwork.ChannelPtrVector()
    ret = libMulticastNetwork.create_channels(param, channels)
    if ret:
        print("create channels failed, ret: {}".format(ret))
        sys.exit(1)
    channel_map = {}
    # 不同的组播消息通道，用于接收和发送消息
    for c in channels:
        print("message channel name: {}, id: {}".format(c.name(), c.id()))
        channel_map[c.name()] = c

    pointcloud_channel = channel_map["lidar"]
    notify_channel = channel_map["notify"]
    cmd_channel = channel_map["vehiclecontrol"]
    prepare_channel = channel_map["prepare"]
    ins_channel = channel_map["ins"]
    image_channel = channel_map["camera"]

    if not libMulticastNetwork.InitImageDecoder():
        print("image decoder init error")
        sys.exit(1)

    img_id = 0
    save_results = False
    recv_prepare = False
    start_test = False
    actor_id = "apollo_testee"

    # model = Predictor()
    model = Predictor_image()
    main()
