#!/usr/bin/env python

import numpy as np, copy, math, sys, argparse

import os
import sys
import time
from scipy.spatial.transform import Rotation as R
from collections import namedtuple
import pandas as pd

# Get the directory path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the desired path relative to the current file's location
libs_dir = os.path.join(current_dir, "AB3DMOT/")
xinshuo_lib = os.path.join(current_dir, "Xinshuo_PyToolbox")
# Add the path to sys.path
sys.path.append(libs_dir)
sys.path.append(xinshuo_lib)
from AB3DMOT_libs.matching import data_association
from AB3DMOT_libs.kalman_filter import KF
from AB3DMOT_libs.box import Box3D


class BoundingBox(object):
    __slots__ = ('frame_id', 'timestamp', 'type', 'node_id', 'robot', 'x', 'y',
                 'z', 'heading', 'vx', 'vy', 'vz', 'height', 'length', 'width')


def BBox2DataFrame(bboxes):
    cols = ('frame_id', 'timestamp', 'type', 'node_id', 'robot', 'x', 'y', 'z',
            'heading', 'vx', 'vy', 'vz', 'height', 'length', 'width')
    vals = []
    for bbox in bboxes:
        bbox.type = "VEHICLE"
        val = (bbox.frame_id, bbox.timestamp, bbox.type, bbox.node_id,
               bbox.robot, bbox.x, bbox.y, bbox.z, bbox.heading, bbox.vx,
               bbox.vy, bbox.vz, bbox.height, bbox.length, bbox.width)
        vals.append(val)
    df = pd.DataFrame(vals, columns=cols)
    return df


# A Baseline of 3D Multi-Object Tracking
class AB3DMOT():

    def __init__(self, ID_init=0):
        self.ID_start = 1

        # counter
        self.trackers = []
        self.frame_count = 0
        self.ID_count = [ID_init]
        self.id_now_output = []

        # cfg file
        cfg = namedtuple('cfg', [
            'description', 'speed', 'save_root', 'dataset', 'split',
            'det_name', 'cat_list', 'score_threshold', 'num_hypo', 'ego_com',
            'vis', 'affi_pro'
        ])
        cfg.dataset = 'KITTI'
        cfg.det_name = 'pointrcnn'
        cfg.score_threshold = -10000  # can be changed
        cfg.affi_pro = True
        # config
        self.cat = "Car"
        self.affi_process = cfg.affi_pro  # post-processing affinity
        self.get_param(cfg, self.cat)

    def get_param(self, cfg, cat):
        # get parameters for each dataset

        if cfg.dataset == 'KITTI':
            if cfg.det_name == 'pvrcnn':  # tuned for PV-RCNN detections
                if cat == 'Car':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
                elif cat == 'Pedestrian':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4
                elif cat == 'Cyclist':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
                else:
                    assert False, 'error'
            elif cfg.det_name == 'pointrcnn':  # tuned for PointRCNN detections
                if cat == 'Car':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
                elif cat == 'Pedestrian':
                    algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.6, 1, 4
                elif cat == 'Cyclist':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
                else:
                    assert False, 'error'
            elif cfg.det_name == 'deprecated':
                if cat == 'Car':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
                elif cat == 'Pedestrian':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 1, 3, 2
                elif cat == 'Cyclist':
                    algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
                else:
                    assert False, 'error'
            else:
                assert False, 'error'

        else:
            assert False, 'no such dataset'

        # add negative due to it is the cost
        if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1
        self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
         algm, metric, thres, max_age, min_hits

        # define max/min values for the output affinity matrix
        if self.metric in ['dist_3d', 'dist_2d', 'm_dis']:
            self.max_sim, self.min_sim = 0.0, -100.
        elif self.metric in ['iou_2d', 'iou_3d']:
            self.max_sim, self.min_sim = 1.0, 0.0
        elif self.metric in ['giou_2d', 'giou_3d']:
            self.max_sim, self.min_sim = 1.0, -1.0

    def track_detections(self, dic_dets):
        start = time.time()
        cat_res, _ = self.track(dic_dets)
        self.ID_start = max(self.ID_start, self.ID_count[0])  ##global counter
        trk_result = cat_res[0]
        end = time.time()
        # print("time for tracking", (end - start))
        # track detections - now we are considering Car class model for all classes - ToDo : add cyclist and pedestrian categories
        # h,w,l,x,y,z,theta, ID, other info, confidence
        bbox_array = []
        for i, trk in enumerate(trk_result):
            if np.size(trk) == 0:
                continue
            if trk[9] > 0.5:
                bbox = BoundingBox()
                bbox.frame_id = dic_dets["frameid"]
                bbox.timestamp = dic_dets["timestamp"]
                bbox.type = 37  # MOTOVEHICLE
                bbox.node_id = int(trk[7])
                bbox.robot = False
                bbox.x, bbox.y, bbox.z = trk[3], trk[4], trk[5]
                bbox.heading = trk[6]
                bbox.length, bbox.width, bbox.height = trk[2], trk[1], trk[0]
                bbox.vx, bbox.vy, bbox.vz = trk[-3], trk[-2], trk[-1]
                bbox_array.append(bbox)
        return bbox_array

    def process_dets(self, dets, info):
        dets_new = []
        for i, det in enumerate(dets):
            det_tmp = Box3D.array2bbox_raw(det)
            dets_new.append(det_tmp)
        return dets_new

    def within_range(self, theta):

        if theta >= np.pi: theta -= np.pi * 2
        if theta < -np.pi: theta += np.pi * 2

        return theta

    def orientation_correction(self, theta_pre, theta_obs):

        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)

        if abs(theta_obs -
               theta_pre) > np.pi / 2.0 and abs(theta_obs -
                                                theta_pre) < np.pi * 3 / 2.0:
            theta_pre += np.pi
            theta_pre = self.within_range(theta_pre)

        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0: theta_pre += np.pi * 2
            else: theta_pre -= np.pi * 2

        return theta_pre, theta_obs

    def prediction(self):
        trks = []
        for t in range(len(self.trackers)):
            # propagate locations
            kf_tmp = self.trackers[t]
            kf_tmp.kf.predict()
            kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])
            # update statistics
            kf_tmp.time_since_update += 1
            trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks

    def update(self, matched, unmatched_trks, dets, info):
        # update matched trackers with assigned detections
        dets = copy.copy(dets)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0],
                            0]  # a list of index
                assert len(d) == 1, 'error'

                # update statistics
                trk.time_since_update = 0  # reset because just updated
                trk.hits += 1

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(dets[d[0]])
                trk.kf.x[3], bbox3d[3] = self.orientation_correction(
                    trk.kf.x[3], bbox3d[3])
                # kalman filter update with observation
                trk.kf.update(bbox3d)
                trk.kf.x[3] = self.within_range(trk.kf.x[3])
                trk.info = info[d, :][0]

    def birth(self, dets, info, unmatched_dets):

        new_id_list = list()  # new ID generated for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            trk = KF(Box3D.bbox2array(dets[i]), info[i, :], self.ID_count[0])
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            self.ID_count[0] += 1

        return new_id_list

    def output(self):

        num_trks = len(self.trackers)
        results = []
        for trk in reversed(self.trackers):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            d = Box3D.array2bbox(trk.kf.x[:7].reshape(
                (7, )))  # bbox location self
            d = Box3D.bbox2array_raw(d)

            if ((trk.time_since_update < self.max_age)
                    and (trk.hits >= self.min_hits
                         or self.frame_count <= self.min_hits)):
                velocity = trk.get_velocity()
                results.append(
                    np.concatenate((d, [trk.id], trk.info,
                                    velocity.ravel())).reshape(1, -1))
            num_trks -= 1

            # deadth, remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(num_trks)

        return results

    def process_affi(self, affi, matched, unmatched_dets, new_id_list):

        trk_id = self.id_past  # ID in the trks for matching

        det_id = [-1 for _ in range(affi.shape[0])]  # initialization
        for match_tmp in matched:
            det_id[match_tmp[0]] = trk_id[match_tmp[1]]

        count = 0
        assert len(unmatched_dets) == len(new_id_list), 'error'
        for unmatch_tmp in unmatched_dets:
            det_id[unmatch_tmp] = new_id_list[
                count]  # new_id_list is in the same order as unmatched_dets
            count += 1
        assert not (
            -1 in det_id), 'error, still have invalid ID in the detection list'
        affi = affi.transpose()

        permute_row = [
            trk_id.index(output_id_tmp)
            for output_id_tmp in self.id_past_output
        ]
        affi = affi[permute_row, :]
        assert affi.shape[0] == len(self.id_past_output), 'error'

        max_index = affi.shape[1]
        permute_col = list()
        to_fill_col, to_fill_id = list(), list(
        )  # append new columns at the end, also remember the ID for the added ones
        for output_id_tmp in self.id_now_output:
            try:
                index = det_id.index(output_id_tmp)
            except:  # some output ID does not exist in the detections but rather predicted by KF
                index = max_index
                max_index += 1
                to_fill_col.append(index)
                to_fill_id.append(output_id_tmp)
            permute_col.append(index)

        # expand the affinity matrix with newly added columns
        append = np.zeros((affi.shape[0], max_index - affi.shape[1]))
        append.fill(self.min_sim)
        affi = np.concatenate([affi, append], axis=1)

        # find out the correct permutation for the newly added columns of ID
        for count in range(len(to_fill_col)):
            fill_col = to_fill_col[count]
            fill_id = to_fill_id[count]
            row_index = self.id_past_output.index(fill_id)

            # construct one hot vector because it is proapgated from previous tracks, so 100% matching
            affi[row_index, fill_col] = self.max_sim
        affi = affi[:, permute_col]

        return affi

    def track(self, dets_all):

        dets, info = dets_all['dets'], dets_all[
            'info']  # dets: N x 7, float numpy array
        self.frame_count += 1

        # recall the last frames of outputs for computing ID correspondences during affinity processing
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]

        # process detection format
        dets = self.process_dets(dets, info)
        # print(dets[0], "dets with classes")

        # tracks propagation based on velocity
        trks = self.prediction()
        # print("trks after prediction: ", trks)

        # matching
        trk_innovation_matrix = None
        if self.metric == 'm_dis':
            trk_innovation_matrix = [
                trk.compute_innovation_matrix() for trk in self.trackers
            ]
        matched, unmatched_dets, unmatched_trks, cost, affi = \
         data_association(dets, trks, self.metric, self.thres, self.algm, trk_innovation_matrix)

        # print("matched : ", matched)
        # print("unmatched_dets : ", unmatched_dets)
        # print("unmatched_trks : ", unmatched_trks)

        self.update(matched, unmatched_trks, dets, info)
        new_id_list = self.birth(dets, info, unmatched_dets)

        results = self.output()
        if len(results) > 0:
            results = [np.concatenate(results)
                       ]  # h,w,l,x,y,z,theta, ID, other info, confidence
        else:
            results = [np.empty((0, 10))]
        self.id_now_output = results[0][:, 7].tolist(
        )  # only the active tracks that are outputed
        if self.affi_process:
            affi = self.process_affi(affi, matched, unmatched_dets,
                                     new_id_list)

        return results, affi


def euler_from_quaternion(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z  # in radians


def yaw_to_quaternion(yaw):
    r = R.from_euler('z', yaw, degrees=False)
    return r.as_quat()


if __name__ == '__main__':
    print("tracking node initialzied")
    ID_start = 1
    AB3DMOT(ID_init=ID_start)
