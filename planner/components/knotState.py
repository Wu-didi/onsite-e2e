import numpy as np


class KnotState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.h = 0.0
        self.curvature = 0.0

    def SetValue(self, state1):
        self.x = state1.x
        self.y = state1.y
        self.h = state1.h

    def Rotate(self, theta):
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        position_matrix = np.array([self.x, self.y])
        rotated_postion = np.dot(position_matrix, rotation_matrix)  # 坐标系旋转
        self.x = rotated_postion[0]
        self.y = rotated_postion[1]
        self.h += theta
