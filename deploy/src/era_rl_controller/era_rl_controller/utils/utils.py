import numpy as np


def convert_joint_order(data, from_order, to_order):
    reordered_data = np.array([data[from_order.index(joint)] if joint in from_order else 0.0 for joint in to_order])

    return reordered_data
