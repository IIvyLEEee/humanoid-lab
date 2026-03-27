from .hist_observations import ObsHistory


class BaseRLInterface:
    def __init__(self, config):
        self.config = config
        self.obs_hist = ObsHistory(self.config.get("observations", {}))

    def get_joint_sequence(self):
        return self.joint_seq

    def get_default_joint_pos(self):
        return self.default_joint_pos

    def get_Kp(self):
        return self.Kp

    def get_Kd(self):
        return self.Kd

    def get_action_scale(self):
        return self.action_scale

    def perform_inference(self, root_quat, root_ang_vel, mes_q, mes_qdot):
        raise NotImplementedError("Subclasses should implement this method")
