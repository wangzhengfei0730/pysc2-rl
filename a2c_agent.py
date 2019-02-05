import tensorflow as tf


class A2CAgent():
    def __init__(self, sess, args):
        self.sess = sess
    
    def build(self, static_shape_channels, resolution, scope=None, reuse=None):
        self._build(static_shape_channels, resolution)
    
    def _build(self, static_shape_channels, resolution):
        pass
        