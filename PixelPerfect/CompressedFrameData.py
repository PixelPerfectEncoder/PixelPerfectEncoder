class CompressedFrameData:
    def __init__(self, compressed_residual, compressed_descriptors, bit_rate, is_i_frame=None):
        self.residual = compressed_residual
        self.descriptors = compressed_descriptors
        self.bit_rate = bit_rate
        if is_i_frame is not None:
            self.is_i_frame = is_i_frame
    