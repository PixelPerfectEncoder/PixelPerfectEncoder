class CompressedFrameData:
    def __init__(self, compressed_residual, compressed_descriptors, qp, is_i_frame=None):
        self.residual = compressed_residual
        self.descriptors = compressed_descriptors
        self.qp = qp
        if is_i_frame is not None:
            self.is_i_frame = is_i_frame
    