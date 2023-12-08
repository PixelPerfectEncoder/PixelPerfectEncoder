class PBlockDescriptors:
    def __init__(self, row_mv, col_mv, is_sub_block, frame_seq, row, col):
        self.row_mv = row_mv
        self.col_mv = col_mv
        self.is_sub_block = is_sub_block
        self.frame_seq = frame_seq
        self.row = row
        self.col = col

class IBlockDescriptors:
    def __init__(self, mode, is_sub_block, row, col):
        self.mode = mode
        self.is_sub_block = is_sub_block
        self.row = row
        self.col = col