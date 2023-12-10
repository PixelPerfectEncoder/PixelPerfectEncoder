class CodecConfig:
    def __init__(
        self,
        block_size,
        block_search_offset = 2,
        i_Period: int = -1,
        qp: int = 0,
        approximated_residual_n: int = 2,
        do_approximated_residual: bool = False,
        do_entropy: bool = False,
        RD_lambda: float = 0,
        VBSEnable: bool = False,
        FMEEnable: bool = False,
        FastME: bool = False,
        nRefFrames: int = 1,
        DisplayBlocks: bool = False,
        DisplayMvAndMode: bool = False,
        DisplayRefFrames: bool = False,
        FastME_LIMIT: int = -1,
        RCTable: dict = dict(),
        RCflag: int = 0,
        targetBR: float = 0,
        fps: int = 30,
        total_frames: int = 0,
        filename = '',
        is_firstpass = False,
    ) -> None:
        self.block_size = block_size
        self.sub_block_size = block_size // 2
        self.block_search_offset = block_search_offset
        self.i_Period = i_Period
        self.qp = qp
        self.approximated_residual_n = approximated_residual_n
        self.do_approximated_residual = do_approximated_residual
        self.do_entropy = do_entropy
        self.RD_lambda = RD_lambda
        self.VBSEnable = VBSEnable
        self.FMEEnable = FMEEnable
        self.FastME = FastME
        self.nRefFrames = nRefFrames
        self.DisplayBlocks = DisplayBlocks
        self.DisplayMvAndMode = DisplayMvAndMode
        self.DisplayRefFrames = DisplayRefFrames
        self.FastME_LIMIT = FastME_LIMIT
        self.RCTable = RCTable
        self.RCflag = RCflag
        self.targetBR = targetBR
        self.fps = fps
        self.total_frames = total_frames
        self.filename = filename
        self.is_firstpass = is_firstpass
        self.dQPLimit = dQPLimit  # Maximum allowed delta QP