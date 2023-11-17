class CodecConfig:
    def __init__(
        self,
        block_size,
        block_search_offset,
        i_Period: int = -1,
        quant_level: int = 2,
        approximated_residual_n: int = 2,
        do_approximated_residual: bool = False,
        do_dct: bool = False,
        do_quantization: bool = False,
        do_entropy: bool = False,
        RD_lambda: float = 0,
        VBSEnable: bool = False,
        FMEEnable: bool = False,
        FastME: bool = False,
    ) -> None:
        self.block_size = block_size
        self.sub_block_size = block_size // 2
        self.block_search_offset = block_search_offset
        self.i_Period = i_Period
        self.quant_level = quant_level
        self.approximated_residual_n = approximated_residual_n
        self.do_approximated_residual = do_approximated_residual
        self.do_dct = do_dct
        self.do_quantization = do_quantization
        self.do_entropy = do_entropy
        self.RD_lambda = RD_lambda
        self.VBSEnable = VBSEnable
        self.FMEEnable = FMEEnable
        self.FastME = FastME  