class CodecConfig:
    def __init__(
            self, 
            block_size, 
            block_search_offset,
            do_approximated_residual : bool,
            do_dct : bool,
            do_quantization : bool,
            do_entropy : bool) -> None:
        self.block_size = block_size
        self.block_search_offset = block_search_offset
        self.do_approximated_residual = do_approximated_residual
        self.do_dct = do_dct
        self.do_quantization =  do_quantization
        self.do_entropy = do_entropy
