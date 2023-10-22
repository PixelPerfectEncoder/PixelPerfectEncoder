class CodecConfig:
    def __init__(
            self, 
            block_size, 
            block_search_offset,
            do_approximated_residual) -> None:
        self.block_size = block_size
        self.block_search_offset = block_search_offset
        self.do_approximated_residual = do_approximated_residual