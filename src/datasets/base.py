
class BaseDataset:
    def __init__(self, src_path) -> None:
        self.src_path = src_path # path to the source data
        pass

    def load(self) -> None:
        raise NotImplementedError("load method must be implemented")
    
    def split(self) -> None:
        raise NotImplementedError("split method must be implemented")
    
