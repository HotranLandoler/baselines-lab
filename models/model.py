from abc import abstractmethod

from cogdl.models import BaseModel


@deprecated
class Model(BaseModel):
    """Cogdl BaseModel that allow parameters resetting"""
    @abstractmethod
    def reset_parameters(self):
        pass
