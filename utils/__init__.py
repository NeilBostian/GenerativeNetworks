from .image_repository import ImageRepository

class obj():
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)