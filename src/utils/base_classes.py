import argparse


class SDFModule:
    @classmethod
    def add_arguments(parser: argparse.ArgumentParser):
        pass

    @classmethod
    def set_role(cls, role):
        def decorator(subclass):
            subclass.role = role
            return subclass
        return decorator

    def __init__(self, args, **kwargs):
        pass
    
    def initialize(self):
        return self

    def unload(self):
        pass