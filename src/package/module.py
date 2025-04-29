"""A Python module with a dummy class"""


class DummyClass:
    def __init__(self):
        print("DummyClass.__init__")

    def dummy_method(self):
        print("DummyClass.dummy_method")

    @staticmethod
    def static_method():
        print("DummyClass.static_method")

    @classmethod
    def class_method(cls):
        print("DummyClass.class_method")
