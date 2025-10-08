class IntWrapper:
    def __init__(self, value: int):
        self.value = value

    def __int__(self):
        return self.value

    def __str__(self):
        return str(self.value)
