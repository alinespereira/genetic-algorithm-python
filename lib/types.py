class MyDict(dict):
    
    
    def __init__(self, iterable):
        super().__init__(iterable)
        for key, value in self.items():
            setattr(self, key, value)