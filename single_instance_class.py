class Single:
    '''
    单例类
    '''
    _instance = None

    def __new__(cls, a):
        if not cls._instance:
            cls._instance = super(Single, cls).__new__(cls)
            # cls._instance.a = a
        return cls._instance
    
    def __init__(self, a) -> None:
        self.a = a

s1 = Single(1)
s2 = Single(2)
s1.a = 5
print(s2.a)