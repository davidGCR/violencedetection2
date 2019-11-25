
class Point(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self): 
        return self._x 
    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self): 
        return self._y
    @y.setter
    def y(self, y):
        self._y = y
    
    def __str__(self):
        return 'Point('+str(self._x)+','+str(self._y)+')'