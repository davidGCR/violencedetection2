class BBox():
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h  = h

    def rectangle_area(self):
        return self.w*self.h