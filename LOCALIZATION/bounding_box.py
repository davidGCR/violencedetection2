import math
from LOCALIZATION.point import Point

class BoundingBox(object):
    def __init__(self, pmin, pmax, iou=0, occluded = 0, score=0):

        self._pmin = pmin
        self._pmax = pmax
        self._score = score
        self._occluded = occluded
        self._iou = iou
        pcenter = Point(-1,-1)
        pcenter.x = self._pmin.x + int((self._pmax.x - self._pmin.x) / 2)
        pcenter.y = self._pmin.y + int((self._pmax.y - self._pmin.y) / 2)
        self._pcenter = pcenter

        dy = self._pmax.y - self._pmin.y
        dx = self._pmax.x - self._pmin.x
        self._area = dx*dy
    
    @property
    def center(self):
        return self._pcenter
    
    @property
    def score(self): 
        return self._score 
    @score.setter
    def score(self, score):
        self._score = score


    @property
    def iou(self): 
        return self._iou 
    @iou.setter
    def iou(self, iou):
        self._iou = iou
    
    @property
    def pmin(self): 
        return self._pmin 
    @pmin.setter
    def pmin(self, pmin):
        self._pmin = pmin

    @property
    def pmax(self): 
        return self._pmax
    @pmax.setter
    def pmax(self, pmax):
        self._pmax = pmax
    
    @property
    def occluded(self): 
        return self._occluded
    @occluded.setter
    def occluded(self, occluded):
        self._occluded = occluded

    @property
    def area(self): 
        return self._area
    @area.setter
    def area(self, area):
        self._area = area
        
    
    def percentajeArea(self, overArea):
        per = overArea * 100 / self._area
        return per

    def __eq__(self, other):
        return isinstance(other, BoundingBox) and self._pmin.x == other._pmin.x and self._pmin.y == other._pmin.y and self._pmax.x == other._pmax.x and self._pmax.y == other._pmax.y and self._iou == other._iou and self._area == other._area


    def __hash__(self):
        # use the hashcode of self.ssn since that is used
        # for equality checks as well
        return hash((self._pmin.x, self._pmin.y, self._pmax.x, self._pmax.y, self._iou, self._area))
    
    def __str__(self):
        return 'BoundingBox(('+str(self._pmin.x)+','+str(self._pmin.y)+')'+'('+str(self._pmax.x)+','+str(self._pmax.y)+' , area: '+ str(self._area)+', iou:'+str(self._iou)+', score: '+str(self._score)+'))'
    
    