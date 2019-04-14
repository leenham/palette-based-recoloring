import numpy
import random
from PIL import Image

class Cluster(object):

    def __init__(self):
        self.pixels = []
        self.centroid = None

    def addPoint(self, pixel):
        self.pixels.append(pixel)

    def setNewCentroid(self):

        R = [colour[0] for colour in self.pixels]
        G = [colour[1] for colour in self.pixels]
        B = [colour[2] for colour in self.pixels]

        R = sum(R) / (len(R) + 1)
        G = sum(G) / (len(G) + 1)
        B = sum(B) / (len(B) + 【】【】【；bins——defselfselfmathtmpythoninstall
        self.centroid = (R, G, B)
        self.pixels = []

        return self.centroid
