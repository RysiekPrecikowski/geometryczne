# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import json as js


# %%
class PointsCollection:
    def __init__(self, points = [], color = None, marker = None):
        self.points = np.array(points)
        self.color = color
        self.marker = marker
        
class LinesCollection:
    def __init__(self, lines = [], color = None):
        self.color = color
        self.lines = lines
        
    def add(self, line):
        self.lines.append(line)
        
    def get_collection(self):
        if self.color:
            return mcoll.LineCollection(self.lines, [mcolors.to_rgba(self.color)] * len(self.lines))
        else:
            return mcoll.LineCollection(self.lines)

class Plot:
    def __init__(self, points=[], lines=[], json = None):
        if json is None:
            self.points = points
            self.lines = lines
        else:
            self.points = [PointsCollection(pointsCol) for pointsCol in js.loads(json)["points"]]
            self.lines = [LinesCollection(linesCol) for linesCol in js.loads(json)["lines"]]
            
    def draw(self, title = None):
        ax = plt.axes()
        for collection in self.points:
            if collection.points.size > 0:
                ax.scatter(*zip(*collection.points), c=collection.color, marker=collection.marker)
        for collection in self.lines:
            ax.add_collection(collection.get_collection())
        ax.autoscale()

        if title is not None:
            plt.title(label= title)
        
        plt.draw()
        
    def toJSON(self):
        return js.dumps({"points": [pointCol.points.tolist() for pointCol in self.points], 
                          "lines":[linesCol.lines for linesCol in self.lines]})


# %%
import random
import math
from random import randint


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbiór a

cords = (-10 ** 14, 10 ** 14)
a = [(randint(cords[0], cords[1]), randint(cords[0], cords[1])) for _ in range(10**5)]
Plot([PointsCollection(a)]).draw()

with open("a.txt", "w") as fp:
    js.dump(a, fp)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbiór b

cords = (-10 ** 14, 10 ** 14)

b = [(randint(cords[0], cords[1]), randint(cords[0], cords[1])) for _ in range(10**5)]
Plot([PointsCollection(b)]).draw()

with open("b.txt", "w") as fp:
    js.dump(b, fp)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbior c okrag
n = 1000
r = 100
c = [None] * n
p = (0.0, 0.0)
for i in range(n):
    d = random.random() * (2*math.pi)
    c[i] = (p[0] + r*math.sin(d), p[1] + r*math.cos(d))

Plot([PointsCollection(c)]).draw()

with open("c.txt", "w") as fp:
    js.dump(c, fp)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbior d prosta

n = 1000

p1 = (-1.0, 0.0)
p2 = (1.0, 0.1)

lineA = (p1[1] - p2[1]) / (p1[0] - p2[0])
lineB = p1[1] - lineA * p1[0]

d = [None] * n
for i in range(n):
    x = randint(-1000,1000)
    d[i] = (x, lineA*x + lineB)

Plot([PointsCollection(d)]).draw()

with open("d.txt", "w") as fp:
    js.dump(d, fp)


# %%
def load():
    with open("a.txt", "r") as fp:
        a = js.load(fp)

    with open("b.txt", "r") as fp:
        b = js.load(fp)

    with open("c.txt", "r") as fp:
        c = js.load(fp)

    with open("d.txt", "r") as fp:
        d = js.load(fp)

    return a, b, c, d
a,b,c,d = load()


# %%
def det1(a,b,c):
    return a[0]*b[1] + a[1]*c[0] + b[0]*c[1] - c[0]*b[1] - a[1]*b[0] - a[0]*c[1]


# %%
def det2(a,b,c):
    return (a[0] - c[0])*(b[1] - c[1]) - (a[1] - c[1])*(b[0] - c[0])


# %%
def classify(points, detFun, e = 10** -14, a = (-1.0,0.0), b = (1.0,0.1)):
    # res = {
    #     'left': [],
    #     'right': [],
    #     'collinear': []
    # }

    left = []
    right = []
    collinear = []

    for p in points:
        d = detFun(a,b,p)

        if d > e:
            # res['left'].append(p)
            left.append(p)
        elif d < -e:
            # res['right'].append(p)
            right.append(p)
        else:
            # res['collinear'].append(p)
            collinear.append(p)
    # return res
    return left, collinear, right


# %%
def plotClassification(points, detFun, e = 10**-14, a = (-1.0,0.0), b = (1.0,0.1)):
    # res = classify(points,detFun,e,a,b)

    # classified = [PointsCollection(res['left'], color='blue'),
    #               PointsCollection(res['right'], color= 'green'),
    #               PointsCollection(res['collinear'], color= 'pink')]
    # Plot(classified).draw()

    # print("Left: ", len(res['left']))
    # print("Right: ",len(res['right']))
    # print("Collinear: ",len(res['collinear']))

    left, collinear, right = classify(points, detFun, e, a, b)

    
    classified = [PointsCollection(left, color='blue'),
                  PointsCollection(right, color='green'),
                  PointsCollection(collinear, color= 'pink')]

    
    Plot(classified).draw()

    print("Left: ", len(left))
    print("Right: ",len(right))
    print("Collinear: ",len(collinear))


# %%
def det_np3x3(a,b,c):
    arr = np.array([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]])
    return np.linalg.det(arr)


# %%
def det_np2x2(a,b,c):
    arr = np.array([[a[0] - c[0], a[1] - c[1]], [b[0] - c[0], b[1] - c[1]]])
    return np.linalg.det(arr)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

plotClassification(b,det2)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
plotClassification(b,det_np2x2)


# %%
def countDiff(points, detFun1, detFun2, e = 10**-14, a = (-1.0,0.0), b = (1.0,0.1)):
    #czerwone to punkty sklasyfikowane przez detFun1 jako wspolliniowe, a detFun2 nie
    #zielone to punkyu sklasyfikowane przez detFun2 jako wpolliniowe, a detFun1 nie
    def cat(p, f):
        d = f(a,b,p)

        if d > e:
            return 'l'
        elif d < -e:
            return 'r'
        else:
            return 'c'
    res1 = []
    res2 = []        
    for p in points:
        if cat(p,detFun1) is not cat(p,detFun2):
            if cat(p, detFun1) is not 'c':
                res2.append(p)
            else:
                res1.append(p)
            

            

    get_ipython().run_line_magic('matplotlib', 'ipympl')
    Plot([PointsCollection(res1, color= 'red'),PointsCollection(res2, color='green')]).draw()

    
    return len(res1) + len(res2)


# %%
countDiff(b,det2,det_np2x2)


# %%



