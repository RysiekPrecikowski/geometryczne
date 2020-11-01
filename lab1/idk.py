# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
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
from random import randint,uniform


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbiór a
def aDataset():
    cords = (-1000, 1000)
    d = [(randint(cords[0], cords[1]), randint(cords[0], cords[1])) for _ in range(10**5)]
    
    # Plot([PointsCollection(a)]).draw()
    with open("a.txt", "w") as fp:
        js.dump(d, fp)

    return d

aDataset()

# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbiór b
def bDataset():
    cords = (-10 ** 14, 10 ** 14)

    d = [(randint(cords[0], cords[1]), randint(cords[0], cords[1])) for _ in range(10**5)]
    # Plot([PointsCollection(d)]).draw()

    with open("b.txt", "w") as fp:
        js.dump(d, fp)
    
    return d


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbior c okrag
def cDataset():
    n = 1000
    r = 100
    d = [None] * n
    p = (0.0, 0.0)
    for i in range(n):
        d = random.random() * (2*math.pi)
        d[i] = (p[0] + r*math.sin(d), p[1] + r*math.cos(d))

    # Plot([PointsCollection(d)]).draw()

    with open("c.txt", "w") as fp:
        js.dump(d, fp)
    
    return d


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbior d prosta
def dDataset():
    n = 1000

    p1 = (-1.0, 0.0)
    p2 = (1.0, 0.1)

    lineA = (p1[1] - p2[1]) / (p1[0] - p2[0])
    lineB = p1[1] - lineA * p1[0]

    d = [None] * n
    for i in range(n):
        x = randint(-1000,1000)
        d[i] = (x, lineA*x + lineB)

    # Plot([PointsCollection(d)]).draw()

    with open("d.txt", "w") as fp:
        js.dump(d, fp)
    
    return d


# %%


def plotDataset(d):
    get_ipython().run_line_magic('matplotlib', 'ipympl')
    Plot([PointsCollection(d)]).draw()


# %%
def det3x3(a,b,c):
    return a[0]*b[1] + a[1]*c[0] + b[0]*c[1] - c[0]*b[1] - a[1]*b[0] - a[0]*c[1]


# %%
def det2x2(a,b,c):
    return (a[0] - c[0])*(b[1] - c[1]) - (a[1] - c[1])*(b[0] - c[0])


# %%
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
from random import randint,uniform


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbiór a
def aDataset():
    cords = (-1000, 1000)
    d = [(uniform(cords[0], cords[1]), uniform(cords[0], cords[1])) for _ in range(10**5)]
    
    Plot([PointsCollection(a)]).draw()
    with open("a.txt", "w") as fp:
        js.dump(d, fp)

    return d



# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbiór b
def bDataset():
    cords = (-10 ** 14, 10 ** 14)

    d = [(randint(cords[0], cords[1]), randint(cords[0], cords[1])) for _ in range(10**5)]
    Plot([PointsCollection(d)]).draw()

    with open("b.txt", "w") as fp:
        js.dump(d, fp)
    
    return d


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbior c okrag
def cDataset():
    n = 1000
    r = 100
    d = [None] * n
    p = (0.0, 0.0)
    for i in range(n):
        d = random.random() * (2*math.pi)
        d[i] = (p[0] + r*math.sin(d), p[1] + r*math.cos(d))

    Plot([PointsCollection(d)]).draw()

    with open("c.txt", "w") as fp:
        js.dump(d, fp)
    
    return d


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

# zbior d prosta
def dDataset():
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
    
    return d


# %%


def plotDataset(d):
    get_ipython().run_line_magic('matplotlib', 'ipympl')
    Plot([PointsCollection(d)]).draw()


# %%
def det3x3(a,b,c):
    return a[0]*b[1] + a[1]*c[0] + b[0]*c[1] - c[0]*b[1] - a[1]*b[0] - a[0]*c[1]


# %%
def det2x2(a,b,c):
    return (a[0] - c[0])*(b[1] - c[1]) - (a[1] - c[1])*(b[0] - c[0])


# %%
def detNp3x3(a,b,c):
    arr = np.array([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]])
    return np.linalg.det(arr)


# %%
def detNp2x2(a,b,c):
    arr = np.array([[a[0] - c[0], a[1] - c[1]], [b[0] - c[0], b[1] - c[1]]])
    return np.linalg.det(arr)


# %%
def classify(points, detFun, e = 10** -14, a = (-1.0,0.0), b = (1.0,0.1)):
    left = []
    right = []
    collinear = []

    for p in points:
        d = detFun(a,b,p)

        if d > e:
            left.append(p)
        elif d < -e:
            right.append(p)
        else:
            collinear.append(p)
 
    return left, collinear, right


# %%
def plotClassification(points, detFun, e = 10**-14, a = (-1.0,0.0), b = (1.0,0.1)):
    
    left, collinear, right = classify(points, detFun, e, a, b)

    
    classified = [PointsCollection(left, color='blue'),
                  PointsCollection(right, color='green'),
                  PointsCollection(collinear, color= 'pink')]

    
    Plot(classified).draw()

    print("Left: ", len(left))
    print("Right: ",len(right))
    print("Collinear: ",len(collinear))


# %%
def countDiff(points, detFun1, detFun2, e = 10**-14, a = (-1.0,0.0), b = (1.0,0.1), show = False):
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
            

    if show is True:
        get_ipython().run_line_magic('matplotlib', 'ipympl')
        Plot([PointsCollection(res1, color= 'red'),PointsCollection(res2, color='green')]).draw()

    print ("Difference:",len(res1)+len(res2))


# %%
from pathlib import Path
def load():
    file = Path("a.txt")
    if file.is_file():
        with open(file, "r") as fp:
            a = js.load(fp)
    else:
        a = aDataset()
    
    file = Path("b.txt")
    if file.is_file():
        with open(file, "r") as fp:
            b = js.load(fp)
    else:
        b = bDataset()
    
    file = Path("c.txt")
    if file.is_file():
        with open(file, "r") as fp:
            c = js.load(fp)
    else:
        c = cDataset()
    
    file = Path("d.txt")
    if file.is_file():
        with open(file, "r") as fp:
            d = js.load(fp)
    else:
        d = dDataset()

    return a, b, c, d
a,b,c,d = load()

# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
Plot([PointsCollection(a)]).draw()

# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
Plot([PointsCollection(b)]).draw()

# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
Plot([PointsCollection(c)]).draw()

# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
Plot([PointsCollection(d)]).draw()

# %%
epsilon = 10**-14


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

plotClassification(a,det2x2,e = epsilon)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
plotClassification(a,det3x3,e= epsilon)


# %%
countDiff(a,det2x2,det3x3, e=epsilon)


# %%
countDiff(a,det2x2,detNp3x3, e=epsilon)


# %%
countDiff(a,det3x3,detNp3x3, e=epsilon)


# %%
epsilon = 10**-8


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

plotClassification(a,det2x2,e = epsilon)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
plotClassification(a,det3x3,e= epsilon)


# %%
countDiff(a,det2x2,det3x3, e=epsilon)


# %%
countDiff(a,det2x2,detNp3x3, e=epsilon)


# %%
countDiff(a,det3x3,detNp3x3, e=epsilon)


# %%
epsilon = 10**-4


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

plotClassification(a,det2x2,e = epsilon)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
plotClassification(a,det3x3,e= epsilon)


# %%
countDiff(a,det2x2,det3x3, e=epsilon)


# %%
countDiff(a,det2x2,detNp3x3, e=epsilon)


# %%
countDiff(a,det3x3,detNp3x3, e=epsilon)


# %%
epsilon = 10**-2


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

plotClassification(a,det2x2,e = epsilon)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
plotClassification(a,det3x3,e= epsilon)


# %%
countDiff(a,det2x2,det3x3, e=epsilon)


# %%
countDiff(a,det2x2,detNp3x3, e=epsilon)


# %%
countDiff(a,det3x3,detNp3x3, e=epsilon)


# %%
epsilon = 10**-1


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

plotClassification(a,det2x2,e = epsilon)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
plotClassification(a,det3x3,e= epsilon)


# %%
countDiff(a,det2x2,det3x3, e=epsilon)


# %%
countDiff(a,det2x2,detNp2x2, e=epsilon)


# %%
countDiff(a,det3x3,detNp3x3, e=epsilon)


# %%
epsilon = 10**0


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

plotClassification(a,det2x2,e = epsilon)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
plotClassification(a,det3x3,e= epsilon)


# %%
countDiff(a,det2x2,det3x3, e=epsilon)


# %%
countDiff(a,det2x2,detNp3x3, e=epsilon)


# %%
countDiff(a,det3x3,detNp3x3, e=epsilon)



# %%
epsilon = 10**-14


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')

plotClassification(b,det2x2,e = epsilon)


# %%
get_ipython().run_line_magic('matplotlib', 'ipympl')
plotClassification(b,det3x3,e= epsilon)



# %%
countDiff(b,det2x2,det3x3, e=epsilon)


# %%
countDiff(b,det2x2,detNp3x3, e=epsilon)


# %%
countDiff(b,det3x3,detNp3x3, e=epsilon)