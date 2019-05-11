class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, other):
        return Vector(self.x + other.x , self.y + other.y)

    def sub(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def mag2(self):
        return self.x ** 2  + self.y ** 2

    def mag(self):
        return self.mag2() ** .5

    def norm(self):
        return Vector(self.x / self.mag(), self.y / self.mag())

    def scale(self, s):
        return Vector(self.x * s, self.y * s)

class Circle:

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def inside(self, vector):
        return (self.radius ** 2) >= (self.center.sub(vector).mag2())

    def intersects(self, circle):
        return self.inside(circle.center)

    def moveX(self, d):
        self.center.x += d

    def moveY(self, d):
        self.center.y += d

    def getCenter(self):
        return self.center

    def setCenter(self, center):
        self.center = center

    def bound(self, minX, maxX, minY, maxY):
        self.center.x = max(minX, self.center.x)
        self.center.y = max(minY, self.center.y)
        self.center.x = min(maxX, self.center.x)
        self.center.y = min(maxY, self.center.y)
