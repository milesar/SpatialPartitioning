"""
Naive nearest-neighboor search and test set generation for KDTree and Bucket
validation and testing.
"""

import numpy as np
import pprint as pp
import math
import svgwrite


def initialize_array(dimensions, n_points, max, points):
    for x in range(0, n_points):
        points.append(np.random.randint(low=0, high=max, size=dimensions))


def euclidean_distance(p1, p2):
    """ Calculates the distance between two 'points' in d dimensions"""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def find_neighboors(point, k, points):
    """
    Finds the k nearest-neighboors using brute force comparison to all other
    points in the set.
    """
    points.sort(key=lambda x: euclidean_distance(x, point))
    return points[:k]


def draw_points(max_v, points):

    dwg = svgwrite.Drawing(filename="points.svg", size=(str(max_v)+"px", str(max_v)+"px"))
    hlines = dwg.add(dwg.g(id='hlines', stroke='gray'))
    for y in range(max_v // 10):
        hlines.add(
            dwg.line(
                start=(0, 0), end=(max_v, max_v)))
    vlines = dwg.add(dwg.g(id='vline', stroke='gray'))
    for x in range(max_v // 10):
        vlines.add(
            dwg.line(start=(0, 0), end=(max_v, max_v)))
    for x in points:
        dwg.add(dwg.rect(
            insert=x,
            size=('2px', '2px')))

    dwg.save()


if __name__ == '__main__':
    points = []
    dims = 2
    max_points = 20
    max_v = 100
    k = 3
    target = np.random.randint(low=0, high=max_v, size=dims)

    initialize_array(dims, max_points, max_v, points)
    pp.pprint(len(points))
    pp.pprint(points)
    pp.pprint(find_neighboors(target, k, points))
    draw_points(max_v, points)
