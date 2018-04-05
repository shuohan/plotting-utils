#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def vectorize(vect):
    vect = vect.reshape((vect.size, 1))
    return vect


class Line:
    def __init__(self, point, directional_vect):
        self.point = vectorize(point)
        self.directional_vect = vectorize(directional_vect)


class Plane:

    def __init__(self, point, normal_vect):

        self.point = vectorize(point)
        self.normal_vect = vectorize(normal_vect)
        self.normal_vect = self.normal_vect / np.linalg.norm(self.normal_vect)

    def calc_3rd_coordinates(self, coords, orders):

        """
        use 2 of the coordinate components (x, y, z) to calculate the 3rd
        coordinate on this plane; supports multiple point input
        - coords: nx2 array, set of 2 coordiniate components, each row
          represents a point, n is the number of points sent in
        - orders: i, j in {1, 2, 3}; 1: x coordinate, 2: y coordinate, 3: z
          coordinate, specifying which components in coords are sent in.
        """

        # a * x + b * y + c * z + d = 0 the equation of a plane
        d = -self.point.T.dot(self.normal_vect)
        result_idx = int(np.setdiff1d(range(3), orders))
        tmp = -d - coords.dot(self.normal_vect[orders, :])
        results = tmp * 1.0 / self.normal_vect[result_idx, :]

        return results

    def translate(self, offsets):

        """
        - offsets: 1x3 array, x, y, z offsets
        """

        offsets = np.array(offsets)
        offsets = offsets.reshape(offsets.size, 1)
        point_on_plane = self.point + offsets
        new_plane = Plane(point_on_plane, self.normal_vect)

        return new_plane

    def rotate(self, rotation):

        """
        - rotation: 3x3 matrix
        """

        normal = rotation.dot(self.normal_vect)
        point = rotation.dot(self.point)

        return Plane(point, normal)
