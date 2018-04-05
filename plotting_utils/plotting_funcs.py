import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import measure


def set_equal_3d_axis(axes, points):

    """ 
    sets the same range length for all directions around the middle point.
    Modified from http://stackoverflow.com/questions/13685386/
    """

    if type(points) is list:
        points = np.vstack(points)
    columns = np.split(points, 3, axis=1)
    range_radius = max([col.max() - col.min() for col in columns]) / 2
    mid_x, mid_y, mid_z = [(col.max() + col.min()) / 2 for col in columns]

    axes.set_xlim(mid_x - range_radius, mid_x + range_radius)
    axes.set_ylim(mid_y - range_radius, mid_y + range_radius)
    axes.set_zlim(mid_z - range_radius, mid_z + range_radius)
    axes.set_aspect('equal')


def plot_line(ax, line, *args, **kwargs):

    """
    - line: instance of Line
    - args, kwargs: the optional args for matplotlib plot
    """

    limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    length = np.sqrt(np.sum([np.diff(lim) ** 2 for lim in limits]))
    radius = length / 2
    line_ends = np.vstack([line.point - line.directional_vect.T * radius,
                      line.point + line.directional_vect.T * radius])
    ax.plot(line_ends[:, 0], line_ends[:, 1], line_ends[:, 2], *args, **kwargs)


def plot_plane(ax, plane, *args, **kwargs):

    """
    - plane: Plane instance
    - args, kwargs: optional parameters for matplotlib plot
    Modified from http://stackoverflow.com/questions/3461869/
    """

    limits = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]

    # choose the direction with the largest normal component as z
    z_idx = np.argmax(np.abs(plane.normal_vect))
    x_idx = (z_idx - 2) % 3
    y_idx = (z_idx - 1) % 3

    # the coordinates of 4 corners of the plane, counter-clockwise
    xy = np.meshgrid(limits[x_idx], limits[y_idx])
    xy = np.hstack([c.reshape(c.size, 1) for c in xy])
    xy = xy[[0, 1, 3, 2], :] 
    z = plane.calc_3rd_coordinates(xy, [x_idx, y_idx])

    vertices = np.hstack([xy, z])
    reverse_order = np.argsort([x_idx, y_idx, z_idx]) # restore the order
    vertices = vertices[:, reverse_order]
    
    poly_collection = Poly3DCollection([vertices], *args, **kwargs)
    ax.add_collection3d(poly_collection)


def plot_3d_image(ax, image, threshold=0.5, *args, **kwargs):
    vertices, faces = measure.marching_cubes(image, level=threshold)
    mesh = Poly3DCollection(vertices[faces], **kwargs)
    ax.add_collection3d(mesh)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_equal_3d_axis(ax, vertices)


def get_colors(number_of_colors, colormap_name='Paired'):
    colormap = cm.get_cmap(colormap_name)
    color_steps = np.linspace(0, 1, number_of_colors)
    colors = [colormap(c) for c in color_steps]
    if len(colors) > 0:
        colors = np.vstack(colors)
    else:
        colors = np.zeros([0, 4])
    return colors


def get_marker_styles(number):
    template = ['^', 'o', 's', 'v', '*', '+']
    num1 = int(number / len(template))
    num2 = number - num1 * len(template)
    result = template * num1 + template[:num2]
    return result
