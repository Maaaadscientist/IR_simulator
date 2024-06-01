import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Photon:
    def __init__(self, x, y, z, dx, dy, dz, geometry):
        self.position = np.array([x, y, z])
        self.direction = np.array([dx, dy, dz])
        self.status = 'active'  # 'active' or 'absorbed'
        self.surface, self.channel = self.init_surface_and_channel(geometry)

    def init_surface_and_channel(self, geometry):
        """
        Initialize the surface and channel based on the initial position.
        
        Parameters:
        geometry (Geometry): The solid matter geometry to check for initial position.

        Returns:
        tuple: surface number (int), channel number (int)
        """
        for square in geometry.squares:
            point, normal, surface = square
            if self.is_on_surface(point, normal):
                channel = geometry._get_channel(self.position, point, normal)
                return surface, channel
        return None, None

    def is_on_surface(self, plane_point, plane_normal):
        """
        Check if the photon's initial position is on the given surface.

        Parameters:
        plane_point (np.array): A point on the plane.
        plane_normal (np.array): The normal vector of the plane.

        Returns:
        bool: True if the photon is on the surface, False otherwise.
        """
        # Plane equation: (point - plane_point) . plane_normal = 0
        return np.abs(np.dot(self.position - plane_point, plane_normal)) < 1e-6

    def check_and_update_status(self, geometry):
        """
        Check if the photon's ray intersects with the solid matter geometry.
        If yes, update the photon's position, direction, surface, and channel.
        If not, update the photon's status to 'absorbed'.
        
        Parameters:
        geometry (Geometry): The solid matter geometry to check for intersection.
        """
        intersection, normal, surface, channel = self.intersects(geometry)
        if intersection is not None:
            self.position = intersection
            self.direction = self.reflect(self.direction, normal)
            self.surface = surface
            self.channel = channel
        else:
            self.status = 'absorbed'

    def intersects(self, geometry):
        """
        Check if the photon's ray intersects with the solid matter geometry.

        Parameters:
        geometry (Geometry): The solid matter geometry to check for intersection.

        Returns:
        tuple: (intersection point (np.array), normal vector (np.array), surface number (int), channel number (int)) if there is an intersection, (None, None, None, None) otherwise.
        """
        return geometry.intersects_with_ray(self.position, self.direction)

    def reflect(self, direction, normal):
        """
        Reflect the direction vector based on the normal vector at the point of intersection.

        Parameters:
        direction (np.array): The incoming direction vector.
        normal (np.array): The normal vector at the point of intersection.

        Returns:
        np.array: The reflected direction vector.
        """
        return direction - 2 * np.dot(direction, normal) * normal


class Geometry:
    def intersects_with_ray(self, position, direction):
        """
        Placeholder method for checking ray intersection with the geometry.
        
        Parameters:
        position (np.array): The starting position of the ray.
        direction (np.array): The direction vector of the ray.

        Returns:
        tuple: (intersection point (np.array), normal vector (np.array), surface number (int), channel number (int)) if there is an intersection, (None, None, None, None) otherwise.
        """
        raise NotImplementedError("This method should be implemented in the derived geometry classes.")


class BookGeometry(Geometry):
    def __init__(self, angle_degrees):
        """
        Initialize the BookGeometry with the angle between the two squares.

        Parameters:
        angle_degrees (float): The angle between the two squares in degrees.
        """
        self.angle_radians = np.radians(angle_degrees)
        self.square_size = 50.0  # size of the squares in mm
        self.squares = self._compute_square_planes()

    def _compute_square_planes(self):
        """
        Compute the planes representing the two squares.
        
        Returns:
        list of tuples: Each tuple contains a point on the plane, the normal vector of the plane, and the surface number.
        """
        # Plane 1: Lying on the x-y plane
        point1 = np.array([0, 0, 0])
        normal1 = np.array([0, 0, 1])
        surface1 = 1

        # Plane 2: Rotated around the x-axis by the given angle
        point2 = np.array([0, 0, 0])
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(self.angle_radians), -np.sin(self.angle_radians)],
            [0, np.sin(self.angle_radians), np.cos(self.angle_radians)]
        ])
        normal2 = rotation_matrix @ np.array([0, 0, 1])
        surface2 = 2

        return [(point1, normal1, surface1), (point2, normal2, surface2)]

    def intersects_with_ray(self, position, direction):
        """
        Check if the photon's ray intersects with either of the two squares.

        Parameters:
        position (np.array): The starting position of the ray.
        direction (np.array): The direction vector of the ray.

        Returns:
        tuple: (intersection point (np.array), normal vector (np.array), surface number (int), channel number (int)) if there is an intersection, (None, None, None, None)) otherwise.
        """
        for square in self.squares:
            point, normal, surface = square
            intersection = self._ray_intersects_plane(position, direction, point, normal)
            if intersection is not None:
                channel = self._get_channel(intersection, point, normal)
                return intersection, normal, surface, channel
        return None, None, None, None

    def _ray_intersects_plane(self, ray_origin, ray_direction, plane_point, plane_normal):
        """
        Check if a ray intersects a plane.

        Parameters:
        ray_origin (np.array): The origin of the ray.
        ray_direction (np.array): The direction of the ray.
        plane_point (np.array): A point on the plane.
        plane_normal (np.array): The normal vector of the plane.

        Returns:
        np.array: The intersection point if there is an intersection, None otherwise.
        """
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        denom = np.dot(plane_normal, ray_direction)
        if np.abs(denom) < 1e-6:
            return None  # Ray is parallel to the plane

        t = np.dot(plane_point - ray_origin, plane_normal) / denom
        if t <= 0:
            return None  # Intersection is behind the ray origin

        intersection_point = ray_origin + t * ray_direction
        if self._is_point_within_square(intersection_point, plane_point, plane_normal):
            return intersection_point
        else:
            return None

    def _is_point_within_square(self, point, plane_point, plane_normal):
        """
        Check if a point is within the bounds of the square.

        Parameters:
        point (np.array): The point to check.
        plane_point (np.array): A point on the plane.
        plane_normal (np.array): The normal vector of the plane.

        Returns:
        bool: True if the point is within the square, False otherwise.
        """
        if np.allclose(plane_normal, [0, 0, 1]) or np.allclose(plane_normal, [0, 0, -1]):
            in_plane_x = plane_point[0] <= point[0] <= plane_point[0] + self.square_size
            in_plane_y = plane_point[1] <= point[1] <= plane_point[1] + self.square_size
            return in_plane_x and in_plane_y
        else:
            # For the second plane, transformed coordinates need to be checked
            rotated_point = self._rotate_point_back_to_plane(point, plane_normal)
            in_plane_x = 0 <= rotated_point[0] <= self.square_size
            in_plane_y = 0 <= rotated_point[1] <= self.square_size
            return in_plane_x and in_plane_y

    def _get_channel(self, point, plane_point, plane_normal):
        """
        Determine the channel number of the point within the grid.

        Parameters:
        point (np.array): The intersection point.
        plane_point (np.array): A point on the plane.
        plane_normal (np.array): The normal vector of the plane.

        Returns:
        int: The channel number (1 to 16).
        """
        if np.allclose(plane_normal, [0, 0, 1]) or np.allclose(plane_normal, [0, 0, -1]):
            local_x = point[0] - plane_point[0]
            local_y = point[1] - plane_point[1]
        else:
            rotated_point = self._rotate_point_back_to_plane(point, plane_normal)
            local_x = rotated_point[0]
            local_y = rotated_point[1]

        channel_x = int(local_x // (self.square_size / 4))
        channel_y = int(local_y // (self.square_size / 4))
        return channel_y * 4 + channel_x + 1

    def _rotate_point_back_to_plane(self, point, plane_normal):
        """
        Rotate a point back to the coordinate system of the plane.

        Parameters:
        point (np.array): The point to rotate.
        plane_normal (np.array): The normal vector of the plane.

        Returns:
        np.array: The rotated point.
        """
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(-self.angle_radians), -np.sin(-self.angle_radians)],
            [0, np.sin(-self.angle_radians), np.cos(-self.angle_radians)]
        ])
        return rotation_matrix @ point

def visualize_geometry(photon, geometry):
    """
    Visualize the geometry of the setup and the photon's path.
    
    Parameters:
    photon (Photon): The photon object to visualize.
    geometry (BookGeometry): The geometry of the setup.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def draw_square_with_grid(square_origin, rotation_matrix):
        """
        Draw a square with a 4x4 grid on it.
        
        Parameters:
        square_origin (np.array): The origin point of the square.
        rotation_matrix (np.array): The rotation matrix for the square orientation.
        """
        square_size = geometry.square_size
        grid_size = square_size / 4

        for i in range(5):
            # Draw grid lines parallel to x-axis
            start_x = square_origin + rotation_matrix @ np.array([0, i * grid_size, 0])
            end_x = square_origin + rotation_matrix @ np.array([square_size, i * grid_size, 0])
            ax.plot([start_x[0], end_x[0]], [start_x[1], end_x[1]], [start_x[2], end_x[2]], 'k--')

            # Draw grid lines parallel to y-axis
            start_y = square_origin + rotation_matrix @ np.array([i * grid_size, 0, 0])
            end_y = square_origin + rotation_matrix @ np.array([i * grid_size, square_size, 0])
            ax.plot([start_y[0], end_y[0]], [start_y[1], end_y[1]], [start_y[2], end_y[2]], 'k--')

    # Draw the first square (bottom)
    square_origin1 = np.array([0, 0, 0])
    rotation_matrix1 = np.eye(3)
    draw_square_with_grid(square_origin1, rotation_matrix1)

    # Draw the second square (rotated)
    rotation_matrix2 = np.array([
        [1, 0, 0],
        [0, np.cos(geometry.angle_radians), -np.sin(geometry.angle_radians)],
        [0, np.sin(geometry.angle_radians), np.cos(geometry.angle_radians)]
    ])
    square_origin2 = square_origin1  # Assuming both squares share the same origin point for simplicity
    draw_square_with_grid(square_origin2, rotation_matrix2)

    # Draw the photon's trajectory
    ax.quiver(photon.position[0], photon.position[1], photon.position[2],
              photon.direction[0], photon.direction[1], photon.direction[2],
              length=50, color='r', arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.set_zlim([0, 50])
    plt.show()

# Example usage
geometry = BookGeometry(30)
#photon = Photon(25, 10, 0, 1, 1, 0.01, geometry)
photon = Photon(20, 23, 0, 0, 0, 1, geometry)
visualize_geometry(photon, geometry)
print(f"Photon status: {photon.status}, Surface: {photon.surface}, Channel: {photon.channel}")
photon.check_and_update_status(geometry)
visualize_geometry(photon, geometry)

print(f"Photon status: {photon.status}, Surface: {photon.surface}, Channel: {photon.channel}")

