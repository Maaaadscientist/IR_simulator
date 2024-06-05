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
    def __init__(self, angle_degrees, dcr_file=None):
        """
        Initialize the BookGeometry with the angle between the two squares
        and a csv file of DCR.

        Parameters:
        angle_degrees (float): The angle between the two squares in degrees.
        dcr_file (string): The path of the dcr file.
        """
        self.angle_radians = np.radians(angle_degrees)
        self.square_size = 50.0  # size of the squares in mm
        self.squares = self._compute_square_planes()
        self.dcr = np.full(32, 5600.0)  # default DCR value
        if dcr_file and os.path.exists(dcr_file):
            self._read_dcr_from_csv(dcr_file)
        self._generate_cumulative_dcr()

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
        # Ensure normal2 points towards the first SiPM
        if normal2[2] > 0:
            normal2 = -normal2
        surface2 = 2

        return [(point1, normal1, surface1), (point2, normal2, surface2)]

    def _read_dcr_from_csv(self, dcr_file):
        with open(dcr_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sipm = int(row['sipm']) - 1
                channel = int(row['channel']) - 1
                index = sipm * 16 + channel
                self.dcr[index] = float(row['dcr'])

    def _generate_cumulative_dcr(self):
        self.cumulative_dcr = np.cumsum(self.dcr / np.sum(self.dcr))

    def random_dark_event_position(self):
        channel_index = np.searchsorted(self.cumulative_dcr, np.random.random())
        sipm = channel_index // 16 + 1
        channel = channel_index % 16 + 1
        return self._random_position_in_channel(sipm, channel)

    def _random_position_in_channel(self, sipm, channel):
        local_x = ((channel-1) % 4) * (self.square_size / 4) + np.random.random() * (self.square_size / 4)
        local_y = ((channel-1) // 4) * (self.square_size / 4) + np.random.random() * (self.square_size / 4)
        if sipm == 1:
            return np.array([local_x, local_y, 0])
        else:
            rotated_position = self._rotate_point_to_plane(np.array([local_x, local_y, 0]))
            return rotated_position

    def get_normal_at_position(self, position):
        for point, normal, surface in self.squares:
            if self._is_point_within_square(position, point, normal):
                return normal
        return None

    #def generate_isotropic_direction(self, normal):
    #    phi = np.random.uniform(0, 2 * np.pi)
    #    costheta = np.random.uniform(-1, 1)
    #    theta = np.arccos(costheta)
    #    direction = np.array([
    #        np.sin(theta) * np.cos(phi),
    #        np.sin(theta) * np.sin(phi),
    #        np.cos(theta)
    #    ])

    #    # Rotate the direction to align with the normal vector
    #    rotation_matrix = self._rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)
    #    return rotation_matrix @ direction
    def generate_isotropic_direction(self, normal, angle= 0.5* np.pi):
        """
        Generate a random direction within a cone centered around the given normal vector.

        Parameters:
        normal (np.array): The normal vector.
        angle (float): The maximum angle in radians for the cone.

        Returns:
        np.array: The random direction vector.
        """
        if normal is None:
            raise ValueError("Normal vector is None. Cannot generate isotropic direction.")

        ## Randomly sample theta within the given angle
        #theta = np.random.uniform(0, angle)
        # Randomly sample theta within the given angle using a distribution that favors smaller angles
        theta = np.arccos(np.random.uniform(np.cos(angle), 1))
        #theta = np.arccos(np.cos(angle))
        # Uniformly sample phi between 0 and 2*pi
        phi = np.random.uniform(0, 2 * np.pi)
        #phi = 0#np.random.uniform(0, 2 * np.pi)
        
        # Convert spherical coordinates to Cartesian coordinates
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Rotate the direction to align with the normal vector
        rotation_matrix = self._rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)
        return rotation_matrix @ direction

    def _rotation_matrix_from_vectors(self, vec1, vec2):
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2 + 1e-6))
        return rotation_matrix

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
        if abs(t) <= 1e-10: # precision issue
            return None  # Intersection is behind the ray origin

        intersection_point = ray_origin + t * ray_direction
        if self._is_point_within_square(intersection_point, plane_point, plane_normal):
            return intersection_point
        else:
            return None

    def _is_point_within_square(self, point, plane_point, plane_normal):
        if np.allclose(plane_normal, [0, 0, 1]) or np.allclose(plane_normal, [0, 0, -1]):
            # For surface 1
            z_correct = np.isclose(point[2], 0)
            in_plane_x = 0 <= point[0] <= self.square_size
            in_plane_y = 0 <= point[1] <= self.square_size
            return z_correct and in_plane_x and in_plane_y
        else:
            # For surface 2
            angle = self.angle_radians#np.arccos(np.dot(plane_normal, [0, 0, 1]))  # Calculate the angle from the normal vector
            z_correct = np.isclose(point[2], point[1] * np.tan(angle))
            in_plane_x = 0 <= point[0] <= self.square_size
            in_plane_y = 0 <= point[1] <= self.square_size * np.cos(angle)
            return z_correct and in_plane_x and in_plane_y

    #def _is_point_within_square(self, point, plane_point, plane_normal):
    #    """
    #    Check if a point is within the bounds of the square.

    #    Parameters:
    #    point (np.array): The point to check.
    #    plane_point (np.array): A point on the plane.
    #    plane_normal (np.array): The normal vector of the plane.

    #    Returns:
    #    bool: True if the point is within the square, False otherwise.
    #    """
    #    if np.allclose(plane_normal, [0, 0, 1]) or np.allclose(plane_normal, [0, 0, -1]):
    #        in_plane_x = plane_point[0] <= point[0] <= plane_point[0] + self.square_size
    #        in_plane_y = plane_point[1] <= point[1] <= plane_point[1] + self.square_size
    #        return in_plane_x and in_plane_y
    #    else:
    #        # For the second plane, transformed coordinates need to be checked
    #        rotated_point = self._rotate_point_back_to_plane(point, plane_normal)
    #        in_plane_x = 0 <= rotated_point[0] <= self.square_size
    #        in_plane_y = 0 <= rotated_point[1] <= self.square_size
    #        return in_plane_x and in_plane_y

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

    def _rotate_point_to_plane(self, point):
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(self.angle_radians), -np.sin(self.angle_radians)],
            [0, np.sin(self.angle_radians), np.cos(self.angle_radians)]
        ])
        return rotation_matrix @ point

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

def old_visualize_geometry(photon, geometry):
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
    if photon.position[2] > 1e-10:
        point_color = 'green'
        point_label = 'SiPM rotated'
    else:
        point_color = 'blue'
        point_label = 'SiPM bottom'
    
    ax.scatter(photon.position[0], photon.position[1], photon.position[2], color=point_color, s=10, label=point_label)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.set_zlim([0, 50])
    ax.legend()
    plt.show()
def visualize_geometry(photon, geometry):
    """
    Visualize the geometry of the setup and the photon's path, along with projections onto the x-y, y-z, and x-z planes.
    
    Parameters:
    photon (Photon): The photon object to visualize.
    geometry (BookGeometry): The geometry of the setup.
    """
    fig = plt.figure(figsize=(15, 10))

    # Create 3D plot
    ax_3d = fig.add_subplot(221, projection='3d')

    def draw_square_with_grid(square_origin, rotation_matrix, ax):
        """
        Draw a square with a 4x4 grid on it.
        
        Parameters:
        square_origin (np.array): The origin point of the square.
        rotation_matrix (np.array): The rotation matrix for the square orientation.
        ax (Axes3D): The axis to draw the square on.
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
    draw_square_with_grid(square_origin1, rotation_matrix1, ax_3d)

    # Draw the second square (rotated)
    rotation_matrix2 = np.array([
        [1, 0, 0],
        [0, np.cos(geometry.angle_radians), -np.sin(geometry.angle_radians)],
        [0, np.sin(geometry.angle_radians), np.cos(geometry.angle_radians)]
    ])
    square_origin2 = square_origin1  # Assuming both squares share the same origin point for simplicity
    draw_square_with_grid(square_origin2, rotation_matrix2, ax_3d)

    # Draw the photon's trajectory
    ax_3d.quiver(photon.position[0], photon.position[1], photon.position[2],
                 photon.direction[0], photon.direction[1], photon.direction[2],
                 length=50, color='r', arrow_length_ratio=0.1)
    if photon.position[2] > 1e-10:
        point_color = 'green'
        point_label = 'SiPM rotated'
    else:
        point_color = 'blue'
        point_label = 'SiPM bottom'
    
    ax_3d.scatter(photon.position[0], photon.position[1], photon.position[2], color=point_color, s=50, label=point_label)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_xlim([0, 50])
    ax_3d.set_ylim([0, 50])
    ax_3d.set_zlim([0, 50])
    ax_3d.legend()

    # Create x-y projection
    ax_xy = fig.add_subplot(222)
    #draw_square_with_grid(square_origin1, rotation_matrix1, ax_xy)
    #draw_square_with_grid(square_origin2, rotation_matrix2, ax_xy)
    ax_xy.quiver(photon.position[0], photon.position[1], 20*photon.direction[0],20*photon.direction[1],
    #ax_xy.quiver(photon.position[0],photon.position[0] + photon.direction[0], photon.position[1], photon.position[1]+photon.direction[1],
                 color='r', angles='xy', scale_units='xy', scale=1, headwidth=3)
    ax_xy.scatter(photon.position[0], photon.position[1], color=point_color, s=50, label=point_label)
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.set_xlim([0, 50])
    ax_xy.set_ylim([0, 50])
    ax_xy.legend()
    ax_xy.set_title('X-Y Projection')
    ax_xy.set_aspect('equal', adjustable='box')

    # Create y-z projection
    ax_yz = fig.add_subplot(223)
    #draw_square_with_grid(square_origin1, rotation_matrix1, ax_yz)
    #draw_square_with_grid(square_origin2, rotation_matrix2, ax_yz)
    ax_yz.quiver(photon.position[1], photon.position[2], 20*photon.direction[1],20* photon.direction[2],
                  color='r', angles='xy', scale_units='xy', scale=1, headwidth=3)
    ax_yz.scatter(photon.position[1], photon.position[2], color=point_color, s=50, label=point_label)
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.set_xlim([0, 50])
    ax_yz.set_ylim([0, 50])
    ax_yz.legend()
    ax_yz.set_title('Y-Z Projection')
    ax_yz.set_aspect('equal', adjustable='box')

    # Create x-z projection
    ax_xz = fig.add_subplot(224)
    #draw_square_with_grid(square_origin1, rotation_matrix1, ax_xz)
    #draw_square_with_grid(square_origin2, rotation_matrix2, ax_xz)
    ax_xz.quiver(photon.position[0], photon.position[2], 20*photon.direction[0], 20*photon.direction[2],
                 color='r', angles='xy', scale_units='xy', scale=1, headwidth=3)
    ax_xz.scatter(photon.position[0], photon.position[2], color=point_color, s=50, label=point_label)
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.set_xlim([0, 50])
    ax_xz.set_ylim([0, 50])
    ax_xz.legend()
    ax_xz.set_title('X-Z Projection')
    ax_xz.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

#
## Example usage
#geometry = BookGeometry(30)
##print(geometry.random_dark_event_position())
##photon = Photon(25, 10, 0, 1, 1, 0.01, geometry)
#x,y,z = geometry.random_dark_event_position()
#dx, dy, dz = geometry.generate_isotropic_direction(geometry.get_normal_at_position(np.array([x,y,z])))
#
#print(dx,dy,dz)
##photon = Photon(20, 23, 0, 0, 0, 1, geometry)
##print(x,y,z)
##if z > 0:
##    dz = -1
##else:
##    dz = 1
#photon = Photon(x, y, z, dx, dy, dz, geometry)
#while (photon.status != "absorbed"):
#    print(f"Photon status: {photon.status}, Surface: {photon.surface}, Channel: {photon.channel}, direction: {photon.direction}")
#    print(f"Photon position: {photon.position}, direction: {photon.direction}")
#    visualize_geometry(photon, geometry)
#    photon.check_and_update_status(geometry)
#
#
