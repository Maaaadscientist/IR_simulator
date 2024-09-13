import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from reflectance import ReflectanceCalculator

class Photon:
    def __init__(self, x, y, z, dx, dy, dz, geometry, wavelength=900):
        self.position = np.array([x, y, z])
        self.direction = np.array([dx, dy, dz])
        self.status = 'active'  # 'active' or 'absorbed'
        self.surface, self.channel = self.init_surface_and_channel(geometry)
        self.wavelength = wavelength
        #self.reflectance_calculator = ReflectanceCalculator('reflectance.csv')
        self.reflectance_calculator = ReflectanceCalculator('/Users/allen/codes/IR_simulator/reflectance.csv')
        self.surface_position = self.calculate_surface_position()
    
    def calculate_surface_position(self):
        x, y, z = self.position
        if z == 0:
            # Photon is on the bottom surface
            return np.array([x, y])
        elif z > -1e-10:
            # Photon is on the top surface, recalculate y
            y_prime = np.sqrt(y**2 + z**2)
            return np.array([x, y_prime])
        else:
            raise ValueError("Unexpected z-coordinate for the photon. z should be >= 0.")

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
            incident_angle = np.arccos(np.dot(-self.direction, normal))
            self.direction = self.reflect(self.direction, normal)
            self.surface = surface
            self.channel = channel
            self.surface_position = self.calculate_surface_position()
            
            # Calculate the incident angle
            reflectance = self.reflectance_calculator.get_reflectance(self.wavelength, incident_angle)
            # Determine if the photon is still active based on reflectance
            if np.random.random() > reflectance:
                #print(f'angle: {incident_angle/ np.pi} * pi and reflectance:{reflectance}')
                self.status = 'hit'
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
    def __init__(self, angle_degrees, fixed_eta=None, dcr_file=None):
        """
        Initialize the BookGeometry with the angle between the two squares
        and a csv file of DCR.

        Parameters:
        angle_degrees (float): The angle between the two squares in degrees.
        dcr_file (string): The path of the dcr file.
        """
        self.angle_radians = np.radians(angle_degrees)
        self.square_size = 50.0  # size of the squares in mm
        self.center_distance = 200
        self.squares = self._compute_square_planes()
        self.dcr = np.full(32, 5600.0)  # default DCR value
        self.crosstalk = np.full(32, 0.15)
        if dcr_file and os.path.exists(dcr_file):
            self._read_dcr_from_csv(dcr_file)
        self._generate_cumulative_dcr()
        self.fixed_eta = fixed_eta

    def _compute_square_planes(self):
        """
        Compute the planes representing the two squares.
   
        Returns:
        list of tuples: Each tuple contains a point on the plane, the normal vector of the plane, and the surface number.
        """
        # Square 1: Centered at (0, 0, 0)
        center_point1 = np.array([0, 0, 0])
        normal1 = np.array([0, 0, 1])  # Normal pointing upwards along z-axis
        surface1 = 1
   
        # Square 2: Centered at (0, 0, center_distance) and rotated around x-axis
        center_point2 = np.array([0, 0, self.center_distance])
   
        # Define the normal vector before rotation
        normal2 = np.array([0, 0, -1])  # Initially pointing down towards Surface 1
   
        # Create rotation matrix around the x-axis
        rotation_axis = np.array([1, 0, 0])  # Rotate around x-axis
        rotation_matrix = self._rotation_matrix_around_axis(rotation_axis, self.angle_radians)
   
        # Rotate the normal vector
        normal2 = rotation_matrix @ normal2
   
        # Surface2's point is the center point after rotation
        point2 = center_point2
   
        surface2 = 2
   
        return [(center_point1, normal1, surface1), (point2, normal2, surface2)]

    def _rotation_matrix_around_axis(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.

        Parameters:
        axis (np.array): The axis to rotate around (should be a unit vector).
        theta (float): The rotation angle in radians.

        Returns:
        np.array: The rotation matrix.
        """
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)

        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

        return np.array([
            [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
        ])

    def _get_plane_axes(self, plane_normal):
        if np.allclose(plane_normal, [0, 0, 1]) or np.allclose(plane_normal, [0, 0, -1]):
            u = np.array([1, 0, 0])
            v = np.array([0, 1, 0])
        else:
            u = np.cross(plane_normal, [0, 0, 1])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(plane_normal, [0, 1, 0])
            u /= np.linalg.norm(u)
            v = np.cross(plane_normal, u)
        return u, v
    
    def _compute_local_coordinates(self, point, plane_point, plane_normal):
        u, v = self._get_plane_axes(plane_normal)
        local_vector = point - plane_point
        local_x = np.dot(local_vector, u)
        local_y = np.dot(local_vector, v)
        return local_x, local_y

    
    def _local_to_global(self, local_x, local_y, plane_point, plane_normal):
        u, v = self._get_plane_axes(plane_normal)
        global_point = plane_point + local_x * u + local_y * v
        return global_point

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
    def random_light_event_position(self, channel_index, sigma):
        sipm = channel_index // 16 + 1
        channel = channel_index % 16 + 1
        return self._random_gaussian_position_in_channel(sipm, channel, sigma)

    def random_light_event_center(self, sipm, sigma):
        return self._random_gaussian_position_in_center(sipm, sigma)


    def _random_position_in_channel(self, sipm, channel):
        grid_size = self.square_size / 4
        local_x = ((channel - 1) % 4) * grid_size + np.random.random() * grid_size - self.square_size / 2
        local_y = ((channel - 1) // 4) * grid_size + np.random.random() * grid_size - self.square_size / 2
        local_point = np.array([local_x, local_y, 0])
    
        if sipm == 1:
            global_point = local_point  # Since plane_point is at (0,0,0)
        else:
            global_point = self._rotate_point_back_to_plane(local_point)
        return global_point

    def _random_gaussian_position_in_channel(self, sipm, channel, sigma):
        # Calculate the center of the given channel
        center_x = ((channel - 1) % 4) * (self.square_size / 4) + (self.square_size / 8) - self.square_size / 2
        center_y = ((channel - 1) // 4) * (self.square_size / 4) + (self.square_size / 8) - self.square_size / 2
    
        # Generate a random position around the center using a Gaussian distribution
        local_x = np.random.normal(center_x, sigma)
        local_y = np.random.normal(center_y, sigma)
        local_point = np.array([local_x, local_y, 0])
    
        if sipm == 1:
            global_point = local_point
        else:
            global_point = self._rotate_point_back_to_plane(local_point)
        return global_point

    def _random_gaussian_position_in_center(self, sipm, sigma):
        # Center is now at (0, 0)
        local_x = np.random.normal(0.0, sigma)
        local_y = np.random.normal(0.0, sigma)
        local_point = np.array([local_x, local_y, 0])
    
        if sipm == 1:
            global_point = local_point
        else:
            global_point = self._rotate_point_back_to_plane(local_point)
        return global_point

    #def get_normal_at_position(self, position):
    #    for point, normal, surface in self.squares:
    #        if self._is_point_within_square(position, point, normal):
    #            return normal
    #    return None
    def get_normal_at_position(self, position):
        for point, normal, surface in self.squares:
            is_within = self._is_point_within_square(position, point, normal)
            if is_within:
                return normal
        return None

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
        if not self.fixed_eta is None:
            theta = np.arccos(self.fixed_eta)
        else:
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
        """
        Returns the rotation matrix that rotates vec1 to vec2.
        """
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
    
        c = np.dot(a, b)
        if c >= 1.0 - 1e-6:
            # Vectors are the same
            return np.eye(3)
        elif c <= -1.0 + 1e-6:
            # Vectors are opposite
            # Find an orthogonal vector to use as rotation axis
            axis = np.array([1, 0, 0])
            if np.abs(a[0]) > 0.9:
                axis = np.array([0, 1, 0])
            axis = axis - a * np.dot(a, axis)
            axis /= np.linalg.norm(axis)
            kmat = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_matrix = -np.eye(3) + 2 * np.outer(axis, axis)
            return rotation_matrix
        else:
            # General case
            v = np.cross(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / s**2)
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
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        denom = np.dot(plane_normal, ray_direction)
        if np.abs(denom) < 1e-6:
            return None
        t = np.dot(plane_point - ray_origin, plane_normal) / denom
        if t <= 1e-10:
            return None
        intersection_point = ray_origin + t * ray_direction
        if self._is_point_within_square(intersection_point, plane_point, plane_normal):
            return intersection_point
        else:
            return None

    def _is_point_within_square(self, point, plane_point, plane_normal):
        local_x, local_y = self._compute_local_coordinates(point, plane_point, plane_normal)
        half_size = self.square_size / 2
        in_plane_x = -half_size <= local_x <= half_size
        in_plane_y = -half_size <= local_y <= half_size
        distance_to_plane = np.abs(np.dot(point - plane_point, plane_normal))
        is_in_plane = distance_to_plane < 1e-6
        return is_in_plane and in_plane_x and in_plane_y

    def _get_channel(self, point, plane_point, plane_normal):
        local_x, local_y = self._compute_local_coordinates(point, plane_point, plane_normal)
        half_size = self.square_size / 2
        local_x += half_size
        local_y += half_size
        channel_x = int(local_x // (self.square_size / 4))
        channel_y = int(local_y // (self.square_size / 4))
        return channel_y * 4 + channel_x + 1

    def _rotate_point_to_plane(self, point):
        """
        Rotate and translate a point from the global coordinate system to the local coordinate system of Surface 2.
        """
        # Translate point to Surface 2's center
        translated_point = point - self.squares[1][0]
        # Apply the inverse rotation (rotate back to local coordinates)
        rotation_axis = np.array([1, 0, 0])
        inverse_rotation_matrix = self._rotation_matrix_around_axis(rotation_axis, -self.angle_radians)
        local_point = inverse_rotation_matrix @ translated_point
        return local_point

    def _rotate_point_back_to_plane(self, local_point):
        """
        Rotate and translate a point from the local coordinate system of Surface 2 back to the global coordinate system.
        """
        # Apply rotation
        rotation_axis = np.array([1, 0, 0])
        rotation_matrix = self._rotation_matrix_around_axis(rotation_axis, self.angle_radians)
        rotated_point = rotation_matrix @ local_point
        # Translate back to global coordinates
        global_point = rotated_point + self.squares[1][0]
        return global_point



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


    # Adjust the draw_square_with_grid function
    
    def draw_square_with_grid(plane_point, plane_normal, ax):
        """
        Draw a square with a 4x4 grid on it.
    
        Parameters:
        plane_point (np.array): The center point of the square.
        plane_normal (np.array): The normal vector of the plane.
        ax (Axes3D): The axis to draw the square on.
        """
        square_size = geometry.square_size
        grid_size = square_size / 4
    
        # Get the plane's local axes u and v
        u, v = geometry._get_plane_axes(plane_normal)
    
        # Build the square corners around the center
        half_size = square_size / 2
        corners = [
            plane_point - u * half_size - v * half_size,  # Bottom-left
            plane_point + u * half_size - v * half_size,  # Bottom-right
            plane_point + u * half_size + v * half_size,  # Top-right
            plane_point - u * half_size + v * half_size,  # Top-left
            plane_point - u * half_size - v * half_size   # Close the loop
        ]
    
        # Draw the square edges
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        z_coords = [corner[2] for corner in corners]
        ax.plot(x_coords, y_coords, z_coords, 'k-')
    
        # Draw grid lines
        for i in range(1, 4):
            fraction = i / 4.0
            # Lines parallel to u
            start = plane_point - u * half_size + v * (-half_size + fraction * square_size)
            end = plane_point + u * half_size + v * (-half_size + fraction * square_size)
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k--')
    
            # Lines parallel to v
            start = plane_point + u * (-half_size + fraction * square_size) - v * half_size
            end = plane_point + u * (-half_size + fraction * square_size) + v * half_size
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k--')


    # Draw the squares
    for plane_point, plane_normal, surface in geometry.squares:
        draw_square_with_grid(plane_point, plane_normal, ax_3d)

    # Draw the photon's trajectory
    photon_length = 50  # Adjust length as needed
    photon_end = photon.position + photon_length * photon.direction

    ax_3d.plot([photon.position[0], photon_end[0]],
               [photon.position[1], photon_end[1]],
               [photon.position[2], photon_end[2]],
               color='r', label='Photon path')

    # Draw the photon's initial position
    if photon.position[2] > geometry.squares[0][0][2] + 1e-6:
        point_color = 'green'
        point_label = 'SiPM rotated'
    else:
        point_color = 'blue'
        point_label = 'SiPM bottom'

    ax_3d.scatter(photon.position[0], photon.position[1], photon.position[2],
                  color=point_color, s=50, label=point_label)

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    # Set the plotting ranges to include both squares and the photon path
    # Compute the min and max values
    all_points = [photon.position, photon_end]

    # Include corners of both squares
    for plane_point, plane_normal, surface in geometry.squares:
        u, v = geometry._get_plane_axes(plane_normal)
        half_size = geometry.square_size / 2
        corners = [
            plane_point - u * half_size - v * half_size,
            plane_point + u * half_size - v * half_size,
            plane_point + u * half_size + v * half_size,
            plane_point - u * half_size + v * half_size
        ]
        all_points.extend(corners)

    all_points = np.array(all_points)
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    min_z, max_z = np.min(all_points[:, 2]), np.max(all_points[:, 2])

    # Determine the maximum range among x, y, z to set equal aspect ratio
    max_range = np.array([max_x - min_x, max_y - min_y, max_z - min_z]).max() / 2.0

    mid_x = (max_x + min_x) * 0.5
    mid_y = (max_y + min_y) * 0.5
    mid_z = (max_z + min_z) * 0.5

    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set equal aspect ratio for 3D plot
    ax_3d.set_box_aspect([1,1,1])  # Requires Matplotlib 3.3+

    ax_3d.legend()

    # Create x-y projection
    ax_xy = fig.add_subplot(222)
    # Draw squares in x-y projection
    for plane_point, plane_normal, surface in geometry.squares:
        u, v = geometry._get_plane_axes(plane_normal)
        # Project square corners onto x-y plane
        square_size = geometry.square_size
        corners = [
            plane_point - u * half_size - v * half_size,
            plane_point + u * half_size - v * half_size,
            plane_point + u * half_size + v * half_size,
            plane_point - u * half_size + v * half_size,
            plane_point - u * half_size - v * half_size,
        ]
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        ax_xy.plot(x_coords, y_coords, 'k--')

    # Draw photon path projection onto x-y plane
    ax_xy.plot([photon.position[0], photon_end[0]],
               [photon.position[1], photon_end[1]], color='r')
    ax_xy.scatter(photon.position[0], photon.position[1], color=point_color,
                  s=50, label=point_label)
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.legend()
    ax_xy.set_title('X-Y Projection')
    ax_xy.set_aspect('equal', adjustable='box')

    # Set limits for x-y projection
    ax_xy.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_xy.set_ylim(mid_y - max_range, mid_y + max_range)

    # Create y-z projection
    ax_yz = fig.add_subplot(223)
    for plane_point, plane_normal, surface in geometry.squares:
        u, v = geometry._get_plane_axes(plane_normal)
        square_size = geometry.square_size
        corners = [
            plane_point - u * half_size - v * half_size,  # Bottom-left
            plane_point + u * half_size - v * half_size,  # Bottom-right
            plane_point + u * half_size + v * half_size,  # Top-right
            plane_point - u * half_size + v * half_size,  # Top-left
            plane_point - u * half_size - v * half_size   # Close the loop
        ]
    
        y_coords = [corner[1] for corner in corners]
        z_coords = [corner[2] for corner in corners]
        ax_yz.plot(y_coords, z_coords, 'k--')

    ax_yz.plot([photon.position[1], photon_end[1]],
               [photon.position[2], photon_end[2]], color='r')
    ax_yz.scatter(photon.position[1], photon.position[2], color=point_color,
                  s=50, label=point_label)
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.legend()
    ax_yz.set_title('Y-Z Projection')
    ax_yz.set_aspect('equal', adjustable='box')

    # Set limits for y-z projection
    ax_yz.set_xlim(mid_y - max_range, mid_y + max_range)
    ax_yz.set_ylim(mid_z - max_range, mid_z + max_range)

    # Create x-z projection
    ax_xz = fig.add_subplot(224)
    for plane_point, plane_normal, surface in geometry.squares:
        u, v = geometry._get_plane_axes(plane_normal)
        square_size = geometry.square_size
        corners = [
            plane_point - u * half_size - v * half_size,  # Bottom-left
            plane_point + u * half_size - v * half_size,  # Bottom-right
            plane_point + u * half_size + v * half_size,  # Top-right
            plane_point - u * half_size + v * half_size,  # Top-left
            plane_point - u * half_size - v * half_size   # Close the loop
        ]
    
        x_coords = [corner[0] for corner in corners]
        z_coords = [corner[2] for corner in corners]
        ax_xz.plot(x_coords, z_coords, 'k--')

    ax_xz.plot([photon.position[0], photon_end[0]],
               [photon.position[2], photon_end[2]], color='r')
    ax_xz.scatter(photon.position[0], photon.position[2], color=point_color,
                  s=50, label=point_label)
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    ax_xz.legend()
    ax_xz.set_title('X-Z Projection')
    ax_xz.set_aspect('equal', adjustable='box')

    # Set limits for x-z projection
    ax_xz.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_xz.set_ylim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()
