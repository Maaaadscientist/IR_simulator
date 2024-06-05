import numpy as np

from simulator import Photon, BookGeometry,visualize_geometry 
from reflectance import ReflectanceCalculator

# Example usage
geometry = BookGeometry(30)
#print(geometry.random_dark_event_position())
#photon = Photon(25, 10, 0, 1, 1, 0.01, geometry)
x,y,z = geometry.random_dark_event_position()
dx, dy, dz = geometry.generate_isotropic_direction(geometry.get_normal_at_position(np.array([x,y,z])))

print(dx,dy,dz)
#photon = Photon(20, 23, 0, 0, 0, 1, geometry)
#print(x,y,z)
#if z > 0:
#    dz = -1
#else:
#    dz = 1
# Initialize the ReflectanceCalculator with the path to your CSV file
file_path = 'reflectance.csv'
reflectance_calculator = ReflectanceCalculator(file_path)

# Example usage within the simulation
wavelength = 405
angle_rad = np.radians(15)  # Example angle in radians
reflectance = reflectance_calculator.get_reflectance(wavelength, angle_rad)
print(f'Reflectance at {wavelength} nm and {np.degrees(angle_rad)} degrees: {reflectance}')

photon = Photon(x, y, z, dx, dy, dz, geometry)
while (photon.status != "absorbed"):
    print(f"Photon status: {photon.status}, Surface: {photon.surface}, Channel: {photon.channel}, direction: {photon.direction}")
    print(f"Photon position: {photon.position}, direction: {photon.direction}")
    visualize_geometry(photon, geometry)
    photon.check_and_update_status(geometry)



