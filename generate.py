import numpy as np

from simulator import Photon, BookGeometry,visualize_geometry 
from reflectance import ReflectanceCalculator

# Example usage
geometry = BookGeometry(30)
#print(geometry.random_dark_event_position())
#photon = Photon(25, 10, 0, 1, 1, 0.01, geometry)
fired_channels = []
x,y,z = geometry.random_dark_event_position()
def generate_and_propagate_photons(x,y,z,lambda_=5, pde=0.1, init_fire=True):
    for _ in range(np.random.poisson(lambda_)):
        dx, dy, dz = geometry.generate_isotropic_direction(geometry.get_normal_at_position(np.array([x,y,z])))
    
        photon = Photon(x, y, z, dx, dy, dz, geometry)
        if init_fire:
            fired_channels.append((photon.surface, photon.channel))
            init_fire = False
        while (photon.status == "active"):
            visualize_geometry(photon, geometry)
            photon.check_and_update_status(geometry)
            if photon.status == 'hit':
                print("hit the surface!")
                if np.random.random() < pde:
                    print(f"photon detected, {photon.surface}, {photon.channel}")
                    fired_channels.append((photon.surface, photon.channel))
                    x,y,z = photon.position
                    generate_and_propagate_photons(x,y,z, init_fire=False)
generate_and_propagate_photons(x,y,z)
print(fired_channels)


