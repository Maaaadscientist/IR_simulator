import os, sys
import numpy as np
import random
import argparse

from simulator import Photon, BookGeometry,visualize_geometry 
from reflectance import ReflectanceCalculator
from borel_random import generate_random_borel
import ROOT
# Example usage
def parse_arguments():
    parser = argparse.ArgumentParser(description='Photon Simulation Parameters')
    parser.add_argument('--N', type=int, default=100, help='Number of tests')
    parser.add_argument('--angle', type=float, default=45, help='Number of tests')
    parser.add_argument('--ext', type=int, default=1, help='turn on external crosstalk')
    parser.add_argument('--fix_eta', type=float, default=None, help='turn off eta random')
    parser.add_argument('--output', type=str, default="output.root", help='output string')
    args = parser.parse_args()
    return args.N, args.angle, args.ext, args.fix_eta, args.output

N, angle, ext, fix_eta, output= parse_arguments()
# Parse command line arguments
if fix_eta == None:
    geometry = BookGeometry(angle)
else:
    geometry = BookGeometry(angle, np.radians(fix_eta))
#def generate_and_propagate_photons(x,y,z,lambda_=5, pde=0.1, init_fire=True):
#    for _ in range(np.random.poisson(lambda_)):
#        dx, dy, dz = geometry.generate_isotropic_direction(geometry.get_normal_at_position(np.array([x,y,z])))
#    
#        photon = Photon(x, y, z, dx, dy, dz, geometry)
#        if init_fire:
#            surface_position = photon.surface_position
#            print(surface_position,photon.surface, 0)
#            fired_channels.append((surface_position,photon.surface, 0))
#            init_fire = False
#        while (photon.status == "active"):
#            photon.check_and_update_status(geometry)
#            if photon.status == 'hit':
#                if np.random.random() < pde:
#                    visualize_geometry(photon, geometry)
#                    fired_channels.append((surface_position,photon.surface, 1))
#                    surface_position = photon.surface_position
#                    print(surface_position,photon.surface, 1)
#                    x,y,z = photon.position
#                    generate_and_propagate_photons(x,y,z, init_fire=False)
def generate_and_propagate_photons(x, y, z, event_idx, lambda_=5, pde=0.1, init_fire=True):
    local_fired_channels = []
    for _ in range(np.random.poisson(lambda_)):
        dx, dy, dz = geometry.generate_isotropic_direction(geometry.get_normal_at_position(np.array([x, y, z])))
        photon = Photon(x, y, z, dx, dy, dz, geometry)
        surface_x, surface_y = photon.surface_position
        if init_fire:
            local_fired_channels.append((surface_x, surface_y, photon.surface, 1))  # Initial fire
            init_fire = False
        while photon.status == "active":
            photon.check_and_update_status(geometry)
            if photon.status == 'hit':
                if np.random.random() < pde:
                    surface_x, surface_y = photon.surface_position
                    #visualize_geometry(photon, geometry)
                    
                    local_fired_channels.append((surface_x, surface_y, photon.surface, 0))  # Subsequent fires
                    x, y, z = photon.position
    return local_fired_channels


def generate_and_propagate_photons_noExt(x,y,z):
    photon = Photon(x, y, z, 0, 0, 1, geometry)
    fired_channels.append((photon.surface, photon.channel))
#print(geometry.random_dark_event_position())
#photon = Photon(25, 10, 0, 1, 1, 0.01, geometry)
seed = random.randint(0, 2**32 - 1)
last_six_digits = seed % 1000000
np.random.seed(seed)
output_path = os.path.abspath(output)
output_root_path = '/'.join(output_path.split('/')[0:-1])
print(output_root_path)

if not os.path.isdir(output_root_path):
    os.makedirs(output_root_path)
f1 = ROOT.TFile(f"{output}","recreate")
tree = ROOT.TTree("tree", "Photon Data per Event")

# Define branches
event_id = np.zeros(1, dtype=int)
photon_x = ROOT.std.vector('float')()
photon_y = ROOT.std.vector('float')()
surface_ids = ROOT.std.vector('int')()
init_fire_index = ROOT.std.vector('int')()

tree.Branch("event_id", event_id, "event_id/I")
tree.Branch("photon_x", photon_x)
tree.Branch("photon_y", photon_y)
tree.Branch("surface_ids", surface_ids)
tree.Branch("init_fire_index", init_fire_index)

crosstalk = 0.15
if ext != 0: 
    for i in range(N):
        print(f"event: {i}")
        if i % 100 == 0:
            print(f"{i} events generated")

        # Clear vectors for each event
        photon_x.clear()
        photon_y.clear()
        surface_ids.clear()
        init_fire_index.clear()

        fired_channels = []
        for _ in range(10):  # Expecting 10 photons per event
            x, y, z = geometry.random_light_event_center(10, 2)
            fired_channels.extend(generate_and_propagate_photons(x, y, z, i))

        # Fill the TTree with aggregated data for this event
        event_id[0] = i
        for x, y, surface, init_fire in fired_channels:
            photon_x.push_back(x)
            photon_y.push_back(y)
            surface_ids.push_back(surface)
            init_fire_index.push_back(init_fire)
        
        tree.Fill()

# Write and close ROOT file
f1.Write()
f1.Close()
