import os, sys
import numpy as np
import random
import argparse

from simulator import Photon, BookGeometry,visualize_geometry 
from reflectance import ReflectanceCalculator
from borel_random import generate_random_borel, generate_random_generalized_poisson
import ROOT
# Example usage
def parse_arguments():
    parser = argparse.ArgumentParser(description='Photon Simulation Parameters')
    parser.add_argument('--N', type=int, default=100, help='Number of tests')
    parser.add_argument('--pe', type=int, default=10, help='Number of tests')
    parser.add_argument('--ct', type=float, default=0.15, help='crosstalk parameter')
    parser.add_argument('--red_pde', type=float, default=0.08, help='red light pde')
    parser.add_argument('--emit', type=int, default=5, help='Number of photons emitted')
    parser.add_argument('--angle', type=float, default=45, help='Number of tests')
    parser.add_argument('--distance', type=float, default=200, help='distance of the centers of the two surface')
    parser.add_argument('--ext', type=int, default=1, help='turn on external crosstalk')
    parser.add_argument('--fix_eta', type=float, default=None, help='turn off eta random')
    parser.add_argument('--output', type=str, default="output.root", help='output string')
    args = parser.parse_args()
    return args.N, args.pe, args.ct, args.red_pde, args.emit, args.angle, args.distance, args.ext, args.fix_eta, args.output

N, pe, lambda_, red_pde, photon_emit, angle, distance, ext, fix_eta, output= parse_arguments()
#lambda_ = (1 - 1 / (photon_emit * red_pde)) / 2
# Parse command line arguments
if fix_eta == None:
    geometry = BookGeometry(angle)
else:
    geometry = BookGeometry(angle, np.radians(fix_eta))

def generate_and_propagate_photons(x, y, z, event_idx, photon_emit=photon_emit, pde=red_pde, init_fire=True):
    #lambda_ = (1 - 1 / (photon_emit * red_pde)) / 2
    local_fired_channels = []
    for _ in range(np.random.poisson(photon_emit)):
        dx, dy, dz = geometry.generate_isotropic_direction(geometry.get_normal_at_position(np.array([x, y, z])))
        photon = Photon(x, y, z, dx, dy, dz, geometry)
        #print(photon.status)
        #visualize_geometry(photon, geometry)
        surface_x, surface_y = photon.surface_position
        if init_fire:
            local_fired_channels.append((surface_x, surface_y, photon.surface, 1))  # Initial fire
            init_fire = False
        while photon.status == "active":
            photon.check_and_update_status(geometry)
            #print(photon.status)
            #visualize_geometry(photon, geometry)
            if photon.status == 'hit':
                if np.random.random() < pde:
                    #print(photon.status)
                    #visualize_geometry(photon, geometry)
                    surface_x, surface_y = photon.surface_position
                    pe_ct = generate_random_borel(lambda_)
                    for i in range(pe_ct): 
                        #print("single pixel:", i , "/", pe_ct)
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
tree.Branch("surface_id", surface_ids)
tree.Branch("init_fire_index", init_fire_index)

if ext != 0: 
    for evt in range(N):
        #print(f"event: {i}")
        if evt % 100 == 0:
            print(f"{evt} events generated")

        # Clear vectors for each event
        photon_x.clear()
        photon_y.clear()
        surface_ids.clear()
        init_fire_index.clear()

        fired_channels = []
        prompt_pe=generate_random_generalized_poisson(pe, lambda_)
        for i in range(prompt_pe):  # Expecting 10 photons per event
            #print(i+1,"/",prompt_pe)
            x, y, z = geometry.random_light_event_center(10, 1)
            fired_channels.extend(generate_and_propagate_photons(x, y, z, i))

        # Fill the TTree with aggregated data for this event
        event_id[0] = evt
        for x, y, surface, init_fire in fired_channels:
            photon_x.push_back(x)
            photon_y.push_back(y)
            surface_ids.push_back(surface)
            init_fire_index.push_back(init_fire)
        
        tree.Fill()

# Write and close ROOT file
f1.Write()
f1.Close()
