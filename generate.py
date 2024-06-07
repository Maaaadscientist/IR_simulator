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
    parser.add_argument('--angle', type=float, default=30, help='Number of tests')
    parser.add_argument('--ext', type=int, default=True, help='turn on external crosstalk')
    args = parser.parse_args()
    return args.N, args.angle, args.ext

N, angle, ext= parse_arguments()
# Parse command line arguments
geometry = BookGeometry(angle)
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
                if np.random.random() < pde:
                    fired_channels.append((photon.surface, photon.channel))
                    x,y,z = photon.position
                    generate_and_propagate_photons(x,y,z, init_fire=False)

def generate_and_propagate_photons_noExt(x,y,z):
    photon = Photon(x, y, z, 0, 0, 1, geometry)
    fired_channels.append((photon.surface, photon.channel))
#print(geometry.random_dark_event_position())
#photon = Photon(25, 10, 0, 1, 1, 0.01, geometry)
seed = random.randint(0, 2**32 - 1)
last_six_digits = seed % 1000000
np.random.seed(seed)
if not os.path.isdir(f"degree{int(angle)}"):
    os.makedirs(f"degree{int(angle)}")
f1 = ROOT.TFile(f"degree{int(angle)}/output_30degree_seed{last_six_digits}.root","recreate")
h1 = ROOT.TH1F("all_channel", "all_channel", 32, 0, 32)
crosstalk = 0.15
if ext != 0: 
    for i in range(N):
        if i%100 == 0:
            print(f"{i} events Generated") 
        fired_channels = []
        x,y,z = geometry.random_dark_event_position()
        generate_and_propagate_photons(x,y,z)
        for sipm, ch in fired_channels:
            h1.Fill((sipm-1)*16 + ch -1, generate_random_borel(crosstalk, 20))
else:
    for i in range(N):
        if i%100 == 0:
            print(f"{i} events Generated without external crosstalk") 
        fired_channels = []
        x,y,z = geometry.random_dark_event_position()
        generate_and_propagate_photons_noExt(x,y,z)
        for sipm, ch in fired_channels:
            h1.Fill((sipm-1)*16 + ch -1, generate_random_borel(crosstalk, 20))
    
h1.Write()
f1.Close()


