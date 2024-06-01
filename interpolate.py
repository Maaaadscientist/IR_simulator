import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.style.use('nature')

scale = 1.5
labelsize=28
titlesize=40
textsize=24
size_marker = 100

labelsize *= scale
titlesize*= scale
textsize*=scale
size_marker*=scale
# Set global font sizes
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (24,15)
plt.rcParams['font.size'] = textsize  # Sets default font size
plt.rcParams['axes.labelsize'] = labelsize
plt.rcParams['axes.titlesize'] = titlesize
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['legend.fontsize'] = labelsize
plt.rcParams['errorbar.capsize'] = 4
plt.rcParams['lines.markersize'] = 10  # For example, 8 points
plt.rcParams['lines.linewidth'] = 1.5 # For example, 2 points
# Set global parameters using rcParams
plt.rcParams['axes.titlepad'] = 20  # Padding above the title
plt.rcParams['axes.labelpad'] = 15  # Padding for both x and y axis labels


# Load the data
file_path = 'reflectance.csv'
reflectance_data = pd.read_csv(file_path)

# Ensure the data is sorted by wavelength and angle
reflectance_data = reflectance_data.sort_values(by=['wavelength', 'degree'])

# Extract unique angles and wavelengths
angles = np.sort(reflectance_data['degree'].unique())
wavelengths = np.sort(reflectance_data['wavelength'].unique())

# Create a grid of reflectance values
reflectance_grid = reflectance_data.pivot(index='wavelength', columns='degree', values='reflectance').values

# Create the interpolation function
interpolator = RegularGridInterpolator((wavelengths, angles), reflectance_grid)

#def get_reflectance(wavelength, angle):
#    return interpolator((wavelength, angle))
def get_reflectance(wavelength, angle):
    # Clamp the wavelength and angle to the nearest boundary if out of bounds
    wavelength = np.clip(wavelength, wavelengths[0], wavelengths[-1])
    angle = np.clip(angle, angles[0], angles[-1])
    
    return interpolator((wavelength, angle))

# Example usage
#wavelength = 405
#angle = 15
#reflectance = get_reflectance(wavelength, angle)
#print(f'Reflectance at {wavelength} nm and {angle} degrees: {reflectance}')


# Ensure data is properly sorted and available
#angles = np.sort(reflectance_data['degree'].unique())
#wavelengths = np.sort(reflectance_data['wavelength'].unique())
#reflectance_grid = reflectance_data.pivot(index='wavelength', columns='degree', values='reflectance').values

# Create a meshgrid for plotting
X, Y = np.meshgrid(angles, wavelengths)
Z = reflectance_grid
# Define a regular grid
xi = np.linspace(X.min(), X.max(), 500)
yi = np.linspace(Y.min(), Y.max(), 500)
xi, yi = np.meshgrid(xi, yi)
# Interpolate to the regular grid
zi = get_reflectance(yi, xi)

# Create the imshow plot
cp = plt.imshow(zi, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis', aspect='auto')

#cp = plt.contour(X,Y,Z, levels=100, cmap='viridis')
plt.colorbar(cp, label='Reflectance')
#plt.title('Reflectance as a function of Wavelength and Incident Angle')
plt.title(' ')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Wavelength (nm)')
plt.savefig('ref.pdf')
