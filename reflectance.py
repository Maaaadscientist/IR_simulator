import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class ReflectanceCalculator:
    def __init__(self, file_path):
        # Load the data
        reflectance_data = pd.read_csv(file_path)

        # Ensure the data is sorted by wavelength and angle
        reflectance_data = reflectance_data.sort_values(by=['wavelength', 'degree'])

        # Extract unique angles and wavelengths
        self.angles = np.sort(reflectance_data['degree'].unique())
        self.wavelengths = np.sort(reflectance_data['wavelength'].unique())

        # Create a grid of reflectance values
        reflectance_grid = reflectance_data.pivot(index='wavelength', columns='degree', values='reflectance').values

        # Create the interpolation function
        self.interpolator = RegularGridInterpolator((self.wavelengths, self.angles), reflectance_grid)

    def get_reflectance(self, wavelength, angle_rad):
        # Convert the angle from radians to degrees
        angle_deg = np.degrees(angle_rad)

        # Clamp the angle to 8 degrees if less than 8, and to 60 degrees if greater than 60
        if angle_deg < 8:
            angle_deg = 8
        elif angle_deg > 60:
            angle_deg = 60

        # Clamp the wavelength to the nearest boundary if out of bounds
        wavelength = np.clip(wavelength, self.wavelengths[0], self.wavelengths[-1])

        return self.interpolator((wavelength, angle_deg))

