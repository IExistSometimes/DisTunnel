import numpy as np
import pandas as pd
import os
import math
import scipy
from shapely.geometry import Polygon
import meshio
from scipy.stats import gaussian_kde
rng = np.random.default_rng(12345678909876543211234567890987654321)
import cv2
import ndlsp as nd

def determine_waviness(point_data_array, set_data_array, max_wavelength = 3, min_wavelength = 0.5, wavelength_resolution = 0.1, num_bonus_wavelengths = 11, subset_size = 2000, quantile = 0.95, repeats = 20):
    #Generates a list of waviness lists, with each waviness list consisting of an array of x frequencies, an array of y frequencies, an array of phase angles at the origin, an array of amplitudes, and the origin position for the analysis
    waviness_packet = []
    for i in range(len(set_data_array)):
        #Prepare the important data for the set
        read_start = int(sum(set_data_array[0:i,2]))
        read_end = int(sum(set_data_array[0:i+1,2]))
        #print(set_data_array[i,0], set_data_array[i,1])
        points = point_data_array[read_start:read_end, 0:3]
        
        #The periodogram function is pretty intensive for large numbers of points, so set an optional limit and take a subset for clusters with too many points
        if read_end-read_start > subset_size:
            points = points[rng.choice(len(points), size=np.min((subset_size,len(points)-1)), replace=False),:]

        #Transform into our standard cluster reference frame (down-dip vector being positive y)
        set_dip, set_dipdir = abcd_to_dipdipdir(set_data_array[i, 3],set_data_array[i, 4], set_data_array[i, 5])
        set_centroid = set_data_array[i,7:]
        points_cluster_frame = rotate_points(points,set_dipdir, set_dip, centre=set_centroid, return_to_centre = False).transpose()#Rotate to be flat with the downdip direction being +y

        #Find an estimate of the Nyquist frequency (or its equivalent for the Lomb-Scargle periodogram)
        hull = scipy.spatial.ConvexHull(points_cluster_frame[0:2,:].transpose())
        area = hull.volume
        
        #Average distance between adjacent points assuming uniform distribution is just the square root of the area divided by the number of points
        spacing = np.sqrt(area/len(points))
        
        #The highest observable frequency in cycles per distance should be the frequency which has half a cycle per average spacing
        #Therefore its wavelength is twice the average spacing
        Nyquist_wavelength = 2*spacing
        
        #Don't bother being clever, space the searched wavelengths in the region of interest according to the resolution
        #The results are rotationally symmetric order 2 about the origin, so we only need to check a haf-space
        
        #Also add bonus frequencies around 0, these allow detection of waviness parallel or normal to dip
        #Also add some small negative frequencies, these are necessary for the peak detection to work consistently
        
        min_wavelength = max((min_wavelength,Nyquist_wavelength))
        wavelengths = np.arange(min_wavelength, max_wavelength, wavelength_resolution)
        wavelengths = np.concatenate((wavelengths, np.array([max_wavelength])))
        frequencies = 1/wavelengths
        
        bonus_frequencies = np.linspace(-1/np.max(wavelengths), 1/np.max(wavelengths), num_bonus_wavelengths)

        x_frequencies = np.sort(np.concatenate((bonus_frequencies, frequencies)))
        pos_frequencies = x_frequencies[x_frequencies>0]
        y_frequencies = np.sort(np.concatenate((-1*pos_frequencies, np.array([0]), pos_frequencies)))
        fs = [x_frequencies, y_frequencies]
        
        #Run the actual periodogram
        A, phis, inner_prods = nd.lsp_nd(points_cluster_frame[0:2,:], points_cluster_frame[2,:], fs, retrieve_orthogonality=True)    

        #Determine the quantile we are using for peak detection
        A95 = nd.findQuantile(points_cluster_frame[0:2,:], points_cluster_frame[2,:], fs, q = quantile, N = repeats)

        #Detect the peaks
        amplitudes, (x_peak_freqs, y_peak_freqs) = nd.findPeaks(A, A95, fs, factor = 3)

        peak_phis = np.array([])
        for j in range(len(x_peak_freqs)):
            peak_phis = np.append(peak_phis, phis[x_frequencies==x_peak_freqs[j], y_frequencies==y_peak_freqs[j]])

        #Remove degenerate peaks, first by marking them for deletion by setting their x frequency outside of any reasonable range
        for j in range(len(x_peak_freqs)):
            for k in range(len(x_peak_freqs)):
                if k>j and (abs(y_peak_freqs[j]+y_peak_freqs[k])<0.001 and abs(x_peak_freqs[j]+x_peak_freqs[k])<0.001):
                    x_peak_freqs[k] = -999

        amplitudes = np.delete(amplitudes, x_peak_freqs==-999)
        peak_phis = np.delete(peak_phis, x_peak_freqs==-999)
        y_peak_freqs = np.delete(y_peak_freqs, x_peak_freqs==-999)
        x_peak_freqs = np.delete(x_peak_freqs, x_peak_freqs==-999)

        


        waviness_list = [x_peak_freqs, y_peak_freqs, peak_phis, amplitudes, set_centroid]
        waviness_packet.append(waviness_list)

    return waviness_packet