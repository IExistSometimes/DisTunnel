import numpy as np
import pandas as pd

def import_DSE_files(filestem):
    #Imports the relevant data from DSE files and returns them in the expected array format
    point_header_names = ['x', 'y', 'z', 'Nx', 'Ny', 'Nz', 'joint_set_number', 'cluster_number', 'dip_direction', 'dip'] #Header for DSE xyz-NxNyNz-js-c-dipdir-dip files
    DSE_point_data = pd.read_csv(filestem+' xyz-NxNyNz-js-c-dipdir-dip.txt', sep = '\t', names = point_header_names)
    DSE_point_data = DSE_point_data.sort_values(by=['joint_set_number', 'cluster_number'], ignore_index=True) #Ensure sorted data for convenience
    DSE_point_data_array = DSE_point_data.to_numpy()
    #set_header_names = ['joint_set_number', 'cluster_number', 'total_points', 'a', 'b', 'c', 'd',] #Header for DSE js-c-abcd files
    set_header_names = ['joint_set_number', 'cluster_number', 'total_points', 'a', 'b', 'c', 'd','mystery'] #Header for DSE js-c-abcd files when you have used the merge coplanar surfaces feature
    DSE_set_data = pd.read_csv(filestem+' js-c-abcd.txt', sep ='\t', names=set_header_names)
    DSE_set_data = DSE_set_data.sort_values(by=['joint_set_number', 'cluster_number'], ignore_index=True) #Ensure sorted data for convenience
    DSE_set_data = DSE_set_data.drop(columns='mystery')
    DSE_set_data_array = DSE_set_data.to_numpy()
    
    #Find cluster centroids (no attempt to combat point density variation)
    cluster_centroids = np.zeros((len(DSE_set_data), 3))
    for i in range(len(DSE_set_data)):
        read_start = sum(DSE_set_data.total_points[0:i])
        read_end = sum(DSE_set_data.total_points[0:i+1])
        centroid_x = np.mean(DSE_point_data_array[read_start:read_end, 0])
        centroid_y = np.mean(DSE_point_data_array[read_start:read_end, 1])
        centroid_z = np.mean(DSE_point_data_array[read_start:read_end, 2])
        cluster_centroids[i, 0] = centroid_x
        cluster_centroids[i, 1] = centroid_y
        cluster_centroids[i, 2] = centroid_z
    #DSE_set_data_array.shape
    #cluster_centroids.shape
    DSE_set_data_array = np.concatenate((DSE_set_data_array, cluster_centroids), axis=1)
    return DSE_point_data_array, DSE_set_data_array


def mask_pointcloud_byset(allow_these_sets,point_data_array, set_data_array):
    #Makes subsets of the point data array and set data array which only includes the sets listed in allow_these_sets
    #allow_these_sets is a list of integers which are the discontinuity sets to be returned
    point_indices = [i for i, set_num in enumerate(point_data_array[:,6]) if set_num in allow_these_sets]
    subset_point_data_array = point_data_array[point_indices]
    set_indices =  [i for i, set_num in enumerate(set_data_array[:,1]) if set_num in allow_these_sets]
    subset_set_data_array = set_data_array[set_indices]

    return subset_point_data_array, subset_set_data_array


def mask_pointcloud_bycluster(allow_these_clusters,point_data_array, set_data_array):
    #Makes subsets of the point data array and set data array which only includes the sets listed in allow_these_sets
    #allow_these_sets is a list of integers which are the discontinuity sets to be returned
    point_indices = [i for i, set_num in enumerate(point_data_array[:,7]) if set_num in allow_these_clusters]
    subset_point_data_array = point_data_array[point_indices]
    set_indices =  [i for i, set_num in enumerate(set_data_array[:,2]) if set_num in allow_these_clusters]
    subset_set_data_array = set_data_array[set_indices]

    return subset_point_data_array, subset_set_data_array
    
    
def save_pointcloud(filename, point_data_array, set_data_array):
    point_filename = filename+' xyz-NxNyNz-js-c-dipdir-dip.txt'
    set_filename = filename+' js-c-abcd.txt'
    print('point data array shape',point_data_array.shape)
    print('set data array shape',set_data_array.shape)
    np.savetxt(point_filename, point_data_array, delimiter='\t', fmt='%e\t %e\t %e\t %e\t %e\t %e\t %i\t %i\t %e\t %e')
    np.savetxt(set_filename, set_data_array[:,0:7], delimiter='\t', fmt='%i\t %i\t %i\t %e\t %e\t %e\t %e')


def merge_pointclouds(names_list):
    point_data_arrays = np.zeros((1,10))
    set_data_arrays = np.zeros((1,7))

    set_header_names = ['joint_set_number', 'cluster_number', 'total_points', 'a', 'b', 'c', 'd','mystery'] #Header for DSE js-c-abcd files when you have used the merge coplanar surfaces feature
    point_header_names = ['x', 'y', 'z', 'Nx', 'Ny', 'Nz', 'joint_set_number', 'cluster_number', 'dip_direction', 'dip'] #Header for DSE xyz-NxNyNz-js-c-dipdir-dip files
    running_set_count = int(0)
    for i in range(len(names_list)):
        input_point_data = pd.read_csv(names_list[i]+' xyz-NxNyNz-js-c-dipdir-dip.txt', sep = '\t', names = point_header_names)
        point_data = input_point_data.sort_values(by=['joint_set_number', 'cluster_number'], ignore_index=True) #Ensure sorted data for convenience
        point_data_array = point_data.to_numpy()
        point_data_array[:,6]+=running_set_count
        print('Before append ', point_data_array.shape)
        point_data_arrays = np.append(point_data_arrays, point_data_array, axis=0)
        print('After append ', point_data_arrays.shape)

        input_set_data = pd.read_csv(names_list[i]+' js-c-abcd.txt', sep ='\t', names=set_header_names)
        set_data = input_set_data.sort_values(by=['joint_set_number', 'cluster_number'], ignore_index=True) #Ensure sorted data for convenience
        set_data = set_data.drop(columns='mystery')
        set_data_array = set_data.to_numpy()
        set_data_array[:,0]+=running_set_count
        set_data_arrays = np.append(set_data_arrays, set_data_array, axis = 0)

        running_set_count = int(np.max(point_data_arrays[:,6]))
    #point_data_arrays = np.delete(point_data_arrays, np.s_[0,:])
    #set_data_arrays = np.delete(set_data_arrays, np.s_[0,:])
    point_data_arrays[:,6:8] = point_data_arrays[:,6:8].astype(np.int32)
    set_data_arrays[:,0:3] = set_data_arrays[:,0:3].astype(np.int32)
    return point_data_arrays[1:,:], set_data_arrays[1:,:]
