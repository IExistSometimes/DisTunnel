import pydfnworks #The dfn package from Los Alamos National Laboratories
from pydfnworks import *
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


def rotate_points(points, azimuthal_angle, polar_angle, centre=[0,0,0], return_to_centre = True): #Rotate the given points by an azimuthal angle (anti-clockwise looking down at the x-y plane) and polar angle (measured down from the positive z axis)
    #This function is such that rotating all points in a cluster by its dip direction and dip in degrees about its centroid returns the cluster rotated with its down-dip vector pointing along (0,1,0)
    #DSE defaults to positive y for north so we will use that convention, DSE however also defaults to dipdir = 180 for north, which we will not adopt
    azimuthal_angle = math.radians(azimuthal_angle)
    polar_angle = math.radians(polar_angle)
    
    yawed_points = np.zeros(points.shape)
    #First translate centre over to the origin
    translated_points = points-centre
    #print(translated_points)
    #Second rotate clockwise about the positive z axis by azimuthal_angle (anticlockwise looking down at the plane x-y plane, positive x to positive y, positive y to negative x)
    yaw_matrix = np.array([[math.cos(azimuthal_angle), -math.sin(azimuthal_angle), 0], [math.sin(azimuthal_angle), math.cos(azimuthal_angle), 0],[0,0,1]]).transpose()
    yawed_points = translated_points @ yaw_matrix
    #print(yawed_points)
    #Third rotate clockwise about the positive x axis by polar_angle (anticlockwise looking at the y-z plane from positive x, positive y to positive z, positive z to negative y)
    pitch_matrix = np.array([[1, 0, 0], [0, math.cos(polar_angle), -math.sin(polar_angle)], [0, math.sin(polar_angle), math.cos(polar_angle)]]).transpose()
    pitched_points = yawed_points @ pitch_matrix
    #print(pitched_points)
    if return_to_centre:
        #finally translate back
        returned_points = pitched_points+centre
        return returned_points
    else:
        return pitched_points


def rotate_points_inverse(points, azimuthal_angle, polar_angle, centre=[0,0,0], return_to_centre = True): #Inverts the rotation operation applied by the rotate_points function when run with the same arguments
    #This function is the same as rotate_points except with the order and sign of rotations reversed, allowing the functions to serve as inverses of eachother
    #DSE defaults to positive y for north so we will use that convention
    azimuthal_angle = math.radians(-azimuthal_angle)
    polar_angle = math.radians(-polar_angle)
    
    yawed_points = np.zeros(points.shape)
    #First translate centre over to the origin
    translated_points = points-centre
    #print(translated_points)
    #Second rotate clockwise about the positive x axis by polar_angle (anticlockwise looking at the y-z plane from positive x, positive y to positive z, positive z to negative y
    pitch_matrix = np.array([[1, 0, 0], [0, math.cos(polar_angle), -math.sin(polar_angle)], [0, math.sin(polar_angle), math.cos(polar_angle)]]).transpose()
    pitched_points = translated_points @ pitch_matrix
    #Third rotate clockwise about the positive z axis by azimuthal_angle (anticlockwise looking down at the plane x-y plane, positive x to positive y, positive y to negative x)
    yaw_matrix = np.array([[math.cos(azimuthal_angle), -math.sin(azimuthal_angle), 0], [math.sin(azimuthal_angle), math.cos(azimuthal_angle), 0],[0,0,1]]).transpose()
    yawed_points = pitched_points @ yaw_matrix
    #print(yawed_points)
    #print(pitched_points)
    if return_to_centre:
        #finally translate back
        returned_points = yawed_points+centre
        return returned_points
    else:
        return yawed_points


def project_to_plane(points, axis = 'z'): #project all points to the plane normal to the specified axis, i.e. set the coordinates in that axis to 0
    projected_points = points.copy()
    if axis == 'z':
        projected_points[:,2] = 0
        return projected_points
    elif axis == 'y':
        projected_points[:,1] = 0
        return projected_points
    elif axis == 'x':
        projected_points[:,0] = 0
        return projected_points
    else:
        raise ValueError('project_to_plane given an axis which is neither x, y, nor z')


def abcd_to_dipdipdir(a,b,c): #convert abcd plane representation to dip and dip direction (d is actually irrelevant for this) (0,1,0) is North
    #The plane's normal vector is (a,b,c)
    if c<0: #if we end up with a lower-hemisphere normal, invert it
        a,b,c = -a,-b,-c

    if c== 0:
        dip = 90
    else:
        dip = math.degrees(math.atan(math.sqrt((a**2)+(b**2))/c))
    if b==0:
        dipdir = 90
    else:
        dipdir = math.degrees(math.atan(a/b))
    #dipdir quadrant correction
    if b< 0:
        dipdir = dipdir+180
    if dipdir >= 360:
        dipdir = dipdir-360
    if dipdir < 0:
        dipdir = 360 + dipdir
    return dip, dipdir
    

def form_convex_hulls(point_data_array, set_data_array, enlargement_factor = 1):
    #returns a list of convex hulls for each cluster containing
    hulls_list = []
    for i in range(len(set_data_array)):
        #Prepare the important data for the set
        read_start = int(sum(set_data_array[0:i,2]))
        read_end = int(sum(set_data_array[0:i+1,2]))
        #print(set_data_array[i,0], set_data_array[i,1])
        points = point_data_array[read_start:read_end, 0:3]
        set_dip, set_dipdir = abcd_to_dipdipdir(set_data_array[i, 3],set_data_array[i, 4], set_data_array[i, 5])
        set_centroid = set_data_array[i,7:]
        points_for_hull = rotate_points(points,set_dipdir, set_dip, centre=set_centroid, return_to_centre = False)[:,0:3] #Rotate to be flat with the downdip direction being +y
        #check for bad non-planarity
        x_variation = np.max(points_for_hull[:,0])-np.min(points_for_hull[:,0])
        y_variation = np.max(points_for_hull[:,1])-np.min(points_for_hull[:,1])
        z_variation = np.max(points_for_hull[:,2])-np.min(points_for_hull[:,2])
        if z_variation > x_variation/2 or z_variation > y_variation/2:
            print('Warning: Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1])+' seems not to lie in the plane it is supposed to')
            #raise ValueError('Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1])+' seems not to lie in the plane it is supposed to')
        points_for_hull = points_for_hull[:,0:2] #Slice off the z coordinates, these will later be replaced with zero to effectively project the surface to its plane
        points_for_hull = points_for_hull*enlargement_factor #enlarge the plane so that clusters which share an edge in reality are more likely to be considered intersecting, and account for loss of points near edges which can occur in DSE
        hulls_list.append(scipy.spatial.ConvexHull(points_for_hull)) #Generate the convex hull
    
    return hulls_list

def resample_convex_hull(initial_hull, target_n_points = 1000):
    #Takes a scipy convex hull object in 2D, and returns a scipy convex hull object which approximates the first but with uniformly spaced points through the whole hull
    #This addresses issues where unusually shaped clusters or those formed by merging coplanar clusters would lead to unacceptable results

    #Take the initial hull
    area = initial_hull.volume
    
    initial_points = initial_hull.points
    initial_vertices = initial_hull.points[initial_hull.vertices,:]
    #print('This better not be 2',initial_vertices.shape[0])
    point_linear_spacing = np.sqrt(area/(target_n_points))
    #print(initial_points.shape)
    #print(initial_vertices.shape)
    hull_path = Path(initial_vertices) #This is a matplotlib class with a useful method used later
    
    #Create a set of uniformly distributed points
    x_min = np.min(initial_vertices[:,0])
    x_max = np.max(initial_vertices[:,0])
    y_min = np.min(initial_vertices[:,1])
    y_max = np.max(initial_vertices[:,1])
    grid_x_values = np.arange(x_min,x_max+1.1*point_linear_spacing,point_linear_spacing)
    grid_y_values = np.arange(y_min,y_max+1.1*point_linear_spacing,point_linear_spacing)
    xs, ys = np.meshgrid(grid_x_values, grid_y_values)
    grid_points = np.stack([xs,ys])
    grid_points = grid_points.reshape(2,xs.shape[0]*xs.shape[1])
    grid_points *= 0.999
    #grid_points = grid_points-np.array([[point_linear_spacing/2],[point_linear_spacing/2]])
    
    #Find those points which lie within the initial convex hull
    interior_points= (grid_points[:,hull_path.contains_points(grid_points.transpose())]).transpose()
    #print(hull_path.contains_points(grid_points.transpose()))
    #print(grid_points[:,hull_path.contains_points(grid_points.transpose())])
    #print(interior_points.shape)
    #Combine that set of points with the vertices of the initial hull
    new_points = np.concatenate([initial_vertices,interior_points], axis=0)
    #print(new_points.shape)
    #Create a new scipy hull object for this set of points
    new_hull = scipy.spatial.ConvexHull(new_points)
    
    #Run a couple of checks
    if new_points.shape[0] < 0.9*target_n_points:
        raise Warning('resample_convex_hull produced a final hull with significantly fewer than the target number of points, target:',target_n_points,'actual:',new_points.shape[0])
    if new_points.shape[0] > 2*target_n_points:
        raise Warning('resample_convex_hull produced a final hull with significantly more than the target number of points, target:',target_n_points,'actual:',new_points.shape[0])
    if new_hull.volume > 1.01*area or new_hull.volume < 0.99*area:
        raise Warning('resample_convex_hull produced a final hull with significantly different area to the initial hull')

    
    return new_hull

def resample_hull_list(hull_list, points_per_hull = 1000):
    new_hull_list = []
    for i in range(len(hull_list)):
        new_hull_list.append(resample_convex_hull(hull_list[i], target_n_points = points_per_hull))
    return new_hull_list

def normalise_aspect_ratio(aspect_ratio, num_points, normalisation_scheme):
    #Post-processing for the aspect ratios to tune down any excessive values usually caused by having too few points

    if normalisation_scheme == 'sqrt':
        if num_points<100 and aspect_ratio>math.exp(num_points/40): #Heuristic for large aspect ratio per number of points, allows high aspect ratios only with the support of a large number of points 
            aspect_ratio = np.sqrt(aspect_ratio)
            print('Anomalous aspect ratio normalised')
    elif normalisation_scheme == 'none':
        filler = 0
    else:
        raise ValueError('normalise_aspect_ratio called with invalid normalisation scheme')
    return aspect_ratio

def hulls_to_cov_matching_ellipses(hulls_list, set_data_array, subset_size = 5000, attempts_per_cluster = 10, aspect_ratio_normalisation_method = 'sqrt'):
    #returns an array containing the radii, aspect ratios and betas of each ellipse
    #Ellipses are generated such that they have the same area, aspect ratio (by covariance method), dip, dipdir, and orientation within the plane (by covariance method) as the original clusters
    radii = []
    aspect_ratios = []
    betas = []
    for i in range(len(hulls_list)):
        area = hulls_list[i].volume #In 2D it is the volume attribute which actually contains the area of the hull, the area attribute contains the perimeter
        hull_points = hulls_list[i].points[hulls_list[i].vertices]
        cluster_points = hulls_list[i].points
        #I'm thinking I take a subset of points from all the points in the cluster rather than the points in the hull
        #Idea being that the points in the cluster should have a more uniform density distribution
        found_angle = False
        angle_attempts_made = 0
        while found_angle == False:
            
            if len(cluster_points)<subset_size:
                subset_points = cluster_points.transpose()
                #angle = math.degrees(math.atan(scipy.stats.linregress(subset_points[:,0], subset_points[:,1]).slope))
                eigvals, eigvecs = np.linalg.eig(np.cov(subset_points))
                semi_major = np.argwhere(eigvals==np.max(eigvals))[0,0] #Indices for the semi-major and semi-minor axes
                semi_minor = np.argwhere(eigvals==np.min(eigvals))[0,0]
                angle = math.degrees(math.atan(eigvecs[0,semi_major]/eigvecs[1,semi_major])) #angle clockwise from y+ for the semi-major axis
                aspect_ratio = np.sqrt(eigvals[semi_major]/eigvals[semi_minor]) #Verifying the semi-major and semi-minor is strictly unneccessary here, but because it is necessary when we use the subsets
                found_angle = True

                
            else:

                #If the cluster is so large that we don't want to use all of its points for performance reasons, we can instead take some random subsets of its points
                #and process those, checking to make sure the three subsets have approximately the same characteristics
                #This is mostly a hedge against very large datasets with a large number of large fractures, or datasets with extremely high point density
                #This approach is vulnerable to issues if a cluster has the vast majority of its points in one place, and a small number of points a significant distance away, as this
                #produces a hull with a very large area but potentially misses the extraneous points for the aspect ratio and orientation calculations. Large subset sizes reduce this chance significantly
                #but one can avoid generating these clusters in the first place by trimming small clusters before merging coplanar in DSE

                
                subset_points_1 = cluster_points[rng.integers(0, len(cluster_points), size=subset_size),:].transpose() #Transposes for covariance matrix
                subset_points_2 = cluster_points[rng.integers(0, len(cluster_points), size=subset_size),:].transpose()
                subset_points_3 = cluster_points[rng.integers(0, len(cluster_points), size=subset_size),:].transpose()
                #angle clockwise from y+ for the semi-major axis, will need to be converted later to whatever dfnWorks uses for beta
                eigvals_1, eigvecs_1 = np.linalg.eig(np.cov(subset_points_1))
                eigvals_2, eigvecs_2 = np.linalg.eig(np.cov(subset_points_2))
                eigvals_3, eigvecs_3 = np.linalg.eig(np.cov(subset_points_3))
                angle_1 =  math.degrees(math.atan(eigvecs_1[0,np.argwhere(eigvals_1==np.max(eigvals_1))[0,0]]/eigvecs_1[1,np.argwhere(eigvals_1==np.max(eigvals_1))[0,0]])) #For the semi-major axis, which we need to verify here because the three different subsets may have assigned the indices differently
                angle_2 =  math.degrees(math.atan(eigvecs_2[0,np.argwhere(eigvals_2==np.max(eigvals_2))[0,0]]/eigvecs_2[1,np.argwhere(eigvals_2==np.max(eigvals_2))[0,0]]))
                angle_3 =  math.degrees(math.atan(eigvecs_3[0,np.argwhere(eigvals_3==np.max(eigvals_3))[0,0]]/eigvecs_3[1,np.argwhere(eigvals_3==np.max(eigvals_3))[0,0]]))
                
                angle_range = np.max([angle_1,angle_2,angle_3]) - np.min([angle_1,angle_2,angle_3])
                if angle_range < 45:
                    angle = np.mean([angle_1, angle_2, angle_3])
                    aspect_ratio = np.sqrt(np.mean([np.max(eigvals_1),np.max(eigvals_2),np.max(eigvals_3)])/np.mean([np.min(eigvals_1),np.min(eigvals_2),np.min(eigvals_3)])) #Take the means before the ratio
                    found_angle = True
                else:
                    angle_attempts_made +=1
                    print('Warning: Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1])+' failed to produce angle consensus after ' +str(angle_attempts_made)+' attempts, it may be round or the subset size may be too small')
                    #if angle_attempts_made >= 5:
                    #    print('Warning: Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1])+' failed to produce angle consensus after ' +str(angle_attempts_made)+' attempts, it may be round or the subset size may be too small')
            if angle_attempts_made >= attempts_per_cluster:
                angle = 0
                aspect_ratio = 1
                print('Warning: Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1])+' failed to produce angle consensus, defaulting to circle')
                found_angle = True

        #Small numbers of points can lead to anomalous aspect ratios, normalise them
        aspect_ratio = normalise_aspect_ratio(aspect_ratio, np.min([len(cluster_points), subset_size]), aspect_ratio_normalisation_method)
        
        #Next we find the size of the ellipse to match the area
        semi_major_length = np.sqrt((area*(aspect_ratio))/math.pi)

        #Verify that we were correct in identifying the semi-major axis, this should never trigger and indicates a bug if it does
        if aspect_ratio < 1:
            raise ValueError('hulls_to_matching_ellipses generated an aspect ratio of ', aspect_ratio, ' for set:'+str(set_data_array[i,0])+' cluster:'+str(set_data_array[i,1])+' which is less than 1')

        
        #Append results to list
        radii.append(semi_major_length)
        aspect_ratios.append(aspect_ratio)
        betas.append(angle)

        

    #Convert to arrays and return
    radii = np.array(radii)
    aspect_ratios = np.array(aspect_ratios)
    betas = np.array(betas)
    #print('When you finally implement this, check the angle and beta setting code')
    #print(aspect_ratios)
    return radii, aspect_ratios, betas

def hulls_to_bbox_matching_ellipses(hulls_list, set_data_array, subset_size = 5000, attempts_per_cluster = 10):
    #returns an array containing the radii, aspect ratios and betas of each ellipse
    #Ellipses are generated such that they have the same area, aspect ratio (by min bounding box method), dip, dipdir, and orientation within the plane (by covariance method) as the original clusters
    radii = []
    aspect_ratios = []
    betas = []
    for i in range(len(hulls_list)):
        area = hulls_list[i].volume #In 2D it is the volume attribute which actually contains the area of the hull, the area attribute contains the perimeter
        hull_points = hulls_list[i].points[hulls_list[i].vertices].copy()
        hull_points = hull_points.astype('float32') #for the OpenCV minAreaRect
        cluster_points = hulls_list[i].points
        #I'm thinking I take a subset of points from all the points in the cluster rather than the points in the hull
        #Idea being that the points in the cluster should have a more uniform density distribution
        found_angle = False
        angle_attempts_made = 0

        #Find the aspect ratio using the minimum bounding box method
        rect = cv2.minAreaRect(hull_points)
        width, height = rect[1]
        
        #closed_hull_points = np.concatenate((hull_points,[hull_points[0,:]]), axis=0)
        #r, a, width, height, center_point, corner_points = minBoundingRect(closed_hull_points)
        
        #print(width, height)
        aspect_ratio = np.max([width/height, height/width])
        #print(aspect_ratio)
        #print(closed_hull_points)

        

        #Find the orientation as was done for the covariance matching method
        while found_angle == False:
            
            if len(cluster_points)<subset_size:
                subset_points = cluster_points.transpose()
                #angle = math.degrees(math.atan(scipy.stats.linregress(subset_points[:,0], subset_points[:,1]).slope))
                eigvals, eigvecs = np.linalg.eig(np.cov(subset_points))
                semi_major = np.argwhere(eigvals==np.max(eigvals))[0,0] #Indices for the semi-major and semi-minor axes
                semi_minor = np.argwhere(eigvals==np.min(eigvals))[0,0]
                angle = math.degrees(math.atan(eigvecs[0,semi_major]/eigvecs[1,semi_major])) #angle clockwise from y+ for the semi-major axis
                found_angle = True

                
            else:

                #If the cluster is so large that we don't want to use all of its points for performance reasons, we can instead take some random subsets of its points
                #and process those, checking to make sure the three subsets have approximately the same characteristics
                #This is mostly a hedge against very large datasets with a large number of large fractures, or datasets with extremely high point density
                #This approach is vulnerable to issues if a cluster has the vast majority of its points in one place, and a small number of points a significant distance away, as this
                #produces a hull with a very large area but potentially misses the extraneous points for the aspect ratio and orientation calculations. Large subset sizes reduce this chance significantly
                #but one can avoid generating these clusters in the first place by trimming small clusters before merging coplanar in DSE

                
                subset_points_1 = cluster_points[rng.integers(0, len(cluster_points), size=subset_size),:].transpose() #Transposes for covariance matrix
                subset_points_2 = cluster_points[rng.integers(0, len(cluster_points), size=subset_size),:].transpose()
                subset_points_3 = cluster_points[rng.integers(0, len(cluster_points), size=subset_size),:].transpose()
                #angle clockwise from y+ for the semi-major axis, will need to be converted later to whatever dfnWorks uses for beta
                eigvals_1, eigvecs_1 = np.linalg.eig(np.cov(subset_points_1))
                eigvals_2, eigvecs_2 = np.linalg.eig(np.cov(subset_points_2))
                eigvals_3, eigvecs_3 = np.linalg.eig(np.cov(subset_points_3))
                angle_1 =  math.degrees(math.atan(eigvecs_1[0,np.argwhere(eigvals_1==np.max(eigvals_1))[0,0]]/eigvecs_1[1,np.argwhere(eigvals_1==np.max(eigvals_1))[0,0]])) #For the semi-major axis, which we need to verify here because the three different subsets may have assigned the indices differently
                angle_2 =  math.degrees(math.atan(eigvecs_2[0,np.argwhere(eigvals_2==np.max(eigvals_2))[0,0]]/eigvecs_2[1,np.argwhere(eigvals_2==np.max(eigvals_2))[0,0]]))
                angle_3 =  math.degrees(math.atan(eigvecs_3[0,np.argwhere(eigvals_3==np.max(eigvals_3))[0,0]]/eigvecs_3[1,np.argwhere(eigvals_3==np.max(eigvals_3))[0,0]]))

                angle_range = np.max([angle_1,angle_2,angle_3]) - np.min([angle_1,angle_2,angle_3])
                if angle_range < 45:
                    angle = np.mean([angle_1, angle_2, angle_3])
                    found_angle = True
                else:
                    angle_attempts_made +=1
                    print('Warning: Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1])+' failed to produce angle consensus after ' +str(angle_attempts_made)+' attempts, it may be round or the subset size may be too small')
                    #if angle_attempts_made >= 5:
                    #    print('Warning: Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1])+' failed to produce angle consensus after ' +str(angle_attempts_made)+' attempts, it may be round or the subset size may be too small')
            if angle_attempts_made >= attempts_per_cluster:
                angle = 0
                aspect_ratio = 1
                print('Warning: Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1])+' failed to produce angle consensus, defaulting to circle')
                found_angle = True

        
        #Next we find the size of the ellipse to match the area
        semi_major_length = np.sqrt((area*(aspect_ratio))/math.pi)

        #Verify that we were correct in identifying the semi-major axis, this should never trigger and indicates a bug if it does
        if aspect_ratio < 1:
            raise ValueError('hulls_to_bbox_matching_ellipses generated an aspect ratio of ', aspect_ratio, ' for set:'+str(set_data_array[i,0])+' cluster:'+str(set_data_array[i,1])+' which is less than 1')

        
        #Append results to list
        radii.append(semi_major_length)
        aspect_ratios.append(aspect_ratio)
        betas.append(angle)
    
    #Convert to arrays and return
    radii = np.array(radii)
    aspect_ratios = np.array(aspect_ratios)
    betas = np.array(betas)
    #print('When you finally implement this, check the angle and beta setting code')
    #print(aspect_ratios)
    return radii, aspect_ratios, betas
    
    
def hulls_to_spanning_circles(hulls_list, set_data_array, min_diameter = 0):
    #returns an array containing the radii and locations of each circle's centre in the pointcloud reference frame
    #Circles are generated such that they are the smallest circle which contains all of the points
    centres = []
    radii = []
    for i in range(len(hulls_list)):
        hull_points = hulls_list[i].points[hulls_list[i].vertices,:].copy()      #get the points in the convex hull of each cluster, use copy to avoid mutation weirdness
        hull_points = hull_points.astype('float32')
        centre, radius = cv2.minEnclosingCircle(hull_points)
        centre_offset_cluster_frame = np.array(centre)
        spanning_diameter = 2*radius
        if spanning_diameter < min_diameter:
            spanning_diameter = min_diameter
        radii.append(spanning_diameter/2)
        cluster_dip, cluster_dipdir = abcd_to_dipdipdir(set_data_array[i, 3],set_data_array[i, 4], set_data_array[i, 5])
        #print(cluster_dip, cluster_dipdir)
        cluster_centroid = set_data_array[i,7:]
        #print(centre_offset_cluster_frame)
        centre_offset_cluster_frame = np.concatenate((centre_offset_cluster_frame, np.array([0])), axis=0) #Add 0 back for the z axis
        #print(centre_offset_cluster_frame)
        centre_offset_cloud_frame = rotate_points_inverse(centre_offset_cluster_frame, cluster_dipdir, cluster_dip) #rotate the centre offset vector to the pointcloud reference frame
        centres.append(cluster_centroid+centre_offset_cloud_frame)
    centres = np.array(centres)
    radii = np.array(radii)
    return centres, radii


def hulls_to_spanning_circles_forfigure(hulls_list, set_data_array, min_diameter = 0):
    #returns an array containing the radii and locations of each circle's centre in the cluster reference frame
    #Circles are generated such that they are the smallest circle which contains all of the points
    centres = []
    radii = []
    for i in range(len(hulls_list)):
        hull_points = hulls_list[i].points[hulls_list[i].vertices,:].copy()      #get the points in the convex hull of each cluster, use copy to avoid mutation weirdness
        hull_points = hull_points.astype('float32')
        centre, radius = cv2.minEnclosingCircle(hull_points)
        centre_offset_cluster_frame = np.array(centre)
        spanning_diameter = 2*radius
        if spanning_diameter < min_diameter:
            spanning_diameter = min_diameter
        radii.append(spanning_diameter/2)
        centres.append(centre_offset_cluster_frame)
    centres = np.array(centres)
    radii = np.array(radii)
    return centres, radii  


def redetermine_centroids(hulls_list, set_data_array):
    #Takes a list of hulls and a set data array and returns an array of new centroids based on the resampled convex hulls

    #resample the hulls
    resampled_hulls_list = resample_hull_list(hulls_list, points_per_hull = 10000)

    
    centroid_offsets_cloud_frame = []
    for i in range(len(hulls_list)):
        
        #find the centroid in the reference frames of an individual hull
        centroid_hull_frame = np.array([np.mean(resampled_hulls_list[i].points[:,0]),np.mean(resampled_hulls_list[i].points[:,1]),0])

        #Find the centroid offset in the global frame
        cluster_dip, cluster_dipdir = abcd_to_dipdipdir(set_data_array[i, 3],set_data_array[i, 4], set_data_array[i, 5])
        centroid_offset_cloud_frame = rotate_points_inverse(centroid_hull_frame, cluster_dipdir, cluster_dip)
        centroid_offsets_cloud_frame.append(centroid_offset_cloud_frame)
    centroid_offsets_cloud_frame = np.array(centroid_offsets_cloud_frame)
    new_centroids = set_data_array[:,7:] + centroid_offsets_cloud_frame

    return new_centroids

    

def points_to_dfnWorks_user_defined_polygons_file(points_list, job_name): 
    #Takes a list of arrays of points and formats them as a DFNworks user defined polygons file
    file = open(job_name+'_user_defined_polygons.dat', 'w')
    file.write('nPolygons: '+str(len(points_list))+'\n')
    #print('nPolygons: '+str(len(points_list))+'\n')
    for i in range(len(points_list)):
        file.write(str(len(points_list[i]))+' ')
        for j in range(len(points_list[i])):
            file.write('{'+str(points_list[i][j,0])+','+str(points_list[i][j,1])+','+str(points_list[i][j,2])+'} ')
            #print('('+str(points_list[i][j,0])+','+str(points_list[i][j,1])+','+str(points_list[i][j,2])+') ')
        file.write('\n')
        #print('\n')
    
    file.close()
    

def ellipse_data_bundle_to_dfnWorks_user_defined_polygons_file(ellipse_data_bundle, job_name):
    #Takes an ellipse data bundle and the corresponding array of discontinuity set data and writes it to a dfnWorks user defined polygons file
    #ellipse data bundle is a list of the form [number, radius array, aspect ratio array, beta array, centres array, dip-dipdir array, n vertices array]
    points_to_send = []
    for i in range(ellipse_data_bundle[0]):
        semi_major_length = ellipse_data_bundle[1][i]
        semi_minor_length = semi_major_length/ellipse_data_bundle[2][i]
        #print('Aspect ratio:',ellipse_data_bundle[2][i])
        point_angles = np.linspace(0,360, num=ellipse_data_bundle[6][i], endpoint=False)
        ellipse_points = []
        for j in range(ellipse_data_bundle[6][i]):
            point_angle = point_angles[j]
            #print(point_angle)
            if point_angle == 90 or point_angle == 270:
                point_angle = math.radians(point_angle)
                point_distance = semi_minor_length
            elif point_angle == 0 or point_angle == 180:
                point_angle = math.radians(point_angle)
                point_distance = semi_major_length
            else:
                point_angle = math.radians(point_angle)
                #Solution to the simultaneous equation x^2/semi_minor_length^2 + y^2/semi_major_length^2 = 1 and the equation x = ytan(point_angle) expressed in the form of a radius and an angle clockwise from positive y
                point_distance = np.sqrt((((semi_major_length * semi_minor_length)**2)/(semi_minor_length**2 +((math.tan(point_angle)**2)*(semi_major_length**2))))*(1+(math.tan(point_angle)**2))) 
                
            #print(point_angle, point_distance)
            ellipse_points.append([point_distance*math.sin(point_angle),point_distance*math.cos(point_angle)])
        ellipse_points = np.array(ellipse_points)
        ellipse_points = np.concatenate((ellipse_points, np.zeros((ellipse_data_bundle[6][i], 1))), axis=1) #Add 0s back for the z axis
        #print('Generated points',ellipse_points)
        ellipse_points = rotate_points(ellipse_points, -ellipse_data_bundle[3][i], 0)
        cluster_dip = ellipse_data_bundle[5][i,1]
        cluster_dipdir = ellipse_data_bundle[5][i,0]
        
        #print('Cluster dip and dipdir:',cluster_dip, cluster_dipdir)
        #print('Rotated in the plane', ellipse_points)
        ellipse_points_cloud_rotation = rotate_points_inverse(ellipse_points, cluster_dipdir, cluster_dip)
        #print('In the cloud frame:',ellipse_points_cloud_rotation)
        #print(ellipse_data_bundle[4][i,:])
        
        ellipse_points_cloud_frame = ellipse_points_cloud_rotation + ellipse_data_bundle[4][i,:]
        points_to_send.append(np.array(ellipse_points_cloud_frame))
        #print(ellipse_points_cloud_frame)
    points_to_dfnWorks_user_defined_polygons_file(points_to_send, job_name)
    return points_to_send
    
                    
def hulls_to_dfnWorks_user_defined_polygons_file(hulls_list, set_data_array, job_name): 
    #Takes a list of convex hull objects and the corresponding array of discontinuity set data and formats them as a DFNworks user defined polygons file
    points_to_send = []
    for i in range(len(hulls_list)):
        cluster_dip, cluster_dipdir = abcd_to_dipdipdir(set_data_array[i, 3],set_data_array[i, 4], set_data_array[i, 5])
        #print(cluster_dip, cluster_dipdir)
        cluster_centroid = set_data_array[i,7:]
        cluster_hull_points = np.array(hulls_list[i].points[hulls_list[i].vertices]) #Take the x-y coordinates of the points which are actually in the hull (the points attribute of the hull object contains all of the points including those inside the hull)
        cluster_hull_points = np.concatenate((cluster_hull_points, np.zeros((len(hulls_list[i].vertices), 1))), axis=1) #Add 0s back for the z axis
        cluster_hull_points = rotate_points_inverse(cluster_hull_points, cluster_dipdir, cluster_dip)
        
        cluster_hull_points = cluster_hull_points + cluster_centroid
        points_to_send.append(cluster_hull_points)
    points_to_dfnWorks_user_defined_polygons_file(points_to_send, job_name)
    #print(points_to_send)
    return points_to_send

#def hulls_to_dfnWorks_ellipse_file(hulls_list, set_data_array, job_name, ellipse_type = 'spanning_circle', n_vertices = 8, aspect_ratio_normalisation_scheme = 'sqrt', min_spanning_diameter = 0):
#    #Takes a list of convex hull objects and the corresponding array of discontinuity set data and formats them as a dfnWorks user defined ellipse file
#    
#    ellipse_bundle_to_write = hulls_to_ellipse_data_bundle(hulls_list, set_data_array, ellipse_type = 'spanning_circle', n_vertices = 8, aspect_ratio_normalisation_scheme = aspect_ratio_normalisation_scheme, min_spanning_diameter = 0)
#    
#    ellipse_points = ellipse_data_bundle_to_dfnWorks_user_defined_polygons_file (ellipse_bundle_to_write, job_name)
#    return ellipse_bundle_to_write, ellipse_points

def hulls_to_ellipse_data_bundle(hulls_list, set_data_array, ellipse_type = 'spanning_circle', n_vertices = 40, aspect_ratio_normalisation_scheme = 'sqrt', min_spanning_diameter = 0, attempts_per_cluster = 10):
    #Takes a list of convex hull objects and the corresponding array of discontinuity set data and formats them as a data bundle recognised by my other file writing functions

    if ellipse_type == 'spanning_circle':
        #print('Not yet implemented')
        number = len(hulls_list)
        centres, radii = hulls_to_spanning_circles(hulls_list, set_data_array, min_diameter = min_spanning_diameter)
        betas = np.zeros(number)
        aspect_ratios = np.ones(number)
        ns_vertices = n_vertices*np.ones(number, dtype=np.int8)
        dips = []
        dipdirs = []
        for i in range(len(hulls_list)): 
            #finding the dips and dipdirections and using the trend/plunge orientation option    
            #this should avoid a potential confusion with positive and negative normal vectors interacting differently with rotation
            cluster_dip, cluster_dipdir = abcd_to_dipdipdir(set_data_array[i, 3],set_data_array[i, 4], set_data_array[i, 5])
            dips.append(cluster_dip)
            dipdirs.append(cluster_dipdir)
        dips = np.array(dips)
        dipdirs = np.array(dipdirs)
        orientation_data = np.stack([dipdirs,dips]).transpose()

    elif ellipse_type == 'cov_matching_ellipse':
        number = len(hulls_list)
        radii, aspect_ratios, betas = hulls_to_cov_matching_ellipses(hulls_list, set_data_array,  aspect_ratio_normalisation_method =  aspect_ratio_normalisation_scheme)
        ns_vertices = n_vertices*np.ones(number, dtype=np.int8)
        centres = set_data_array[:,7:]
        dips = []
        dipdirs = []
        for i in range(len(hulls_list)): 
            #finding the dips and dipdirections and using the trend/plunge orientation option    
            #this should avoid a potential confusion with positive and negative normal vectors interacting differently with rotation
            cluster_dip, cluster_dipdir = abcd_to_dipdipdir(set_data_array[i, 3],set_data_array[i, 4], set_data_array[i, 5])
            #print('Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1]))
            #print(cluster_dip, cluster_dipdir)
            dips.append(cluster_dip)
            dipdirs.append(cluster_dipdir)
        dips = np.array(dips)
        dipdirs = np.array(dipdirs)
        orientation_data = np.stack([dipdirs,dips]).transpose()
    
    elif ellipse_type == 'bbox_matching_ellipse':
        number = len(hulls_list)
        radii, aspect_ratios, betas = hulls_to_bbox_matching_ellipses(hulls_list, set_data_array, attempts_per_cluster = attempts_per_cluster)
        ns_vertices = n_vertices*np.ones(number, dtype=np.int8)
        centres = set_data_array[:,7:]
        dips = []
        dipdirs = []
        for i in range(len(hulls_list)): 
            #finding the dips and dipdirections and using the trend/plunge orientation option    
            #this should avoid a potential confusion with positive and negative normal vectors interacting differently with rotation
            cluster_dip, cluster_dipdir = abcd_to_dipdipdir(set_data_array[i, 3],set_data_array[i, 4], set_data_array[i, 5])
            #print('Set:'+str(set_data_array[i,0])+' Cluster:'+str(set_data_array[i,1]))
            #print(cluster_dip, cluster_dipdir)
            dips.append(cluster_dip)
            dipdirs.append(cluster_dipdir)
        dips = np.array(dips)
        dipdirs = np.array(dipdirs)
        orientation_data = np.stack([dipdirs,dips]).transpose()
        
    else:
        raise ValueError('hulls_to_dfnWorks_ellipse_file given an ellipse_type which doesn\'t exist')
    #ellipse data bundle is a list of the form [number, radius array, aspect ratio array, beta array, centres array, dip-dipdir array, n vertices array]
    ellipse_bundle = [number, radii, aspect_ratios, betas, centres, orientation_data, ns_vertices]
    return ellipse_bundle


def zigzagify_cluster(ellipse_params, waviness_params, overlap_factor = 1.1):
    #Takes a single set of ellipse params and waviness params and turns them into a zigzag wave, represented by a list of arrays of points in the cloud reference frame
    #waviness_list = [x_peak_freqs, y_peak_freqs, peak_phis, amplitudes, origin in cloud reference frame]
    #ellipse data bundle is a list of the form [number, radius array, aspect ratio array, beta array, centres array, dipdir-dip array, n vertices array]


    #THIS NEEDS SO MUCH TESTING
    #Test scenarios with different numbers of waves
    #Test scenarios with both phase offsets and a centre offset for the ellipse (the spanning circles ellipses)

    polys_to_return = []
    
    if len(waviness_params[0]) == 1:
        #For a single waviness
        #Make quadrilaterals which go all the way to the edges of the ellipse in the direction normal to the wave
    
        #Establish useful quantities
        
        #First we have to do some geometry in the standard reference frame, then transform our results such that the major and minor ellipse axes are aligned to x and y and the centre is at the origin to ease later processing
    
        #These are all in the standard downdip = y+ reference frame 
        #w1 is the crest->crest wavelength vector, w1h is it unit vectors
        #w_1 = np.array((waviness_params[0][0] and 1/waviness_params[0][0], waviness_params[1][0] and -1/waviness_params[1][0],0)) #The ands here avoid divide by zero issues
        w_1  = np.array((waviness_params[0][0],waviness_params[1][0],0))/(waviness_params[0][0]**2 + waviness_params[1][0]**2)
        w_1h = w_1/((w_1[0]**2+w_1[1]**2)**0.5)
        
    
        #v1h is the unit vector normal to w1
        v_1h = np.array((w_1h[1], -w_1h[0],0))
    
        #First we find the position of the origin which was used in the waviness calculations relative to the centroid of the ellipse
        if any(ellipse_params[4] != waviness_params[4]):
            wave_origin_to_ellipse_centre_cloud_frame = ellipse_params[4] - waviness_params[4]
            wave_origin_to_ellipse_centre_cluster_frame = rotate_points(wave_origin_to_ellipse_centre_cloud_frame,ellipse_params[5][0],ellipse_params[5][1])
            #check for some mistake in the rotations
            print('Centre of ellipse', ellipse_params[4])
            print('Radius of ellipse', ellipse_params[1])
            print('Origin used for waviness', waviness_params[4])
            print('Wave origin to ellipse centre in cloud frame', wave_origin_to_ellipse_centre_cloud_frame)
            print('Wave origin to ellipse centre in cluster frame', wave_origin_to_ellipse_centre_cluster_frame)
            if abs(wave_origin_to_ellipse_centre_cluster_frame[2])> ellipse_params[1]/10 or abs(wave_origin_to_ellipse_centre_cluster_frame[2])>ellipse_params[1]/10:
                raise ValueError('Waviness origin appears not to lie in the plane of the cluster')
            wave_origin_to_ellipse_centre_cluster_frame[2] = 0 #Project to cluster plane
            wave_origin_to_ellipse_centre_ellipse_frame = rotate_points(wave_origin_to_ellipse_centre_cluster_frame, ellipse_params[3], 0)
            print('Wave origin to ellipse centre in ellipse frame', wave_origin_to_ellipse_centre_ellipse_frame)
            wave_origin = -wave_origin_to_ellipse_centre_ellipse_frame #Just renaming
        else:
            print('Centre of ellipse', ellipse_params[4])
            print('Origin used for waviness', waviness_params[4])
            wave_origin = np.array((0,0,0))
    
        #Transform everything into the ellipse reference frame from the cluster refrence frame
        vectors_to_rotate = np.stack((w_1, w_1h, v_1h))
        print('w_1, w_1h, v_1 in the cluster reference frame', vectors_to_rotate)
        rotated_vectors = rotate_points(vectors_to_rotate, ellipse_params[3], 0)
        w_1, w_1h, v_1h = rotated_vectors
        print('w_1, w_1h, v_1 in the ellipse reference frame', rotated_vectors)
        #Now we can move the wave origin an arbitrary distance normal to w1 and an integer number of w1 to move it close to ellipse centre
        
        origin_translation_vectors = np.array([[w_1[0], v_1h[0]],[w_1[1],v_1h[1]]])
        coefficients = np.linalg.solve(origin_translation_vectors, wave_origin[0:2])
        print('Origin translation coefficients', coefficients)
        print('Therefore translation', -np.fix(coefficients[0]*w_1), -coefficients[1]*v_1h)
        wave_origin = wave_origin - np.fix(coefficients[0])*w_1 - coefficients[1]*v_1h
        print('Translated wave origin', wave_origin)
    
        #More useful values, given short names for the algebra
    
        x_max = ellipse_params[1]/ellipse_params[2]
        y_max = ellipse_params[1]
        w_x = w_1[0]
        w_y = w_1[1]
    
        #For each wave crest or trough we can find the coordinates of its intersections with the ellipse by solving a quadratic
        #Please see the documentation or final thesis for the algebra
    
        #For the case when w1 is not parallel to x
        if w_y != 0:
            print('Finding x first and then y')
            w_xony = w_x/w_y
            C_tips = np.sqrt((y_max**2 +(w_xony*x_max)**2))
            n_tip_a = 2*((w_y*C_tips - wave_origin[0]*w_x - wave_origin[1]*w_y)/(w_x**2 + w_y**2))
            n_tip_b = 2*((w_y*(-C_tips) - wave_origin[0]*w_x - wave_origin[1]*w_y)/(w_x**2 + w_y**2))
            print('n tips', n_tip_a, n_tip_b)
            if n_tip_a > n_tip_b:
                n_range_max = np.floor(n_tip_a)
                n_range_min = np.ceil(n_tip_b)
            else:
                n_range_max = np.floor(n_tip_b)
                n_range_min = np.ceil(n_tip_a)
            ns = np.arange(n_range_min, n_range_max+0.5, 1)
    
            #find the special tip points
            C = C_tips
            x_tip = (2*w_xony*C*(x_max**2))/(2*(y_max**2+(w_xony*x_max)**2))
            y_tip = C - w_xony*x_tip  
            z_tip_a = (1-2*(n_tip_a-np.floor(n_tip_a)))*((-1)**np.floor(n_tip_a)) * waviness_params[3][0]
            z_tip_b = (1-2*(n_tip_b-np.floor(n_tip_b)))*((-1)**np.floor(n_tip_b)) * waviness_params[3][0]
            tip_points = np.array([[x_tip, y_tip, z_tip_a],[-x_tip,-y_tip, z_tip_b]])
            print('tip points', tip_points)
            
            main_points = []
            for i in ns:
                print('Using value of n', i)
                C_x = ((i*w_x)/2 + wave_origin[0])*w_x
                
                C_y = ((i*w_y)/2 + wave_origin[1])*w_y
                
                C = (C_x+C_y)/w_y
                
                x_1 = (((2*w_xony*C*x_max**2)+np.sqrt((2*w_xony*C*(x_max**2))**2-(4*(y_max**2 + (w_xony*x_max)**2)*((C*x_max)**2-(x_max*y_max)**2))))/(2*(y_max**2+(w_xony*x_max)**2)))
                x_2 = (((2*w_xony*C*x_max**2)-np.sqrt((2*w_xony*C*(x_max**2))**2-(4*(y_max**2 + (w_xony*x_max)**2)*((C*x_max)**2-(x_max*y_max)**2))))/(2*(y_max**2+(w_xony*x_max)**2)))
                y_1 = C - (w_xony*x_1)
                y_2 = C - w_xony*x_2
                print('Cx', C_x)
                print('Cy',C_y)
                print('C', C)
                print('Point 1', x_1, y_1)
                print('Point 2', x_2, y_2)
                main_points.append([[x_1,y_1,((-1)**i) * waviness_params[3][0]],[x_2,y_2,((-1)**i) * waviness_params[3][0]]])
            
        
        elif w_x != 0:
            print('Finding y first and then x')
            w_yonx = w_y/w_x #
            C_tips = np.sqrt((x_max**2 +(w_yonx*y_max)**2))##
            n_tip_a = 2*((w_x*C_tips - wave_origin[0]*w_x - wave_origin[1]*w_y)/(w_x**2 + w_y**2))#
            n_tip_b = 2*((w_x*(-C_tips) - wave_origin[0]*w_x - wave_origin[1]*w_y)/(w_x**2 + w_y**2))#
            if n_tip_a > n_tip_b:#
                n_range_max = np.floor(n_tip_a)#
                n_range_min = np.ceil(n_tip_b)#
            else:#
                n_range_max = np.floor(n_tip_b)#
                n_range_min = np.ceil(n_tip_a)#
            ns = np.arange(n_range_min, n_range_max+0.5, 1)#
    
            #find the special tip points
            C = C_tips
            y_tip = (2*w_yonx*C*(y_max**2))/(2*(x_max**2+(w_yonx*y_max)**2))#
            x_tip = C - w_yonx*y_tip#  
            z_tip_a = (1-2*(n_tip_a-np.floor(n_tip_a)))*((-1)**np.floor(n_tip_a)) * waviness_params[3][0]
            z_tip_b = (1-2*(n_tip_b-np.floor(n_tip_b)))*((-1)**np.floor(n_tip_b)) * waviness_params[3][0]
            
            tip_points = np.array([[x_tip, y_tip, z_tip_a],[-x_tip,-y_tip, z_tip_b]])
            print('Tip points', tip_points)
    
            #find the rest of the points
            main_points = []
            for i in ns:
                print('Using value of n', i)
                C_x = ((i*w_x)/2 + wave_origin[0])*w_x
                
                C_y = ((i*w_y)/2 + wave_origin[1])*w_y
                C = (C_x+C_y)/w_x#
                y_1 = (((2*w_yonx*C*y_max**2)+np.sqrt((2*w_yonx*C*(y_max**2))**2-(4*(x_max**2 + (w_yonx*y_max)**2)*((C*y_max)**2-(y_max*x_max)**2))))/(2*(x_max**2+(w_yonx*y_max)**2)))##
                y_2 = (((2*w_yonx*C*y_max**2)-np.sqrt((2*w_yonx*C*(y_max**2))**2-(4*(x_max**2 + (w_yonx*y_max)**2)*((C*y_max)**2-(y_max*x_max)**2))))/(2*(x_max**2+(w_yonx*y_max)**2)))##
                x_1 = C - w_yonx*y_1#
                x_2 = C - w_yonx*y_2#
                print('Cx', C_x)
                print('Cy', C_y)
                print('C', C)
                print('Point 1', x_1, y_1)
                print('Point 2', x_2, y_2)
                main_points.append([[x_1,y_1,((-1)**i) * waviness_params[3][0]],[x_2,y_2,((-1)**i) * waviness_params[3][0]]])
    
        #Transform back to the cloud reference frame
        print('Ellipse beta value', ellipse_params[3])
        print('Ellipse dipdir', ellipse_params[5][0])
        print('Ellipse dip', ellipse_params[5][1])
        print('Main points in the ellipse reference frame',main_points)
        main_points = rotate_points_inverse(np.array(main_points), ellipse_params[3], 0) #rotate to the cluster frame
        main_points = rotate_points_inverse(np.array(main_points), ellipse_params[5][0], ellipse_params[5][1]) #rotate cluster to the cloud frame
        tip_points = rotate_points_inverse(tip_points, ellipse_params[3], 0) #rotate to the cluster frame
        tip_points = rotate_points_inverse(tip_points, ellipse_params[5][0], ellipse_params[5][1]) #rotate cluster to the cloud frame
        main_points = main_points + ellipse_params[4]
        tip_points = tip_points + ellipse_params[4]
        print('Main points in the cloud reference frame', main_points)
        #Make the end triangles

        if n_tip_a > n_tip_b:
            #Check which end of the final array each tip is on
            triangle = np.array((tip_points[0,:],main_points[-1][0,:],main_points[-1][1,:]))
            centre = np.mean(triangle, axis = 0)
            print('Tip triangle 1', triangle)
            print('Tip centre 1', centre)
            triangle = triangle + ((triangle-centre)*(overlap_factor-1))
            polys_to_return.append(triangle)
            triangle = np.array((tip_points[1,:],main_points[0][0,:],main_points[0][1,:]))
            centre = np.mean(triangle, axis = 0)
            print('Tip triangle 2', triangle)
            print('Tip centre 2', centre)
            triangle = triangle + ((triangle-centre)*(overlap_factor-1))
            polys_to_return.append(triangle)
        else:
            triangle = np.array((tip_points[1,:],main_points[-1][0,:],main_points[-1][1,:]))
            centre = np.mean(triangle, axis = 0)
            print('Tip triangle 1', triangle)
            print('Tip centre 1', centre)
            triangle = triangle + (triangle-centre)*(overlap_factor-1)
            polys_to_return.append(triangle)
            triangle = np.array((tip_points[0,:],main_points[0][0,:],main_points[0][1,:]))
            centre = np.mean(triangle, axis = 0)
            print('Tip triangle 2', triangle)
            print('Tip centre 2', centre)
            triangle = triangle + (triangle-centre)*(overlap_factor-1)
            polys_to_return.append(triangle)
    
        #Make the main series of quads
    
        for i in range(len(main_points)-1):
            quad = np.array((main_points[i][0,:], main_points[i+1][0,:], main_points[i+1][1,:], main_points[i][1,:]))
            print('There should be four points here which will be put in a quad',main_points[i][0,:], main_points[i+1][0,:], main_points[i+1][1,:], main_points[i][1,:])
            centre = np.mean(quad, axis=0)
            quad = quad + ((quad-centre)*(overlap_factor-1))
            print('This should be the resulting quad', quad)
            polys_to_return.append(quad)
            if i == 0 or i==len(main_points)-2:
                print(quad)
    
    
    elif len(waviness_params[0]) == 2:
    
        #For two wavinesses
        #CHECK non-parallel, for parallel do something similar to single waviness?
        
        #Make a grid of points where each point is at the intersection of a peak or trough of each wave, then convert each cell of that grid into a quadrilateral
        #This does not strictly capture the maxima of the sum of multiple waves, but does allow us to use quadrilaterals, halving the number of fractures we need
        #Maybe do something with triangles at the edges?
        
        #Establish useful quantities
        
        #First we have to do some geometry in the standard reference frame, then transform our results such that the major and minor ellipse axes are aligned to x and y and the centre is at the origin to ease later processing
    
        #These are all in the standard downdip = y+ reference frame 
        #w1 and w2 are the crest->crest wavelength vectors, w1h and w2h are their unit vectors
        #w_1 = np.array((waviness_params[0][0] and 1/waviness_params[0][0], waviness_params[1][0] and -1/waviness_params[1][0],0))
        w_1  = np.array((waviness_params[0][0],waviness_params[1][0],0))/(waviness_params[0][0]**2 + waviness_params[1][0]**2)
        w_1h = w_1/((w_1[0]**2+w_1[1]**2)**0.5)
        #w_2 = np.array((waviness_params[0][1] and 1/waviness_params[0][1], waviness_params[1][1] and -1/waviness_params[1][1],0))
        w_2  = np.array((waviness_params[0][1],waviness_params[1][1],0))/(waviness_params[0][1]**2 + waviness_params[1][1]**2)
        w_2h = w_2/((w_2[0]**2+w_2[1]**2)**0.5)
    
        #v1 and v2 are the vectors between points on the grid, their unit vectors are defined as being normal to w1 and w2, respectively
        v_1h = np.array((w_1h[1], -w_1h[0],0))
        v_2h = np.array((w_2h[1], -w_2h[0],0))
        v_1 = v_1h*(((w_2[0]**2+w_2[1]**2)**0.5)/(2*np.dot(w_2h,v_1h))) 
        v_2 = v_2h*(((w_1[0]**2+w_1[1]**2)**0.5)/(2*np.dot(w_1h,v_2h)))
    
        #We can now find the grid origin in the standard reference frame
        O_grid = v_2*(waviness_params[2][0]/(math.pi)) + v_1*(waviness_params[2][1]/(math.pi))
        

        print('In the cluster frame')
        print('w1, w1h, w2, w2h', w_1, w_1h, w_2, w_2h)
        print('v1, v1h, v2, v2h', v_1, v_1h, v_2, v_2h)
        print('Grid origin', O_grid)
        
        #First we find the position of the origin which was used in the waviness calculations relative to the centroid of the ellipse
        if any(ellipse_params[4] != waviness_params[4]):
            print('Centre of ellipse', ellipse_params[4])
            print('Origin used for waviness', waviness_params[4])
            wave_origin_to_ellipse_centre_cloud_frame = ellipse_params[4] - waviness_params[4]
            wave_origin_to_ellipse_centre_cluster_frame = rotate_points(wave_origin_to_ellipse_centre_cloud_frame,ellipse_params[5][0],ellipse_params[5][1])
            #check for some mistake in the rotations
            if abs(wave_origin_to_ellipse_centre_cluster_frame[2])> ellipse_params[1]/10 or abs(wave_origin_to_ellipse_centre_cluster_frame[2])>ellipse_params[1]/10:
                raise ValueError('Waviness origin appears not to lie in the plane of the cluster')
            wave_origin_to_ellipse_centre_cluster_frame[2] = 0 #Project to cluster plane
            wave_origin_to_ellipse_centre_ellipse_frame = rotate_points(wave_origin_to_ellipse_centre_cluster_frame, ellipse_params[3], 0)
            wave_origin = -wave_origin_to_ellipse_centre_ellipse_frame #Just renaming
        else:
            print('Centre of ellipse', ellipse_params[4])
            print('Origin used for waviness', waviness_params[4])
            wave_origin = np.array((0,0,0))

        
    
        #Transform everything into the ellipse reference frame from the cluster refrence frame
        vectors_to_rotate = np.stack((w_1, w_1h, w_2, w_2h, v_1, v_1h, v_2, v_2h, O_grid))
        rotated_vectors = rotate_points(vectors_to_rotate, ellipse_params[3], 0)
        w_1, w_1h, w_2, w_2h, v_1, v_1h, v_2, v_2h, O_grid = rotated_vectors
        O_grid = O_grid+wave_origin

        print('In the ellipse frame')
        print('w1, w1h, w2, w2h', w_1, w_1h, w_2, w_2h)
        print('v1, v1h, v2, v2h', v_1, v_1h, v_2, v_2h)
        print('Grid origin', O_grid)
    
        #Now we can move the grid origin any even number of v1 and v2 to get it near to the ellipse centre
        if any(O_grid != np.array((0,0,0))):
            grid_vectors = np.array([[v_1[0], v_2[0]],[v_1[1],v_2[1]]])
            coefficients = np.linalg.solve(grid_vectors, O_grid[0:2])
            coefficients = 2*np.fix(coefficients/2)
            O_grid = O_grid + coefficients[0]*v_1 + coefficients[1]*v_2

        print('Grid origin after translation', O_grid)
    
        #Old algebra solution
        #if v_1[0] != 0 and v_1[1] != 0: #Check to avoid divide by zeros
        #    #Algebra which works so long as v1 is not parallel to an axis
        #    nv_2 = (O_grid[0]/v_1[0] - O_grid[1]/v_1[1])/(v_2[0]/v_1[0] - v_2[1]/v_1[1])
        #    nv_1 = O_grid[0]/v_1[0] - nv_2*(v_2[0]/v_1[0])
        #    nv_2 = 2*np.fix(nv_2/2)
        #    nv_1 = 2*np.fix(nv_1/2)
        #elif v_2[0] != 0 and v_2[1] != 0:
        #    #Alternative algebra which works so long as v2 is not parallel to an axis
        #    nv_1 = (O_grid[0]/v_2[0] - O_grid[1]/v_2[1])/(v_1[0]/v_2[0] - v_1[1]/v_2[1])
        #    nv_2 = O_grid[0]/v_2[0] - nv_1*(v_1[0]/v_2[0])
        #    nv_1 = 2*np.fix(nv_1/2)
        #    nv_2 = 2*np.fix(nv_2/2)
        #elif v_1[0] != 0 and v_2[1] != 0:
        #    #Alternative algebra which works so long as v1 is parallel to x and v2 is parallel to y
        #    nv_1 = O_grid[0]/v_1[0]
        #    nv_2 = O_grid[1]/v_2[1]
        #    nv_1 = 2*np.fix(nv_1/2)
        #    nv_2 = 2*np.fix(nv_2/2)
        #elif v_1[1] != 0 and v_2[0] != 0:
        #    #Alternative algebra which works so long as v1 is parallel to y and v2 is parallel to x
        #    nv_1 = O_grid[1]/v_1[1]
        #    nv_2 = O_grid[0]/v_2[0]
        #    nv_1 = 2*np.fix(nv_1/2)
        #    nv_2 = 2*np.fix(nv_2/2) 
        #O_grid = O_grid + nv_1*v_1 + nv_2*v_2
    
        #Now everything should be in the same reference frame and we can just go from here
                                      
    
        
        
        
        semi_major_length = ellipse_params[1]
        semi_major_vector = semi_major_length*np.array((0,1))
        semi_minor_length = ellipse_params[1]/ellipse_params[2]
        semi_minor_vector = semi_minor_length*np.array((1,0))
    
        #Find the number of v vectors you need to definitely exit the ellipse
        nv_1_major = np.ceil(abs( v_1[1] and semi_major_length/v_1[1])+2) #Plus 2 because that's the maximum number of v_1s away from the centre of the ellipse we can be at the origin
        nv_1_minor = np.ceil(abs(v_1[0] and semi_minor_length/v_1[0])+2)
        required_v_1 = int(np.max((nv_1_major, nv_1_minor)))
    
        nv_2_major = np.ceil(abs(v_2[1] and semi_major_length/v_2[1])+2)
        nv_2_minor = np.ceil(abs(v_2[0] and semi_minor_length/v_2[0])+2)
        required_v_2 = int(np.max((nv_2_major, nv_2_minor)))
        
        print('Required v1 and v2', required_v_1, required_v_2)
    
        #Initialise grid points array
        grid_points_array = np.zeros((2*required_v_1+1, 2*required_v_2+1,3))
    
        #Assign the actual xyz coordinates (ellipse frame)
        for i in range(grid_points_array.shape[0]):
            for j in range(grid_points_array.shape[1]):
                grid_points_array[i,j,:] = (i-required_v_1)*v_1 + (j-required_v_2)*v_2 +O_grid
                print(i,j,grid_points_array[i,j,:])
                grid_points_array[i,j,2] = ((-1)**i) * waviness_params[3][0] + ((-1)**j) * waviness_params[3][1]
                print(i,j,grid_points_array[i,j,:])
    
        #Find which points are within our actual ellipse
        ellipseness = (grid_points_array[:,:,0]/semi_minor_length)**2 + (grid_points_array[:,:,1]/semi_major_length)**2
        valid_point_mask = ellipseness<=1
    
        #Then transform back to the cloud reference frame
        for i in range(grid_points_array.shape[0]): #Done slice by slice again to avoid reshapes and the ensuing index ordering logic issues
            grid_points_array[i,:,:] = rotate_points_inverse(grid_points_array[i,:,:], ellipse_params[3], 0) #rotate to the cluster frame
            grid_points_array[i,:,:] = rotate_points_inverse(grid_points_array[i,:,:], ellipse_params[5][0], ellipse_params[5][1]) #rotate cluster to the cloud frame
        #Translate to final positions in the cloud frames
        #grid_points_array = grid_points_array + ellipse_params[4]
    
        for i in range(grid_points_array.shape[0]-1):
            for j in range(grid_points_array.shape[1]-1):
                if np.sum(valid_point_mask[i:i+2, j:j+2]) == 4:
                    #Make a quadrilateral
                    quadrilateral = np.array((grid_points_array[i,j,:],grid_points_array[i+1,j,:], grid_points_array[i+1,j+1,:], grid_points_array[i,j+1,:]))
                    #Expand the shape by a factor so that there is overlap at the edges
                    centre = np.mean(quadrilateral, axis = 0)
                    quadrilateral = quadrilateral + (quadrilateral-centre)*(overlap_factor-1)
                    polys_to_return.append(quadrilateral)
                    #The specificity of the indexing is in case some later process needs these to be ordered for it to generate a closed polygon correctly (and not make a bowtie)
                if np.sum(valid_point_mask[i:i+2, j:j+2]) == 3:
                    #Make a triangle
                    triangle = np.array((grid_points_array[i,j,:],grid_points_array[i+1,j,:], grid_points_array[i+1,j+1,:], grid_points_array[i,j+1,:]))
                    #print('This should be four points, which will be cut down to three for a triangle', triangle)
                    triangle_mask = np.array((valid_point_mask[i,j],valid_point_mask[i+1,j], valid_point_mask[i+1,j+1], valid_point_mask[i,j+1]))
                    #print('This should be the mask which should have three trues or three falses', triangle_mask)
                    #print('This should be 1- the mask', 1-triangle_mask)
                    #Indexing ordering here need not be so strict, but reusing this code is easier than using reshapes
                    triangle = np.delete(triangle, ~triangle_mask, axis=0)
                    #print('After the delete, triangle', triangle)
                    #Expand the shape by a factor so that there is overlap at the edges
                    centre = np.mean(triangle, axis = 0)
                    triangle = triangle + (triangle-centre)*(overlap_factor-1)
                    #print('This is what is actually being added', triangle)
                    polys_to_return.append(triangle)

    elif len(waviness_params[0]) > 2:

        #For arbitrary wavinesses, make a fine grid and triangulate

        #Establish useful quantities

        freq_vectors = np.array(waviness_params[0:2]).transpose()
        #w_1  = np.array((waviness_params[0][0],waviness_params[1][0],0))/(waviness_params[0][0]**2 + waviness_params[1][0]**2)
        
        w_stack = np.zeros((freq_vectors.shape[0],3))
        for i in range(freq_vectors.shape[0]):
            w_stack[i,0] = freq_vectors[i,0]
            w_stack[i,1] = freq_vectors[i,1]
            w_stack[i,:]  = w_stack[i,:]/(w_stack[i,0]**2+w_stack[i,1]**2)

        w_stack = rotate_points(w_stack, ellipse_params[3], 0)
        w_stackh =  w_stack.copy()
        for i in range(freq_vectors.shape[0]):
            w_stackh[i,:] = w_stack[i,:] / np.sqrt(w_stack[:,0]**2 + w_stack[:,1]**2)[i]
        
        #First we find the position of the origin which was used in the waviness calculations relative to the centroid of the ellipse
        if any(ellipse_params[4] != waviness_params[4]):
            print('Centre of ellipse', ellipse_params[4])
            print('Origin used for waviness', waviness_params[4])
            wave_origin_to_ellipse_centre_cloud_frame = ellipse_params[4] - waviness_params[4]
            wave_origin_to_ellipse_centre_cluster_frame = rotate_points(wave_origin_to_ellipse_centre_cloud_frame,ellipse_params[5][0],ellipse_params[5][1])
            #check for some mistake in the rotations
            if abs(wave_origin_to_ellipse_centre_cluster_frame[2])> ellipse_params[1]/10 or abs(wave_origin_to_ellipse_centre_cluster_frame[2])>ellipse_params[1]/10:
                raise ValueError('Waviness origin appears not to lie in the plane of the cluster')
            wave_origin_to_ellipse_centre_cluster_frame[2] = 0 #Project to cluster plane
            wave_origin_to_ellipse_centre_ellipse_frame = rotate_points(wave_origin_to_ellipse_centre_cluster_frame, ellipse_params[3], 0)
            wave_origin = -wave_origin_to_ellipse_centre_ellipse_frame #Just renaming
        else:
            print('Centre of ellipse', ellipse_params[4])
            print('Origin used for waviness', waviness_params[4])
            wave_origin = np.array((0,0,0))


        
       
        
            
        #More useful values, given short names for the algebra
    
        x_max = ellipse_params[1]/ellipse_params[2]
        y_max = ellipse_params[1]

        #Choose a grid spacing which is small compared to the smallest wavelength
        print('Wave vector stack',w_stack)
        print('Wave vector amplitude stack',np.sqrt(w_stack[:,0]**2 + w_stack[:,1]**2))
        print('Min wave vector amplitude',np.min(np.sqrt(w_stack[:,0]**2 + w_stack[:,1]**2)))
        grid_resolution = np.min(np.sqrt(w_stack[:,0]**2 + w_stack[:,1]**2))/4
        print('grid resolution', grid_resolution)
        grid_size = np.ceil(y_max/grid_resolution) #No need to account for orientation, we know the entire ellipse will be contained by a square with side length 2*semi-major length
        grid_size = int(grid_size)
        grid_points_array = np.zeros((2*grid_size+1, 2*grid_size+1,3))
        for i in range(grid_points_array.shape[0]):
            grid_points_array[i,:,0] = grid_resolution*(i-grid_size)
            grid_points_array[:,i,1] = grid_resolution*(i-grid_size)
        effective_grid_points = grid_points_array-wave_origin

        for i in range(w_stack.shape[0]):
            grid_points_array[:,:,2] += waviness_params[3][i]*np.cos(((effective_grid_points[:,:,0]*w_stackh[i,0]+ effective_grid_points[:,:,1]*w_stackh[i,1])*2*math.pi/np.sqrt(w_stack[i,0]**2 + w_stack[i,1]**2))+waviness_params[2][i])  #Project each point to the waviness direction, and give it the z value it needs

        #Find which points are within our actual ellipse
        ellipseness = (grid_points_array[:,:,0]/x_max)**2 + (grid_points_array[:,:,1]/y_max)**2
        valid_point_mask = ellipseness<=1
    
        #Then transform back to the cloud reference frame
        for i in range(grid_points_array.shape[0]): #Done slice by slice again to avoid reshapes and the ensuing index ordering logic issues
            grid_points_array[i,:,:] = rotate_points_inverse(grid_points_array[i,:,:], ellipse_params[3], 0) #rotate to the cluster frame
            grid_points_array[i,:,:] = rotate_points_inverse(grid_points_array[i,:,:], ellipse_params[5][0], ellipse_params[5][1]) #rotate cluster to the cloud frame
        #Translate to final positions in the cloud frames
        grid_points_array = grid_points_array + ellipse_params[4]
    
        for i in range(grid_points_array.shape[0]-1):
            for j in range(grid_points_array.shape[1]-1):
                if np.sum(valid_point_mask[i:i+2, j:j+2]) == 4:
                    #Make two triangles
                    triangle = np.array((grid_points_array[i,j,:],grid_points_array[i+1,j,:], grid_points_array[i+1,j+1,:]))
                    #Expand the shape by a factor so that there is overlap at the edges
                    centre = np.mean(triangle, axis = 0)
                    triangle = triangle + (triangle-centre)*(overlap_factor-1)
                    polys_to_return.append(triangle)


                    triangle = np.array((grid_points_array[i,j,:], grid_points_array[i+1,j+1,:], grid_points_array[i,j+1,:]))
                    centre = np.mean(triangle, axis = 0)
                    triangle = triangle + (triangle-centre)*(overlap_factor-1)
                    polys_to_return.append(triangle)
                if np.sum(valid_point_mask[i:i+2, j:j+2]) == 3:
                    #Make a triangle
                    triangle = np.array((grid_points_array[i,j,:],grid_points_array[i+1,j,:], grid_points_array[i+1,j+1,:], grid_points_array[i,j+1,:]))
                    triangle_mask = np.array((valid_point_mask[i,j],valid_point_mask[i+1,j], valid_point_mask[i+1,j+1], valid_point_mask[i,j+1]))
                    #Indexing ordering here need not be so strict, but reusing this code is easier than using reshapes
                    triangle = np.delete(triangle, ~triangle_mask, axis=0)
                    #Expand the shape by a factor so that there is overlap at the edges
                    centre = np.mean(triangle, axis = 0)
                    triangle = triangle + (triangle-centre)*(overlap_factor-1)
                    polys_to_return.append(triangle)

    print(polys_to_return)
    return polys_to_return


def ellipse_bundle_to_wavy_dfnWorks_user_defined_polygons_file(ellipse_data_bundle, waviness_bundle, job_name, waviness_type = 'smooth', resolution_type = 'distance', resolution = 0.1, overlap_factor = 1.1):
    #Takes an ellipse data bundle and a corresponding waviness bundle and turns them into dfnworks files

    #Should check whether I am sending a cluster which doesn't actually have any waviness

    if waviness_type == 'smooth':
        print('Not yet implemented')
        if resolution_type != 'distance' and resolution_type != 'angle':
            raise ValueError('Resolution type not an acceptable value, acceptable values are \'distance\' and \'angle\'')
        
    
    
    
    
    elif waviness_type == 'zigzag':
        print('Not yet implemented')
        points_to_send = []
        for i in range(ellipse_data_bundle[0]):
            ellipse_parameters = [ellipse_data_bundle[0], ellipse_data_bundle[1][i], ellipse_data_bundle[2][i], ellipse_data_bundle[3][i], ellipse_data_bundle[4][i], ellipse_data_bundle[5][i], ellipse_data_bundle[6][i]] #I apologise for this, I will refactor it later
            waviness_parameters = [waviness_bundle[i][0],waviness_bundle[i][1],waviness_bundle[i][2],waviness_bundle[i][3],waviness_bundle[i][4]]
            zigzagged_points = zigzagify_cluster(ellipse_parameters, waviness_parameters, overlap_factor = overlap_factor)
            points_to_send += zigzagged_points

    else:
        raise ValueError('Waviness type not an acceptable value, acceptable values are \'smooth\' and \'zigzag\'')
    #print(points_to_send)
    points_to_dfnWorks_user_defined_polygons_file(points_to_send, job_name)
    return points_to_send