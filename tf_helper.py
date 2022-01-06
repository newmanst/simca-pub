from SA_helper import *
import os

def get_tf_mat(topdir, filename):
    ## GET TRANSFORMATION MATRIX OF BEAD FILE 
    left, right, stiched  = load_image_stack(topdir, filename)
    opt_img, coords, _, _ = find_spots(stiched, visualize=True, draw=False, save_pic=False, filename="img")
    leftCoords, rightCoords = partition_left_right_groups(coords, leftThresh=230, rightThresh=260)
    leftCoordsMatched, rightCoordsMatched = find_onetoone_coords(leftCoords, rightCoords, rightshift = 257)
    
    if (len(leftCoordsMatched) ==0)  or (len(rightCoordsMatched) == 0):
        return []
    
    LtoR_tf, RtoL_tf = recover_transformation(leftCoordsMatched, rightCoordsMatched)
    return LtoR_tf


def get_avg_tfs(beaddir):
    tfs = []
    for subdir, dirs, files in os.walk(beaddir):
        for fileName in files:
            if fileName.split('.')[-1] == "tif":
#                 print(subdir, fileName)
                cur_tf = get_tf_mat(subdir, fileName)
                if len(cur_tf) != 0:
                    tfs.append(cur_tf)
    avg_tf = np.asarray(tfs).mean(axis=0)
    return avg_tf

def check_all_file_w_tf(beaddir, avg_tf):
    # Go through all bead files again, and visually check that averaged TF works ok 
    tfs = []
    for subdir, dirs, files in os.walk(beaddir):
        for fileName in files:
            if fileName.split('.')[-1] == "tif":
                match_spots(subdir, fileName, avg_tf)
                    
def recover_transformation(coords, coordsPrime):
    '''
    Input: Coordinates and modified coordinates of shape nx3 
    Returns: transformation matrix such that coords*tf = coordsPrime
    tf_inv is coordsPrim*tf_inv = coords
    '''
    coords_3xn = append_ones_col(coords)
    coordsPrime_3xn = append_ones_col(coordsPrime)
    
    coordsInv = np.linalg.pinv(coords_3xn)
    tf = coordsInv.dot(coordsPrime_3xn)
    tf_inv = np.linalg.pinv(tf)
    return tf, tf_inv

def find_onetoone_coords(leftCoords, rightCoords, rightshift = 257):    
    '''
    Return coordinates in left and right that when shifted, are colocalized. 
    Discards remaining un-colocalized spots 
    NOTE: Used for making of transform. Typically coordinates are from bead files. 
    '''
    from sklearn.neighbors import NearestNeighbors, KDTree
    
    # push left and right coordinates together into same list
    x2 = np.concatenate((leftCoords[:,0], rightCoords[:,0]-rightshift))
    y2 = np.concatenate((leftCoords[:,1], rightCoords[:,1]))
    coords = np.column_stack((x2,y2)) #Stack [[x1,y1], [x2,y2], ... ] 

    #Find closest neighbors in all coordinates 
    kdt = KDTree(coords, leaf_size=30, metric='euclidean')
    #Get coordinates of the matched sequences
    matched = kdt.query(coords, k=2, return_distance=False) 

    #Find coords that match left to right AND right to left
    # Indicies on coords that match!! 
    all_left = matched[np.where(matched[:, 0] < len(leftCoords))]
    all_right = matched[np.where(matched[:, 0] > len(leftCoords))]

    all_right_switch = np.column_stack((all_right[:,1], all_right[:,0]))


    exact_matches = [None,None] #just to initiate
    exact_matches = np.reshape(exact_matches, (1,2))

    for leftind in all_left:
        leftind = np.reshape(leftind, (1,2))
        test = all_right_switch[np.where(all_right_switch == leftind[0])]
        if np.size(test) == 2:
            exact_matches = np.concatenate((exact_matches, leftind)) #, axis=0)
    exact_matches = exact_matches[1::]

    #Use indicies given in exact_matches to remap to coords found by KD tree
    leftCoordsMatched = np.array([coords[i] for i in exact_matches[:,0]] )
    rightCoordsMatched = np.array([coords[i] for i in exact_matches[:,1]] )
    
    if (len(leftCoordsMatched) ==0)  or (len(rightCoordsMatched) == 0):
        return [], []
    
    rightCoordsMatched[:,0] = rightCoordsMatched[:,0]+rightshift 
#     print(rightCoordsMatched)

    return leftCoordsMatched, rightCoordsMatched
    

    