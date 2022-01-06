import os  
from PIL import Image #as pil
import matplotlib.pyplot as plt
import numpy as np
import skimage.io

from scipy.ndimage import gaussian_filter

import skimage.morphology as morph
import skimage.exposure as skie
import skimage.transform as tf
from skimage.filters import threshold_otsu, threshold_local, rank
from skimage import img_as_float
from tifffile import TiffFile 

from matplotlib.patches import Circle
import matplotlib as mpl

import pickle

# Radius around the molecules used for intensity calculation
EMITTER_RADIUS = 3 #2
# Area around the molecules excluded for background calculation
EXCLUSION_RADIUS = 4 #3;
# Area around the molecule used for background calculation
BACKGROUND_RADIUS = 6  #4;

#Scalar for thresholding for real spots by stds * scalar above median
STD_SCALE = 1.4 #0.8 # 2.5#1.4# .8
GAUSSIAN_SIGMA = 0.9# 0.7#.9
print("STD_SCALE ", STD_SCALE, " GAUSSIAN_SIGMA: ", GAUSSIAN_SIGMA)

def load_image_stack(topdir, filename):
    """
    Load ome.tif image stack of either 1,2,3 images 
    Differenciates layers of images as either "left" or "right" image
    Rotates images 90 degrees and stiches left and right side together 
    Returns the loaded left, right and stiched images. 
    """
    fulldatapath = topdir + filename
    with TiffFile(fulldatapath) as tif:
        data = tif.asarray()
        left = []
        right = []
        if len(data) == 3:
            left = tf.rotate(data[1], 90)
            right = tf.rotate(data[2], 90)
        elif len(data) == 2:
            left = tf.rotate(data[0], 90)
            right = tf.rotate(data[1], 90)
        elif len(np.shape(data)) == 2: # Really just a single image!  
            left = tf.rotate(data, 90)
            right = tf.rotate(data, 90)
        else: 
            print("ERROR!!!! UNKNOWN IMAGE SIZE PROVIDED", len(np.shape(data)))
            return [], [], []
        half_size = int(len(left)/2)
        stiched = np.concatenate((left[:,:half_size], right[:, half_size:]), axis=1)
#         stiched = np.concatenate((left, right[:, half_size:]), axis=1)
    return left, right, stiched 

def map_to_full_intensity_range(img, display=True, lowerBound = 0.25, upperBound = 99.5):
    """
    Maps provided image to an optimized and normalized intensity range 
    If display is True, also rescales intensities min.max intensities to lowerBound/upperBound percentiles for image to better visualize spots. 
    Returns: both dynamic range optimized image and intensity rescaled image (if desired)
    
    """
    limg = np.arcsinh(img) #Map to full dynamic range 
    limg = limg / limg.max() #Normalize 
    opt_img = 0
    if display:
        low = np.percentile(limg, lowerBound) #lower bound for spots
        high = np.percentile(limg, upperBound) #upper bound for spots 
        #Image that is scaled between bounds for display
        opt_img  = skie.exposure.rescale_intensity(limg, in_range=(low,high))
    return opt_img, limg


def get_local_intensity(coords, limg, radius=3, innerRadius=0):
    '''
    Calculates intensity of pixels around within innerRadius and (outer)radius
    TODO: Make distance calculation more efficient!!! Vectorize, and only look at 
        areas directly near coord... 
    '''
    rows, cols = np.ogrid[:limg.shape[0], :limg.shape[1]]
    local_intensity = [] 
    local_opt_int = []
    for col, row in coords:
        distances = np.sqrt((rows - row)**2 + (cols - col)**2)
#         distances = np.sqrt(np.sum((imgcoords - coord)**2, axis=1)
        mask = distances <= radius
        if innerRadius > 0:
            mask = mask & (distances >= innerRadius)
        local_intensity.append(np.mean(mask*limg))
     
    return np.asarray(local_intensity)*10**8 

def local_thresh(coords, img, opt_img=None):
    """
    Does local thresholding 
    Returns coordinates and emitter intensity that passed thresholding. 
    """
    local_bg = get_local_intensity(coords, img, radius = BACKGROUND_RADIUS, innerRadius= EXCLUSION_RADIUS)
    local_emit = get_local_intensity(coords, img, radius = EMITTER_RADIUS)
    coordsThresh = coords[np.where(local_emit > local_bg)]
        
    return coordsThresh[:,0], coordsThresh[:,1], local_emit-local_bg

def plot_thresholding_bounds(coords, opt_img):
    '''
    Plots bounds around each coordinate in which intesnity calculations are made 
    '''
    fig,ax = plt.subplots(1, figsize=(10, 10))
    plt.imshow(opt_img)
#     print(local_emit, local_bg )
    for xx,yy in coords:
        circ = Circle((xx,yy), BACKGROUND_RADIUS, lw=.6, color='r', fill=False)
        ax.add_patch(circ)
        circ = Circle((xx,yy), EMITTER_RADIUS, lw=.6, color='w', fill=False)
        ax.add_patch(circ)
        circ = Circle((xx,yy), EXCLUSION_RADIUS, lw=.6, color='c', fill=False)
        ax.add_patch(circ)

    fig.savefig("imgs/" + "local_thresholding" + ".png", dpi=300 )

def find_intensity_coords(img, plot_histogram=False, thresholding="basic", orig_img = None):
    ''' 
    Outputs local maxima above threshold 
    '''
    
    lm = morph.local_maxima(img)
    x1, y1 = np.where(lm.T == True) 
    intensities = img[(y1,x1)]

    #Throw out any spots that are below pure threshold 
    # TODO: throw out by avg intensities within radius 
    def basic_thresh(img, x1, y1, intensities):
        thresh = np.median(img.flatten()) + STD_SCALE*np.std(img.flatten()) #0.8362674
        if plot_histogram:
            plt.hist(img.flatten(), bins=70)
            plt.axvline(x=thresh, color='r')
            plt.show()
        x2, y2 = x1[intensities > thresh], y1[intensities > thresh]
        return x2,y2
     
    x1,y1 = basic_thresh(img, x1, y1, intensities)
    coords = np.column_stack((x1,y1))
    if thresholding == "local": 
#         print("Local thresholding")
        x1,y1, emitter_intensity = local_thresh(coords, img)
    
    else:
        if orig_img is not None:
#             print("using orig img")
            img = orig_img
        emitter_intensity = get_local_intensity(coords, img, radius = EMITTER_RADIUS)
        
    return x1,y1, emitter_intensity

def draw_circles(img, coords, save_pic = False, filename='foo', color='r', coords2=[], coords3=[], coords4=[]):
    ''' 
    Draws circles on img at coordinates provided 
    ''' 
    # Loop through coord, and draw a circle at each x,y pair
    def draw_circles_for_all_coords(coords, col='r', radius=3):
        for xx,yy in coords:
            circ = Circle((xx,yy), radius, lw=.6, color=col, fill=False)
            ax.add_patch(circ)
            
   
    fig,ax = plt.subplots(1, figsize=(10, 10))
    plt.imshow(img)
        
    draw_circles_for_all_coords(coords, color)
    
    if coords2 != []: # Draw another set of coords w. different color
        draw_circles_for_all_coords(coords2, 'b')
        
    if coords3 != []: # Draw another set of coords w. different color
        draw_circles_for_all_coords(coords3, 'c',radius=4)
        
    if coords4 != []: # Draw another set of coords w. different color
        draw_circles_for_all_coords(coords4, 'm', radius=4)

    if save_pic:
        fig.savefig("imgs/" +filename + ".png", dpi=300)

def plt_reconstruction(mask, seed, dilated, gau, hdome, orig_img):
#    mask, seed, dilated, gau, hdome, img
    '''
    Plots 3 subplots
    1) Intensites of a slice of the mask, seed, dilated images
    2) Intensites of a slice of the original image and hdome image 
    3) The image-dilate image (hdome)
    
    '''
#     mask = mask[133:133+60, 90:90+70]
#     seed = seed[133:133+60, 90:90+70]
#     dilated = dilated[133:133+60,90:90+70]
#     gau = gau[133:133+60, 90:90+70]
#     hdome=hdome[133:133+60, 90:90+70]
#     orig_img=orig_img[133:133+60, 90:90+70]
    
    fig, (ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=4, figsize=(14, 4))
    yslice = 28

#     ax0.plot(mask[yslice], '0.5', label='mask')
#     ax0.plot(seed[yslice], 'k', label='seed')
#     ax0.plot(dilated[yslice], '-r', label='dilated')
# #     ax0.set_ylim(-0.2, 2)
#     ax0.set_title('image slice')
#     ax0.set_xticks([])
#     ax0.legend()
        
#     ax1.plot(gau[yslice], '0.5', label='Gaussian filtered', alpha= 0.4)
#     ax1.plot(dilated[yslice], 'b--', label='Dilated background', alpha= 0.4)
#     ax1.plot(hdome[yslice], 'k', label='Gau-Dilated')
#     ax1.plot(orig_img[yslice], 'r-', label='Original', alpha =  0.4)
# #     ax0.set_ylim(-0.2, 2)
#     ax1.set_title('image slice')
#     ax1.set_xticks([])
#     ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.imshow(orig_img) #, cmap='gray')
#     ax2.axhline(yslice, color='r', alpha=0.4)
    ax2.set_title('Original image')
    ax2.axis('off')
    
    ax3.imshow(gau) #, cmap='gray')
    ax3.axhline(yslice, color='r', alpha=0.4)
    ax3.set_title('Gaussian filtered image')
    ax3.axis('off')
    
    ax4.imshow(dilated, vmin=gau.min(), vmax=gau.max()) #, cmap='gray')
    ax4.axhline(yslice, color='r', alpha=0.4)
    ax4.set_title('dilated')
    ax4.axis('off')

    ax5.imshow(hdome) #, cmap='gray')
    ax5.axhline(yslice, color='r', alpha=0.4)
    ax5.set_title('image - dilated')
    ax5.axis('off')
    
    fig.tight_layout()
    plt.savefig("imgs/reconstruction.png", dpi=300)
#     plt.imshow(image)
    plt.show()
    
    plt.plot(orig_img[yslice], 'r-', label='Original', alpha =  0.4)
    plt.plot(gau[yslice], '0.5', label='Gaussian filtered', alpha= 0.5)
    plt.plot(dilated[yslice], 'b--', label='Dilated background', alpha= 0.5)
    plt.plot(hdome[yslice], 'k', label='Gau-Dilated')
    
#     ax0.set_ylim(-0.2, 2)
    plt.title('image slice')
    plt.xticks([])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("imgs/reconstruction_slice.png", dpi=300)
    plt.show()
    
def find_spots(img, visualize=True, draw=False, save_pic=False, filename="img", thresholding="basic"):
    """
    Localizes spots in given image. 
    For faster processing, set visualize,draw to False
    Outputs optimal image to visualize spots. 
    Outputs nx2 coords of spots found 
    TODO: STAY AWAY FROM EDGES
    """
    #Denoising w/ gaussian 
    # https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html
    img = img_as_float(img)
#     img = img[150:150+60, 180:180+70] #Uncomment if want to observe smaller region 
    gau = gaussian_filter(img, GAUSSIAN_SIGMA, mode='constant')

    # Perform morphological reconstruction by dilation 
    seed = np.copy(gau)
    seed[1:-1, 1:-1] = gau.min()
#     seed = gau - 0.003 # to isolate regional maxima of height h (0.003 here)
    mask = gau 
    dilated = morph.reconstruction(seed, mask, method="dilation")
    hdome = gau - dilated
#     plt_reconstruction(mask, seed, dilated, gau, hdome, img)
    
    if thresholding != 'basic':
        print("WARNING!!!! USING LOCAL THRESHOLDING. This is likely too restrictive if there are other local reconstruction pre-processing methods (like morphological reconstruction--which is done by default). Ensure all spots desired are actually localized by this method by setting \"draw\" to True")

    opt_img, limg = map_to_full_intensity_range(hdome, display=visualize)
    x1, y1, spot_int = find_intensity_coords(limg, thresholding=thresholding, orig_img = img)  
    
    if draw:    
        draw_circles(opt_img, zip(x1, y1), save_pic=save_pic, filename=filename)
    
    return opt_img, np.column_stack((x1,y1)), limg, spot_int 
        
def partition_left_right_groups(coords, leftThresh=230, rightThresh=260, visualize=False, opt_img=None, getIDX=False):
    '''
    Partition localized emitters to Left and Right groups 
    Returns indicies that separate all spots to left and right 
        for partition if getIDX is True 
    '''
    
    rightIndex = np.where(coords[:,0]>rightThresh)
    rightCoords = coords[rightIndex]
    leftIndex = np.where(coords[:,0]<leftThresh)
    leftCoords = coords[leftIndex]
    
    if visualize:
        if opt_img is None:
            print("Image to draw on not provided!!") 
        else: 
            draw_circles(opt_img, leftCoords, coords2=rightCoords)
    
    if getIDX:
#         spot_idx = np.column_stack((leftIndex, rightIndex))
        return leftCoords, rightCoords, leftIndex, rightIndex
    
    return leftCoords, rightCoords

def append_ones_col(coords):
    # Adds another column to coords that is all ones. 
    return np.column_stack((coords[:,0], coords[:,1], np.ones(len(coords))))

def do_tf(coords, tf):
    '''
    Does matrix transformation 
    INPUT: Coordinates of shape nx2 (x,y) and 3x3 transformation Matrix
    OUTPUT: New transformed coordinates of shape nx2 
    '''
    coords_nx3 = coords
    if np.shape(coords)[1] !=3: 
#         print("UNEXPECTED SHAPE: ", np.shape(coords))
        coords_nx3 = append_ones_col(coords)
    coordsPrime = coords_nx3.dot(tf)
    return np.column_stack((coordsPrime[:,0], coordsPrime[:,1]))

def match_spots(topdir, filename, LtoR_tf):
    left, right, stiched  = load_image_stack(topdir, filename)
    opt_img, coords, _, _ = find_spots(stiched, visualize=True)
    leftCoords, rightCoords = partition_left_right_groups(coords, leftThresh=230, rightThresh=260, visualize=True, opt_img=opt_img)
    rightPrime = do_tf(leftCoords, LtoR_tf)
    draw_circles(opt_img, leftCoords, coords2=rightPrime)

def load_transforms(tf_file = "LtoRtf.pkl"):
    # Load transforms 
    LtoR_tf = pickle.load(open(tf_file, "rb"))
    RtoL_tf = np.linalg.pinv(LtoR_tf)
    return LtoR_tf, RtoL_tf

def process_image(topdir, filename, LtoR_tf, RtoL_tf):
    #Prep image and partition 
    left, right, stiched  = load_image_stack(topdir, filename)
    opt_img, coords, limg, spot_ints = find_spots(stiched, visualize=True, draw=False, thresholding="basic")
    captureCoords, detectCoords, left_idx, right_idx = partition_left_right_groups(coords, getIDX = True ) 
    return captureCoords, detectCoords, limg, opt_img, spot_ints, left_idx, right_idx


def find_colocalized(coordsPrime, orig_coords, dist_thresh = 1.5):
    """
    Filters only coordinates in coordsPrime that is within a euclidean
    distance of dist_thresh to a point in original_coords
    Outputs original coordinates after filtering, index of coordinates
    """
    # Find closest matching of  potentialColMolecules to Donors 
    # These would be "real" colocalizing molecules 
    # 
    def closest_coord(node, nodes):
        if len(nodes) == 0:
            return np.inf, None
        dist_2 = np.sqrt(np.sum((nodes - node)**2, axis=1))
        return np.min(dist_2), np.argmin(dist_2)

    coords_filt = [] 
    coord_index = []
    prime_idx = []
    
    for i, coord in enumerate(coordsPrime):
        dist, idx = closest_coord(coord, orig_coords)
        if dist < dist_thresh: 
            if idx not in coord_index: 
                coords_filt.append(orig_coords[idx])
                coord_index.append(idx)
                prime_idx.append(i) 
                
    return np.asarray(coords_filt), coord_index, prime_idx
      
    