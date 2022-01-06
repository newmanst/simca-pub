from SA_helper import *
import os
from os import path
import pandas as pd 

def calculate_fret(colocalized, LtoR_tf, left_img, rightCoords):
    """
    Calculates fret index using the intensities of donor and acceptor
    Returns real fret coords on acceptor side, a subset of fret_right 
    """
    leakage = 0.15
    
     # Do not proceed if no there are no colocalized spots 
    if len(colocalized) == 0:
        return colocalized
    
    #Calculate coords for left coordinates of FRET 
    fret_right = do_tf(colocalized, LtoR_tf)

    #Use left image, obtain intensity values 
    fret_d = get_local_intensity(colocalized, left_img, radius = EMITTER_RADIUS)
    fret_a = get_local_intensity(fret_right, left_img, radius = EMITTER_RADIUS)

    #Remove crosstalk from acceptor. 
    Ia_nocrosstalk = fret_a - leakage*fret_d
    fret_index = Ia_nocrosstalk / (Ia_nocrosstalk + fret_d)
    return fret_right[np.where(fret_index > 0)]
    
def process_one_image(topdir, filename, LtoR_tf, RtoL_tf, draw = False, dist_thresh=1.5, verbose=False, getEachInt=False):
    """
    Goes through entire spot counting and intensity calculations for one image. 
    1) Loads images and partition to left and right 
    2) Check for good spot distribution 
    3) Filter to colocalized spots: Starting with the right coordinates, find where there are spots in the transformed area also detected in leftCoord. 
    4) Find FRETing molecules based on colocalized spots
    5) Calculate left and right spot intensities 
    6) RETURN number of spots on left, number of spots on right, number colocalized, number fret, mean intensity of left spots, mean intensity of right spots 
    
    INPUT:
    topdir: top directory name 
    filename: name of particular ome.tif file 
    LtoR_tf/RtoL_tf: 3x3 transforms to convert left spots to right and vice versa
    draw: True if drawing of the detected spots is desired 
    dist_thresh: Distance in which transformed spots must match to count as colocalized
    verbose: print results
    getEachInt: True also returns dictionary of intensities:
                left, right, left-colocalized, right-colocalized
    """
    
    #Load and partition image spots
    left, right, stiched  = load_image_stack(topdir, filename)
    if len(stiched) == 0: # ERROR occured while loading image. ignore this one. 
        return ['NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']
    
    opt_img, coords, limg, spot_ints = find_spots(stiched, visualize=True, draw=draw, thresholding="basic")
    leftCoords, rightCoords, left_idx, right_idx = partition_left_right_groups(coords, getIDX = True ) 

    #Check distribution of spots. if not evenly spread out on y-axis, remove!! 
    meanLeftCoords = np.mean(leftCoords, axis=0)
    meanRightCoords = np.mean(rightCoords, axis=0)
    if (meanLeftCoords[1] > np.shape(stiched)[1]/2 + 75) or (meanLeftCoords[1] < np.shape(stiched)[1]/2 - 75):
        print("WARNING! Potentially bad image. Setting values to Nan", str(meanLeftCoords[1]))
        return ['NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN']
    

    #Calculate the potential colocalized molecules on left side 
    leftPrime = do_tf(rightCoords, RtoL_tf) 

    #Match the calculated transformed values to real detected coordinates
    #Colocalized are the left coordinates of real colocalizing spots
    colocalized, col_idx, prime_idx = find_colocalized(leftPrime, leftCoords, dist_thresh = dist_thresh)
    
    fret = calculate_fret(colocalized, LtoR_tf, left, rightCoords)
        
    #Intensities of spots split by left/right 
    left_ints = spot_ints[left_idx]
    right_ints = spot_ints[right_idx]
    
    meanLeft = ['NaN' if len(left_ints)== 0 else np.mean(left_ints)][0]
    meanRight = ['NaN' if len(right_ints)== 0 else np.mean(right_ints)][0]
    
    results = [len(leftCoords), len(rightCoords), len(colocalized), len(fret), meanLeft, meanRight]
#     results = {"Donor": len(leftCoords), "Acceptor": len(rightCoords),
#                 "Colocalized":len(colocalized), "FRET": len(fret), 
#                 "DonorInt": meanLeft, "AcceptorInt": meanRight}
    
    if verbose: print(results)
    
    if draw:
        draw_circles(opt_img, leftCoords, coords2= rightCoords, coords3=colocalized, coords4=fret, save_pic=True, filename="green_red_colocal")
        print("Image saved to: ", "green_red_colocal") 
        
    #TODO add results to dictionary above?? 
    #Extract individual spot intensities
#     intensities = [] 
    if getEachInt:
        Colocalized_green_ints = left_ints[col_idx]
        Colocalized_red_ints = right_ints[prime_idx] 
#         print(type(left_ints), np.shape(left_ints))
#         results.update({"left": list(left_ints), "right": list(right_ints), 
#                         "Col_left": list(Colocalized_green_ints), 
#                         "Col_right": list(Colocalized_red_ints)})
        results.extend([left_ints, right_ints, [Colocalized_green_ints], [Colocalized_red_ints] ])
    return results

def save_counts(df, curFile, topdir, outfile="datasummaryV2"):
    """
    Saves spot counts into csv. 
    Input: DF with columns with labels:
    Donor, Acceptor, Colocalized, FRET, mean DonorInt of FOV, mean Acc int of FOV
    Saves to outfile provided under the top directory 
    """
#     labels = "Donor, Acceptor, Colocalized, FRET, DonorInt, AcceptorInt \n"
#     print(df) #.columns.values)
#     labels = ",".join(df.columns.values) + "\n"
    outfile = topdir + "/" + outfile + ".csv"
    
    writeHeader = False
    if not path.exists(outfile):
        print("Writing Header!! ")
        writeHeader = True
            
    with open(outfile, 'a') as file:
        if writeHeader:
            df.to_csv(file, header = True, index=False)
#             file.write(labels)    
#         file.write(curFile)
#         file.write("\n")
        else:
            df.to_csv(file, header = False, index=False)
    
def process_directory(topdir, bead_dir = "", verbose=False, outfile="datasummaryV2", getEachInt=False):
    """
    Iterates through entire directory holding folders containing ome.tif images
    """
    print("Processing directory: ", topdir)
    LtoR_tf, RtoL_tf = load_transforms(tf_file = topdir + bead_dir + "LtoRtf.pkl")
    
    #UPDATE THESE LABELS IF DIFFERENT
    labels = ["cAb", "dAb","Colocalized", "FRET", "cAbInt", "dAbInt"] 
    if getEachInt:
        labels.extend(["Left_ints", "Right_ints", "ColLeftInts", "ColRightInts"])
        
    dirs = sorted(os.listdir(topdir) )
    for subdir in dirs:
        subdir_path = os.path.join(topdir, subdir)
        if os.path.isdir(subdir_path): #if is a subdirectory
            allFOVs = [] # all Frames of View
            curFile = subdir.split('/')[-1]
            print("Now viewing: ", curFile)    
            for fileName in sorted(os.listdir(subdir_path)):
                if fileName.split('.')[-1] == "tif":
                    print(fileName)
                    img_path = os.path.join(subdir_path,fileName)
                    FOV = process_one_image("", img_path, LtoR_tf, RtoL_tf, 
                                            verbose=verbose, getEachInt=getEachInt)
                    allFOVs.append(FOV)
                    
            if len(allFOVs) > 0:
#                 print(allFOVs)
                df = pd.DataFrame(allFOVs, columns = labels) #.astype('object')
#                 df = df.append(allFOVs, ignore_index=True) 
                df["folder"] = curFile
                save_counts(df, curFile, topdir, outfile)
            
def main(topdir):
    parser = argparse.ArgumentParser()
    parser.add_argument('topdir', metavar="T", type=str, help='Top Directory for images to be analyzed')
    args = parser.parse_args()
    
    process_directory(args.topdir)
    
if __name__ == "__main__": 
    
    main(topdir)
    