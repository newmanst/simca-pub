# simca-pub

'simca-pub' is a repository containing tools for image segmentation and registration for SiMCA - "Single Molecule Colocalization Assay", which improves immunoassay sensitivity and specificity using single-molecule colocalization. 

Currently in bioRxiv: https://www.biorxiv.org/content/10.1101/2021.12.24.474141v1

# Overview 
''simca-pub'' aims to provide code to: 
1) Create transformation matrices for a particular set of calibration images obtained from different imaging systems or lasers. 
2) Conduct registration of images given transformation matrix as calculated in 1 
3) Localize spots from images and output spot counts and respective intensitives for downstream analysis. 

# Documentation 
Detailed documentation regarding math and methods are described in the bioRxiv publication. 

# System Requirements
## Hardware requirements 
'simca-pub' requires only a standard computer with at least enough RAM for loading and modifying a typical TIFF image (O(2MB) ). 

## Software requirements 
### OS Requirements 
This repository is supported for *macOS*. The repository has been tested on the following system:
+ macOS: Catalina (10.15.7)

### Python dependencies  
'simca-pub' works on Python 3.9.5 with the following dependencies:
'''
numpy (1.21.01)
pandas (0.19.2)
scikit-bio (0.5.6)
scikit-image (0.18.3)
scipy (1.6.3)
swifter (1.0.7)
tifffile (2021.10.12)
'''

# Installation and Demo Guide: 
### Install from Github 
git clone git@github.com:newmanst/simca-pub.git
Typical install times on a standard computer should be less than 5 seconds. 

- To run demo notebooks:
    - cd simca-pub
    - jupyter notebook 
    - Open 'getTF.ipynb'
    - Follow directions as stated in notebook 

- Expected demo runtimes on standard computer:
    - Creating a transformation matrix from should take ~17 seconds (getTF.ipynb)
    - Processing one image including loading of transformation matrix and visualization should take ~10seconds (ImageAnalysis.ipynb)
    - Processing entire demo file without visulization (87 images) should take ~55 seconds. 
    
- To run your own data please modify the demo jupyter notebooks with respective locations of the saved directories and run as directed in the demo. 
    
# License
This project is covered under the MIT License 


