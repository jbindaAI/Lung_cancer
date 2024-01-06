### IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import glob
import pylidc as pl
from pylidc.utils import consensus
import torchio as tio
import matplotlib.pyplot as plt
import torch
from scipy import ndimage
import pickle
import sys, os
import ctypes
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Creating LIDC dataset.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--start_idx", action="store", help="Specifing initial index for first nodule.")
parser.add_argument("-a", "--ann_output_filename", action="store", help="Specifing name for the output annotations file.")
parser.add_argument("-m", "--match_table_name", action="store", help="Specifing name for the output match table.")
args = parser.parse_args()
config = vars(args)

start_idx = config["start_idx"]
anns_name = config["ann_output_filename"]
match_name = config["match_table_name"]


### READING DEFINED PATHS FROM paths.txt FILE:
with open("paths.txt", "r") as p:
    paths = p.read().splitlines()
    base_path = paths[0]
    save_path = paths[1]


### DEFINING PARAMETER h:
h = 32


### HELPER CODE TO CATCH TORCH IO WARNINGS
# this code is only needed to catch the warning from torchio, samples with a warning are excluded from the dataset
def flush(stream):
    try:
        ctypes.libc.fflush(None)
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported
def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd
@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.stdout
    stdout_fd = fileno(stdout)

    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        flush(stdout)
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

            
### EXTRACTION OF NODULE INFORMATION
def process_nodule_vol(nodule, h=32):
    # Firstly, function loads all DICOM files linked to scan which contain the nodule.
    dicom = nodule[0].scan.get_path_to_dicom_files()  
    # Annotation has attribute scan, which says to which scan that annotation belongs.
    # It returns scan object which has the get_path...() method. That make us able to get path to all dicom files.
    
    # As each nodule may be annotated by few radiologists, nodule is a list of annotations.
    # It is why I compute nodule malignancy as a median of annotated values.
    median_malig = np.median([ann.malignancy for ann in nodule])
    
    with open("output.txt", "w") as f, stdout_redirected(f, stdout=sys.stderr):
        # Next image data is loaded from DICOM files to tensor of shape: (1, X, Y, Z). 
        tio_image = tio.ScalarImage(dicom)
        spacing = tio_image.spacing # saving initial image spacing
    
        # Image is then resampled to get spacing (1,1,1)
        transform = tio.Resample(1)
        res_image = transform(tio_image)
    
        # I swap the positions of X and Y axes in the image data.
        res_data = torch.movedim(res_image.data, (0,1,2,3), (0,2,1,3))
    
    with open("output.txt") as f:
        content = f.read()
    if "warning" in content.lower():
        raise RuntimeError("SimpleITK Warning .. skip!")
    open("output.txt", "w").close()
    
    # Below with consensus method I compute common mask and common bounding box.
    # Common mask is a consensus mask indicating precisely which voxel of the volume is the part of nodule. It is an 3D array.
    # Common bounding mask is a consensus bounding box, which surrounds nodule in a 3D space. Its coordinates are given by tupple of python slices.
    # consensus method returns also masks separately for each annotator, but it is not important there.
    cmask, cbbox, _ = consensus(nodule, clevel=0.5)
    
    # As I resampled image data (what means I have changed dimmensions of voxel grid) I have to also change coordinates of cbbox.
    # It is achieved by below code. When original spacing is below 1, then when image get resampled to spacing 1, number of voxels is reduced
    # and we have to move cbbox coordinates to lower idx.
    res_cbbox = [(round(cbbox[i].start*spacing[i]), 
                  round(cbbox[i].stop*spacing[i])) for i in range(3)]
    
    ## Below line is resampling mask like a cbbox. cmask is a 3D matrix and each of its
    # dimmensions are resampled accordingly to provided spacing.
    res_cmask = ndimage.zoom(cmask.astype(int), spacing)
    
    # center of cbbox is computed by averaging extreme indexes on each of dimmensions.
    res_cbbox0 = [round((res_cbbox[i][0]+res_cbbox[i][1])/2) for i in range(3)]
    
    # I create zeros-filled array of an image dimmensions.
    g = np.zeros(res_data.shape[1:])
    
    # Array is then filled with cmask content in the positions given by cbbox coordinates.
    # res_cmask is an array with 0,1 values indicating positions with nodule and not.
    # after that g becomes an array filled with zeros and ones, where ones indicates nodule positions.
    g[res_cbbox[0][0]:res_cbbox[0][0]+res_cmask.shape[0], 
      res_cbbox[1][0]:res_cbbox[1][0]+res_cmask.shape[1],
      res_cbbox[2][0]:res_cbbox[2][0]+res_cmask.shape[2],] = res_cmask
    
    # Nodule surrounding volume of dimmensions (h,h,h)==(2k,2k,2k) is extracted.
    k = int(h/2)
    # I define slices tupple. It is the used to extract from res_data nodule surrounding volume.
    # Center of a nodule is given by res_cbbox0 object.
    slices = (
                slice(res_cbbox0[0]-k, res_cbbox0[0]+k),
                slice(res_cbbox0[1]-k, res_cbbox0[1]+k),
                slice(res_cbbox0[2]-k, res_cbbox0[2]+k)
             )
    # Cropping a nodule volume from res_data by defined slices tupple.
    crop = res_data[0][slices]
    
    # Finally, I extract mask volume of dimmensions (h,h,h) from the g tensor.
    g = torch.tensor(g)
    mask = g[slices]
    
    assert crop.shape == torch.Size([h,h,h])
    assert mask.shape == torch.Size([h,h,h])
    
    return median_malig, crop, mask, cbbox


### MAIN PART
# Firstly, proper directories are created.
# Path object is a useful path object from pathlib library.
Path(f"{save_path}/crops").mkdir(parents=True, exist_ok=True)
Path(f"{save_path}/masks").mkdir(parents=True, exist_ok=True)

# Secondly, apth to all patients folders are saved in a list by the glob function.
d = glob.glob(f"{base_path}/*")
d.sort() # alphabetical sorting of paths.
# in that case paths will be sorted in according to the patient id:
# LIDC-IDRI-0001, LIDC-IDRI-0002 etc.

# Then from paths I extract only LIDCI-IDRI ID values:
pids = [elt.split('/')[-1].split('-')[-1] for elt in d]

# Defining nodule attributes to extract. Without diameter, because it has to be extracted in a different way.
attributes = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture"
]

# Defining containers for extracted dataset.
match = []
avg_annotations = {}
for att in attributes:
    avg_annotations[att] = []
avg_annotations["diameter"] = []
avg_annotations["target"] = []
avg_annotations["path"] = []

# New ID for nodules is initialized:
new_id = int(start_idx)

for patient_id in tqdm(pids):
    # Selecting from database scan for the patient with given id. Query object needs transformation to the scan object
    # what is done by the .first() method.
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == f"LIDC-IDRI-{patient_id}").first()
    
    # Clustering annotations. It means assigning annotations to the concrete nodule.
    # Each nodule may has few annotations as there were 4 annotators!
    nodules = scan.cluster_annotations()
    
    if len(nodules) == 0: # in case, when patient (hopefully) hasn't any nodule.
        continue
    
    k = 0
    for nodule in nodules:
        num_annotations=len(nodule)
        # I consider only nodules which have more than 2 annotations and less than 4.
        if num_annotations > 4:
            print("skipping!")
        if (num_annotations > 2 and num_annotations <= 4):
            try:
                median_malig, crop, mask, cbbox = process_nodule_vol(nodule, h=h)
                str_new_id = str(new_id).zfill(4)
                append = False
                if(median_malig > 3):
                    avg_annotations["target"].append(1)
                    append = True
                    new_id += 1
                elif(median_malig < 3):
                    avg_annotations["target"].append(0)
                    append = True
                    new_id += 1
                if(append):
                    avg_annotations["diameter"].append(np.mean([ann.diameter for ann in nodule]))
                    avg_annotations["path"].append(f"{str_new_id}.pt")
                    for att in attributes:
                        # vars() function returns dictionary of object attributes.
                        avg_annotations[att].append(np.mean([vars(ann)[att] for ann in nodule]))
                        
                    match.append([patient_id, k, new_id])
                    torch.save(crop.clone(), f"{save_path}/crops/{str_new_id}.pt")
                    torch.save(mask.clone(), f"{save_path}/masks/{str_new_id}.pt")
            # if creation of crop fails for any reason, skip to next nodule
            except Exception as e:
                print(e)
                continue
        k += 1

# Saving files:
with open(f"{save_path}/{match_name}.pkl", "wb") as handle:
    pickle.dump(match, handle)
    
with open(f"{save_path}/{anns_name}.pkl", "wb") as handle:
    pickle.dump(avg_annotations, handle)

# Deleting temporal file for catching torchIO warnings.
os.remove("output.txt")