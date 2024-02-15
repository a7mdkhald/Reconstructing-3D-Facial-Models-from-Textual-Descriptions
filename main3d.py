import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time
import shutil


from api import PRN
from utils.write import write_obj_with_colors

# ---- init PRN
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
prn = PRN(is_dlib = False) 


# ------------- load data
image_folder = 'TestImages/'
save_folder = 'TestImages\\AFLW2000_results'

# ----------------------------------clear Result file before generating 
for filename in os.listdir(save_folder):
    file_path = os.path.join(save_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # Delete files
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Recursively delete directories
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')



types = ('*.jpg', '*.png')
image_path_list= []
for files in types:
    image_path_list.extend(glob(os.path.join(image_folder, files)))
total_num = len(image_path_list)

for i, image_path in enumerate(image_path_list):
    # read image
    image = imread(image_path)

    # the core: regress position map    
    if 'AFLW2000' in image_path:                        # the detected images with .jpg and .mat 
        mat_path = image_path.replace('jpg', 'mat')
        info = sio.loadmat(mat_path)
        kpt = info['pt3d_68']
        pos = prn.process(image, kpt) # kpt information is only used for detecting face and cropping image
    else:
        pos = prn.process(image) # use dlib to detect face     #1 

    # -- Basic Applications
    # get landmarks
    kpt = prn.get_landmarks(pos)                               #2
    # 3D vertices
    vertices = prn.get_vertices(pos)                           #3
    # corresponding colors
    colors = prn.get_colors(image, vertices)                   #4

    # -- save
    name = image_path.strip().split('\\')[-1]
    print(name)
    np.savetxt(os.path.join(save_folder, name + '.txt'), kpt) 
    write_obj_with_colors(os.path.join(save_folder, name + '.obj'), vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

    sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})
