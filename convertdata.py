import os
import tifffile as tiff
from PIL import Image
import numpy as np
import sys

def convert_data(num_images, full_t_range, train):
    assert num_images > 0, 'Number of images must be positive'

    # Directory of the TIF images
    vol_count = 1
    image_count = 0
    path2data = r'./data/'
    if train:
        path2dir = r'./convertedData/train/'
    else:
        path2dir = r'./convertedData/test/'
    
    for i in range(num_images):
        if not train:
            # Test data is taken from the end of the images
            i = 115-i
            
        # Process the cell image
        name_img = f'cell{i+1}.tif'
        path2img = os.path.join(path2data, name_img)
        try:
            im = tiff.imread(path2img)
        except:
            continue
        # Same process for the mask image, save in ground truth folder
        name_img_mask = f'cell{i+1}_mask.tif'
        path2img_mask = os.path.join(path2data, name_img_mask)
        im_mask = tiff.imread(path2img_mask)

        # Loop over time points and z-levels and export PNG instances
        # If full_t_range is False, only export 4 time points
        #for t in range(im.shape[0]*full_t_range+4-4*full_t_range):
        for t in range(im.shape[0]*full_t_range+1-full_t_range):
            if not full_t_range:
                # Export only 4 time points
                #t = int([0.3, 0.5, 0.7, 0.9][t]*im.shape[0])
                # Export only 1 time point
                t = int(0.75*im.shape[0])

            #for z in range(im.shape[1]):
            for z in [i for i in range(int(im.shape[1]))]: # Export only the middle z-level

                # Extract the 2D image for this time point and z-level
                page_mask = im_mask[t, z, :, :] 

                # Only take an image if the mask actually has data
                if np.max(page_mask) == 0:
                    continue

                # Save the layer as PNG
                page_mask = Image.fromarray(page_mask)
                page_mask.save(os.path.join(path2dir+'gt/', f'vol{vol_count:03}_z{z+1:03}.png'))

                # Extract the 2D image for this time point and z-level
                page = im[t, z, :, :]  
                # Normalize the image to 0-255
                page = (page - np.min(page)) * (255.0 / (np.max(page) - np.min(page)))
                page = page.astype(np.uint8)

                # Save the layer as PNG
                page = Image.fromarray(page)
                page.save(os.path.join(path2dir+'syn/', f'vol{vol_count:03}_z{z+1:03}.png'))
                image_count += 1
                
            vol_count += 1
        print(f'Saved data for cell {i+1}') 
    print(f'Taken {vol_count} volumes and {image_count} images')

if __name__ == '__main__':
    # Convert data with input from command line otherwise with default values   
    if len(sys.argv) == 4:
        num_images = int(sys.argv[1])
        full_t_range = int(sys.argv[2])
        train = int(sys.argv[3])
        convert_data(num_images, full_t_range, train)
    else:
        num_images = 100
        full_t_range = False
        train = True
        convert_data(num_images, full_t_range, train)
        # num_images = 50
        # full_t_range = False
        # train = False
        # convert_data(num_images, full_t_range, train)
