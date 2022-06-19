'''
===============================================================================================
buildup.py
Dependencies: numpy, cv2, h5py
Requirements for main program: the 'SynthText.h5' dataset, being in the current working
    directory.

Nadav Kahlon, December 2021.
===============================================================================================

This program is responsible for processing the dataset found at 'SynthText.h5', and
building up a "concentrated" dataset, organized in a neat hierarchy for later steps of
the process.

The idea is to take the bounding box of each character in the dataset, and map it to
a constant (rectangular) box using a calculated homography.
We also leave out margins, so it is easier to understand the context of the texture
around the character. The resulting "concentrated" images are 64-by-64 pixels.

The buildup process includes the following:
* Randomly split the images to 20% test, 20% validation, and 60% training.
* Concentrate the images on the different characters presented in them.
* Organize the data into an HDF5 file 'NeatSynthText.h', as described below.

To make life easier later on, we attach an index to each font name (so we can talk in
in terms of indices, and not strings representing font names). I call font indices "fontis".
Font names are listed in a global list 'fonti_map', indexed by their corresponding
fonti. A global variable 'total_fonts' also exists, and holds the number of fonts in the
map.

The resulting "neat" data file has the following form:
* At the top of the hierarchy there are 3 main groups dividing the data: 'train', 'test',
  'val'.
* Each of those has 2 subgroups: 'data' and 'indexing'.

* Subgroup 'data' includes the data itself. It has 3 h5 datasets in it:
    > 'images' (of shape (N,64,64,3)) - containing the concentrated images.
    > 'fontis' (of shape (N)) - containing the labels of the images (represented by fontis).
    > 'chars' (of shape (N)) - containing the ASCII values of the characters in the images.

* Subgroutp 'indexing' is used for convenient access into the datasets. The data in it
  are indices into the datasets in 'data'. It uses 3 methods to organize the images: by 
  character (all images of the same character), by word (all images of the same word), and
  by image (all images derived from the same original image):
    > Sub-subgroup 'by_char' holds datasets named after the ASCII value of each character,
      containing lists of indices of examples of the corresponding character.
    > Sub-subgroup 'by_word' holds datasets of lists of indices of examples of the same
      word (the datasets are named arbitrarily by a serial number).
    > Sub-subgroup 'by_img' holds datasets named by the names of the (original) images.
      Each, contains a single value - the index of the first (concentrated) image derived
      from the corresponding original image (the rest of the concentrated images derived
      from it are the ones that follow)

'''

import numpy as np
import cv2
import h5py
from math import ceil
import random

# list of fonts, indexed by their fonti (font index).
fonti_map = [b'Open Sans', b'Raleway', b'Alex Brush', b'Roboto',
            b'Russo One', b'Ubuntu Mono', b'Michroma']
total_fonts = 7 # number of fonts


'''
concentrateImg : "concentrates" an image on a bounding box.
    Input:
        > 'src_img' - source image.
        > 'src_BB' - 2-by-4 matrix whose columns are the vertices of a bounding box.
        > 'target_BB_res' - resolution to map the boundry box to.
        > 'hmargin' - number of pixels to leave as horizontal margins (to the left the 
            and the right of the mapped boundry box)
        > 'vmargin' - number of pixels to leave as vertical margins (above and below the 
            mapped boundry box)
    Returns:
        A numpy array holding an image of resolution (2*hmargin + target_BB_res[0],
        2*vmargin + target_BB_res[1]), being the source image that was mapped using homography
        between the boundry box and the target resolution box, and added margins.
'''
def concentrateImg(src_img, src_BB, target_BB_res=(44,44), hmargin=10, vmargin=10):
    # target boundry box (for the homography)
    target_BB = np.array([[hmargin, vmargin],
                          [hmargin + target_BB_res[0], vmargin],
                          [hmargin + target_BB_res[0], vmargin + target_BB_res[1]],
                          [hmargin, vmargin + target_BB_res[1]]]).transpose()
    # resolution of the target concentrated image (including margins)
    target_img_res = (2*hmargin + target_BB_res[0],
                      2*vmargin + target_BB_res[1])
    # calculate the homography from original BB to target BB
    H = cv2.findHomography(src_BB.transpose(), target_BB.transpose())[0]
    return cv2.warpPerspective(src_img, H, target_img_res)


'''
splitList : splits a list into 3 partitions by the 3 'partition' values (assumes they
    add up to 1).
'''
def splitList(ls, partition1, partition2, partition3):
    # calculate the ending point of the first 2 parts
    end1 = ceil(partition1 * len(ls))
    end2 = ceil((partition1+partition2) * len(ls))
    # split
    part1 = ls[ : end1]
    part2 = ls[end1 : end2]
    part3 = ls[end2 : ]
    return part1, part2, part3


'''
processDataset : preprocesses information in a source dataset of synthetic text images.
    Input:
        > 'db' - HDF5 (opened) file holding the data.
        > 'img_names' - list of names of images in the dataset that you wish to process.
        > 'fonts_included' - a boolean, representing wether or not font label information is
            included in the dataset.
        > 'verbose' - a boolean, representing wether or not you wish to see the process documented
            in real time in the terminal.
    Returns (a tuple of):
        > 'result_imgs' - a list of concentrated (numpy) images of individual characters.
        > 'result_chars' - a list of the characters appearing in each concentrated image.
        > 'result_fontis' - a list of the fontis of the characters in each concentrated image
            (this may be None if 'fonts_included' is False).
        > 'by_char_indexing' - a dictionary indexed by strings representing ASCII values of
            the characters appearing the processed images. Each entry in it is a numpy list
            of indices of concentrated images of the corresponding character.
        > 'by_word_indexing' - a list of numpy lists of indices of concentrated images of the same word.
        > 'by_img_indexing' - a dictionary indexed by image names. Each entry in it is the 
            index of the first concentrated image derived from the corresponding (original) image in
            numpy 0-dimensional array (the rest of the concentrated images derived from this image
            are the concentrated images that follow).
''' 
def processDataset(db, img_names, fonts_included=True, verbose=True):
    # initialize outputs
    result_imgs = []
    result_chars = []
    result_fontis = [] if fonts_included else None
    by_char_indexing = {}
    by_word_indexing = []
    by_img_indexing = {}
    
    global_curr_i = 0 # index of current result image
    
    # process each image:
    for img_i, img_name in enumerate(img_names):
        if verbose: print('\t> Preprocessing image', img_i+1, '/', len(img_names), '...', end=' ')
        local_curr_i = 0 # index of current character processed in the current image
        
        # extract the image and data about the characters
        img = db['data'][img_name][:] / 255.0
        words = db['data'][img_name].attrs['txt']
        char_BBs = db['data'][img_name].attrs['charBB']
        if fonts_included: fonts = db['data'][img_name].attrs['font']
        
        # process each word:
        for word in words:
            curr_word_indices = np.empty(0, dtype=int) # indices of characters of this word
            
            # process each charcater:            
            for char in word:
                # get data about the charcter
                char_BB = char_BBs[:, :, local_curr_i]
                if fonts_included: 
                    font = fonts[local_curr_i]
                    fonti = fonti_map.index(font)
                
                # concentrate image on the character 
                concentrated_img = concentrateImg(img, char_BB)

                # create entry in the 'by_char' dictionary (if necessary)
                if str(char) not in by_char_indexing.keys():
                    by_char_indexing[str(char)] = np.empty(0, dtype=int)
                    
                # create entry in the 'by_img' dictionary (if necessary)
                if img_name not in by_img_indexing.keys():
                    by_img_indexing[img_name] = np.array(global_curr_i)
                
                # add the new character image to the results:
                result_imgs.append(concentrated_img)
                result_chars.append(char)
                if fonts_included: result_fontis.append(fonti)
                # add current example to 'by_char' dictionary
                by_char_indexing[str(char)] = np.append(by_char_indexing[str(char)], global_curr_i)
                # add current example to the current word's indices
                curr_word_indices = np.append(curr_word_indices, global_curr_i)
                
                global_curr_i += 1
                local_curr_i += 1
                
            by_word_indexing.append(curr_word_indices) # add current word to 'by_word' indexing
        
        if verbose: print('done.')
    
    return result_imgs, result_chars, result_fontis, by_char_indexing, by_word_indexing, by_img_indexing


'''
BuildDataset : builds a processed version of a subset of a datset into an h5py group.
    Input:
        > 'db' - HDF5 (opened) file holding the source dataset.
        > 'img_names' - list of names of images in the dataset that you wish to process.
        > 'grp' - h5py group, into which you wish to load the processed dataset.
    Results:
        The h5py group 'grp' containing processed information about the data in 'db', 
        in the format discussed at the top of this file.
'''
def buildDataset(db, img_names, grp):
    # process the input dataset
    (result_imgs, result_chars, result_fontis, by_char_results, by_word_results,
         by_img_results) = processDataset(db, img_names)
    
    # organize the data in the given HDF5 group
    data_sgrp = grp.create_group('data')
    inedxing_sgrp = grp.create_group('indexing')
    by_char_sgrp = inedxing_sgrp.create_group('by_char')
    by_word_sgrp = inedxing_sgrp.create_group('by_word')
    by_img_sgrp = inedxing_sgrp.create_group('by_img')
    
    data_sgrp.create_dataset('images', data=result_imgs)
    data_sgrp.create_dataset('fontis', data=result_fontis)
    data_sgrp.create_dataset('chars', data=result_chars)
    
    for char_ascii in by_char_results.keys():
        by_char_sgrp.create_dataset(char_ascii, data=by_char_results[char_ascii])
    for word_i, indices in enumerate(by_word_results):
        by_word_sgrp.create_dataset(str(word_i), data=indices)
    for img_name in by_img_results.keys():
        by_img_sgrp.create_dataset(img_name, data=by_img_results[img_name])


'''
Main program: build-up a neater processed version of the dataset at 'SynthText.h'
Requirements: the 'SynthText.h5' dataset being the current working directory.
Produces: file 'NeatSynthText.h5' holding the processed version, in the format described
    at the top of this file.
'''
def buildup_main():
    with h5py.File('SynthText.h5', 'r') as db: # import original HDF5 file
        with h5py.File('NeatSynthText.h5', 'w') as new_db: # create new HDF5 file
                  
            # collect image names and randomly split to train-test-validation (60%-20%-20%)
            names = list(db['data'].keys())
            random.Random(100).shuffle(names)
            train_names, test_names, val_names = splitList(names, 0.6, 0.2, 0.2)
                   
            # create main groups
            train_grp = new_db.create_group('train')
            test_grp = new_db.create_group('test')
            val_grp = new_db.create_group('val')
            
            # fill the groups
            print('Building training set...')
            buildDataset(db, train_names, train_grp)
            print('Done building training set.\n')
            print('Building test set...')
            buildDataset(db, test_names, test_grp)
            print('Done building test set.\n')
            print('Building validation set...')
            buildDataset(db, val_names, val_grp)
            print('Done building validation set.\n')

if __name__ == '__main__':
    buildup_main()