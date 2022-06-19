'''
===============================================================================================
classifying.py
Dependencies: tensorflow, numpy, h5py; module buildup.py.
Requirements for main program: the 'models' directory produced by 'training.py', and
    'SynthText_test.h5' synthetic text dataset, being in the current working directory.

Nadav Kahlon, January 2022.
===============================================================================================

This program is the core of the font-classifier.
It uses the data in the 'models' directory produced by 'training.py' to make predictions
on new data, specifically - words of a given font.

The method for classification is the following:
    > First - the word is broken up into the characters that form it, and concentrated images
        are formed (using 'concentrateImg' defined in 'buildup.py')
    > Second - for each character, a model is picked from the 'models' directory: if the
        character has its own model, it is picked; otherwise - the global one is picked.
    > Third - the output of the selected model is evaluated (for each character on its own
        model).
    > And finally - the weighted sum of the output vectors is calculated (using the
        weights stored in "models/weights.h5'), and the highest component is selected
        as the classification output.

The module offers the 'evaluate' function for evaluation of the (optionally weighted) output
probabilities vector produced for every example in a given dataset. It also offers the 'predict'
function used for final classification of examples in a dataset. For full detail - see the
documentation above the definition of each of those functions, below.

The main program uses the input dataset found at 'SynthText_test.h5', and produces an output  
file 'results.csv' containing information about the predicted font of every character 
appearing in the input dataset.

'''

from buildup import processDataset, fonti_map, total_fonts
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session as keras_clear_session
from tensorflow.compat.v1 import logging
import numpy as np
import h5py
import csv
import gc
   

'''
Utility values used by the main program
'''
# names for the different fields in the output csv file.
fieldnames = [' ', 'image', 'char', b'Raleway', b'Open Sans', b'Roboto',
              b'Ubuntu Mono', b'Michroma', b'Alex Brush', b'Russo One']
# the size of the partitions of the test set we are going to process individualy
batch_size = 100


'''
rawEvaluate : evaluates raw output of the models in 'models' directory.
    Input
        > 'images' - a numpy array of shape (N,64,64,3), representing a list of N 64-by-64
            pixels image of characters (concentrated using the 'concentrateImg' function defined
            in 'buildup.py').
        > 'by_char_indexing' a dictionary (or dictionary-like) indexed by strings containing
            ASCII values of characters. Each entry should be a 1-dimensional numpy array (or
            array-like) of indices of images of the corresponding character.
        > 'weighted' - a boolean, representing wether or not output vectors should be weighted.
        > 'verbose' - a boolean, representing wether or not you wish to see the process documented
            in real time in the terminal.
    Returns
        An N-by-'total_fonts' numpy array. The [i,j] element of it is the predicted probability of the
        font of the character in the i'th image being the j'th font (in the 'fonti_map'
        array defined in 'buildup.py'). If 'weighted' is True, each output vector is multiplied
        by the weight of the model that generated it.
    Requirements
        > The 'models' directory (produced by 'training.py'), being in the current working
           directory.
'''
def rawEvaluate(images, by_char_indexing, weighted=False, verbose=True):
    
    # get fundamental information:
    N = images.shape[0]
    weights_file = h5py.File('models/weights.h5', 'r')
    
    results = np.empty((N, total_fonts)) # numpy array to hold the results
    processed = [] # list of indices already-processed examples.
    
    # prevent warning on repeatedly calling predict 
    logging.set_verbosity(logging.ERROR)
    
    # process images of characters having an individual model in 'models' directory 
    for fname in weights_file.keys():
        if fname != 'global' and fname != 'weights.h5' and fname in by_char_indexing.keys():
            model_name = fname
            
            # evaluate output vectors on those images:
            if verbose: print('\t> Process examples of character \'' + chr(int(model_name)) + '\':', end=' ')
            if verbose: print('Load model...', end=' ')
            indices = by_char_indexing[model_name][()]
            model = load_model('models/'+model_name)
            if verbose: print('done. Evaluate raw output...', end=' ')
            results[indices] = model.predict(images[indices])
            if weighted: results[indices] *= weights_file[model_name]
            if verbose: print('done.')
            
            # record those images as processed
            processed += indices.tolist()
            
            # cleanup
            keras_clear_session()
            del(model)
            gc.collect()
    
    # now, use the global model to evaluate output for the rest of the examples:
    if verbose: print('\t> Rest of examples: Load model...', end=' ')
    indices = [i for i in range(N) if i not in processed]
    model = load_model('models/global')
    if verbose: print('done. Evaluate raw output...', end=' ')
    results[indices] = model.predict(images[indices])
    if weighted: results[indices] *= weights_file['global']
    if verbose: print('done.')
    
    # cleanup
    keras_clear_session()
    del(model)
    gc.collect()
    
    if weighted: weights_file.close()
    return results


'''
evaluate : evaluates predicted probability vectors.
    Input:
        > 'images' - a numpy array of shape (N,64,64,3), representing a list of N 64-by-64
            pixels image of characters (concentrated using the 'concentrateImg' function defined
            in 'buildup.py').
        > 'by_char_indexing' a dictionary (or dictionary-like) indexed by strings containing
            ASCII values of characters. Each entry should be a 1-dimensional numpy array (or
            array-like) of indices of images of the corresponding character.
        > 'by_word_indexing' a list, or a dictionary (or a dictionary-like) indexed arbitrarily. 
            Each entry should be a 1-dimensional numpy array (or array-like) of indices of examples
            of the same word. Should cover all the examples.
        > 'verbose' - a boolean, representing wether or not you wish to see the process documented
            in real time in the terminal.
    Returns:
        An N-by-'total_fonts' numpy array. The [i,j] element of it is the predicted probability of the
        font of the character in the i'th image being the j'th font (in the 'fonti_map'
        array defined in 'buildup.py').
    Requirements:
        > The 'models' directory (produced by 'training.py'), being in the current working
           directory.
'''
def evaluate(images, by_char_indexing, by_word_indexing, verbose=True):
    # numpy array to hold the results
    results = np.empty((images.shape[0], total_fonts)) 
    
    # evaluate weighted output vectors
    if verbose: print('\tEvaluating raw output...')
    evaluations = rawEvaluate(images, by_char_indexing, weighted=True, verbose=verbose)
    if verbose: print('\tDone evaluating raw output.')
    if verbose: print('\tEvaluating final output...', end=' ')
    
    # make sure 'by_word_indexing' is a list (and not dictionary/dictionary-like indexed arbitrarily)
    if not isinstance(by_word_indexing, list):
        by_word_indexing = by_word_indexing.values()

    # process each word
    for indices in by_word_indexing:
        indices = indices[()] # convert to numpy array (if it is array-like)
        
        # calculate normalized weighted sum of estimated probability vectors
        weighted_sum = np.sum(evaluations[indices], axis=0)
        norm_weighted_sum = weighted_sum / max(np.sum(weighted_sum),1e-100)
        results[indices] = norm_weighted_sum
    
    if verbose: print('done.')
    return results


'''
predict : classifies examples to fonts.
    Input
        > 'images' - a numpy array of shape (N,64,64,3), representing a list of N 64-by-64
            pixels image of characters (concentrated using the 'concentrateImg' function defined
            in 'buildup.py').
        > 'by_char_indexing' a dictionary (or dictionary-like) indexed by strings containing
            ASCII values of characters. Each entry should be a 1-dimensional numpy array (or
            array-like) of indices of images of the corresponding character.
        > 'by_word_indexing' a list, or a dictionary (or a dictionary-like) indexed arbitrarily. 
            Each entry should be a 1-dimensional numpy array (or array-like) of indices of examples
            of the same word. Should cover all the examples.
        > 'verbose' - a boolean, representing wether or not you wish to see the process documented
           in real time in the terminal.
    Returns
        A numpy 1-dimensional array of N fontis. The i'th fonti is the predicted fonti for
        the character appearing in the i'th image (see 'fonti_map' in buildup.py).
    Requirements
        > The 'models' directory (produced by 'training.py'), being in the current working
           directory.
'''
def predict(images, by_char_indexing, by_word_indexing, verbose=True):
    # calculate probability vectors
    probabilities = evaluate(images, by_char_indexing, by_word_indexing, verbose=verbose)

    # return fontis of highest probability
    return np.argmax(probabilities, axis=1)                       


'''
dumpResultsToCSV : dumps font classifications (and other info) to csv file.
    Input
        > 'predictions' - list of N fontis. The i'th fonti is the predicted fonti for i'th
            example in testing set.
        > 'chars' - list of N ASCII values. The i'th value is the ASCII value of i'th example
            in testing set.
        > 'by_img_indexing' - a dictionary indexed by image names. Each entry in it is the 
            index of the first example in test set that belongs to the corresponding image
            (the rest are the ones that follow).
        > 'writer' - csv writer to write the results to.
        > 'in_writer_index' - the index of the next row in the output writer.
    Produces an output csv file named by 'output_fname' that contains information about
    the predictions.
'''
def dumpResultsToCSV(predictions, chars, by_img_indexing, writer, in_writer_index=0):
    # organize the image names in ascending order of appearances in test set
    img_names = list(by_img_indexing.keys())
    img_names.sort(key=lambda name: by_img_indexing[name])
    
    # setup accounting for checking the current image name
    N = len(predictions)
    curr_img_name = None if not img_names else img_names.pop(0)
    next_img_index = N if not img_names else by_img_indexing[img_names[0]]
    
    # process each example
    for i, (fonti, char) in enumerate(zip(predictions, chars)):
        # check if image name should be changed
        if i == next_img_index:
            curr_img_name = None if not img_names else img_names.pop(0)
            next_img_index = N if not img_names else by_img_indexing[img_names[0]]
    
        # record information regarding the current example in output file
        writer.writerow({
                ' ':               in_writer_index+i,
                'image':           curr_img_name, 
                'char':            chr(char),
                # use predicted fonti and the fonti map to determine font
                b'Raleway':        1 if fonti_map[fonti] == b'Raleway' else 0,
                b'Open Sans':      1 if fonti_map[fonti] == b'Open Sans' else 0,
                b'Roboto':         1 if fonti_map[fonti] == b'Roboto' else 0,
                b'Ubuntu Mono':    1 if fonti_map[fonti] == b'Ubuntu Mono' else 0,
                b'Michroma':       1 if fonti_map[fonti] == b'Michroma' else 0,
                b'Alex Brush':     1 if fonti_map[fonti] == b'Alex Brush' else 0,
                b'Russo One':      1 if fonti_map[fonti] == b'Russo One' else 0
                })


'''
devideToBatches : returns a list of partitions of a given list, with the given partition size.
'''
def partitionList(src_list, partition_size):
    partitions = [] # to hold the results
    start = 0 # index into the beginning of the current partition
    while start < len(src_list):
        end = min(start + partition_size, len(src_list))
        partitions.append(src_list[start : end])
        start = end
    return partitions
    

'''
Main program : uses the input dataset found at 'SynthText_test.h5', and produces an output file 
    'results.csv' containing information about the predicted font of every character appearing
    in it.
Requirements: The 'models' directory produced by 'training.py', being in the current
    working directory.
'''
def classifying_main():
    # set output dictionary writer
    with open('results.csv', 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames)
        writer.writeheader()
        in_writer_index = 0
        
        # load the test set and partition it to batches
        with h5py.File('SynthText_test.h5', 'r') as db:
            img_names = list(db['data'].keys())
            img_names_batches = partitionList(img_names, batch_size)
            
            for batch_i, img_names_batch in enumerate(img_names_batches):
                print('Process batch', batch_i+1, '/', len(img_names_batches), '...')
                
                # preprocess (concentrate the images on the characters in them)
                print('Preprocessing data...')
                processed_data = processDataset(db, img_names_batch, fonts_included=False)
                (imgs, chars, _, by_char_indexing, by_word_indexing, by_img_indexing) = processed_data
                print('Done preprocessing data for bacth', batch_i+1, '/', len(img_names_batches),'.\n')
                
                # make predictions
                print('Making predictions...')
                predictions = predict(np.array(imgs), by_char_indexing, by_word_indexing)
                print('Done making predictions for bacth', batch_i+1, '/', len(img_names_batches),'.\n')
                
                # dump results to csv file
                print('Dumping results to "results.csv"...', end=' ')
                dumpResultsToCSV(predictions, chars, by_img_indexing, writer, in_writer_index)
                in_writer_index += len(imgs)
                print('done.', end=' ')
                
                # cleanup
                print('Cleaning up...', end=' ')
                del (imgs, chars, _, by_char_indexing, by_word_indexing, by_img_indexing)
                del processed_data
                del predictions
                gc.collect()
                print('done.')
                
                print('Done processing batch', batch_i+1, '/', len(img_names_batches), '.\n')
        
    print('')
    print('Wish us luck,')
    print('Nadav ^_^')

if __name__ == '__main__':
    classifying_main()