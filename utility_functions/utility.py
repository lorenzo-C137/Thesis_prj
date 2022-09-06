# import
import wandb
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
import tensorflow as tf
import os
from skimage.morphology import erosion

# download data
def download_data(project_name, artifact_path, entity='lorenzob', files_list=['images.npy', 'annotations.npy']):

    with wandb.init(entity=entity, project=project_name, job_type="downloading_data") as run:

        list_of_files = dict()
        # we first retrive the artifact object
        artifact = run.use_artifact(artifact_path)

        # if we want to download all the files of the dataset
        if files_list==None:
            path = artifact.download()
            for root, dirs, files in os.walk(path, topdown=True):
                for file in files:
                    list_of_files[file]  = np.load(path+'/'+file)

        # if we want to download the only some files
        else:
            for file in files_list:
                # we then assign to a variable the path to the wanted file
                path = artifact.get_path(file).download()
                list_of_files[file]  = np.load(path)

    return list_of_files

# postprocess
def postprocess(y_pred, intensity_img, m_area_trsh=150,
                m_intensity_trsh=0.15, round_n_trsh=0.5,
                n_area_trsh=50):

    '''
    y_pred: 2D array, binary annotation
    intensity_img: 3 D array (H, W, Ch), where the first slice (H, W, 0) is microglia and
        the second one is nuclei intensity image; default values are chosen supposing
        intensity_img values are in range [0-1].
    '''

    # check if the prediction and the instensity images have the same shape
    assert y_pred.shape == (intensity_img.shape[0], intensity_img.shape[1])

    # check intensity_img dimensions
    assert len(intensity_img.shape) == 3
    assert intensity_img.shape[2] == 2

    # check if y_pred is binary
    assert (np.unique(y_pred) == [0, 1]).all()

    # check if the nucleus slice is between 0 and 1
    assert (np.min(intensity_img[:, :, 1]) >= 0, np.max(intensity_img[:, :, 1])) == (True, 1)

    # assign different label for each instance
    pred_lab = label(y_pred, connectivity=2) 

    # transform nuceli intensity image in binary image and concatenate it at the end
    # of intensity image
    rounded_n_img = np.where(intensity_img[:, :, 1] >= round_n_trsh, 1, 0)
    intensity_img = np.concatenate((intensity_img, np.expand_dims(rounded_n_img, axis=-1)), axis=-1)

    # create a mask that account for nuclei inside predicted microglia cells
    # Replace Nuceli mask with this newly created mask (n_in_m)
    n_in_m = np.where(pred_lab > 0, 1, 0) * intensity_img[:,:, 2] # M binary mask x N binary mask
    intensity_img[:, :, 2] = n_in_m


    # For area

    # list of objects. each object represents a single instance (predicted cell pixels).
    # Each object has multiple attributes which can be accessed
    props = regionprops(pred_lab, intensity_img)

    # list of labels that represents instances to be deleted from y_pred
    idxs = []

    for obj in props:
        # array of intensities of microglia in the region of 'n_in_m'
        arr = obj.image_intensity[:, :, 0][np.ma.make_mask(obj.image_intensity[:, :, 2])]

        # arr is empty when the respective n_in_m mask is full of False
        if arr.size == 0:
            int_micr = 0 # intensity of microglia
        else:
            int_micr = np.mean(arr)

        if obj.area < m_area_trsh or int_micr < m_intensity_trsh or arr.size < n_area_trsh:
            # appending label to list of labels
            idxs.append(obj.label)

    # transform from list to array
    idxs = np.array(idxs)

    print('Number of instances to be deleted: ',f'{idxs.size}/{len(props)} ({idxs.size / len(props):.5f})')

    # creating the binary mask to then modify y_pred
    # check if element of 'pred_lab' is in 'idxs' and if it is, return 1 in the 
    # same position of the checked element in pred_lab
    mask = np.isin(pred_lab, idxs)

    #compute percentage of predicted 1 that will be turned to 0 w.r.t. total 1 in y_pred
    perc_true = (np.sum(mask)) / np.sum(y_pred)
    print(f'Percentage of pixels deleted among y_pred==1 : {perc_true:.5f}', '\n')

    # creating new prediction image
    new_pred = np.where(mask == 1, 0, y_pred)

    return new_pred

# dataset creation

# 2D
def my_2Dtensor(img, annot,
              signal_treshold=None,
              step_i_j=None,
              count=0,
              img_size=None, 
              Y_sum_tresh=-1
              ):  
    '''
    This function takes 2 images as input (original_image and an annotation), see
    which sub-images in original_image of size (img_size, img_size) have a mean 
    value grater than signal_treshold, and put them in the X array if they have at least Y_sum_tresh
    number of annotations pixels; put the respective
    annotation sub-images in the Y array
    '''

    '''
    FIRST WE DO THIS 2 LOOPS TO HAVE THE NUMBER OF FINAL SUB-IMAGES TO CONSIDER
    TO CREATE THE TENSORS X AND Y OF THE RIGHT SHAPE

    img: is a 2D array (size1, size2) representing either Micorglia or Nuclei
    count: will be the number of sub-images to consider (shape=(img_size, img_size), np.mean(sub-image) > signal_treshold)
    '''
    # check that the input image is 2D and print its shape
    print('Input image shape for my_2Dtensor: ', img.shape)
    assert len(img.shape) == 2

    # check divisibility
    if img.shape[0]%img_size == 0 and img.shape[1]%img_size == 0:
        print(f'Image divisible by img_size={img_size}')
    else:
        print(f'Image NOT divisible by img_size={img_size}')

    # set variables related to height and width of the image
    height = img.shape[-2]
    width = img.shape[-1]

    # Iterate through rows of the image
    for i in np.arange(0, height, step_i_j):
        
        # Iterate through columns of the image
        for j in np.arange(0, width, step_i_j):

            # choosing indexes which don't go above the image dimensions
            row = min(i+img_size, height) 
            col = min(j+img_size, width)

            # the sub-square 
            new_X_array = img[i: row, j: col]
            new_Y_array = annot[i: row, j: col]
            
            # checking if the sub-square taken is the right shape since numpy doesn't do 'out of bounds check'
            if new_X_array.shape == (img_size, img_size) and np.mean(new_X_array) > signal_treshold and np.sum(new_Y_array[:, :]) > Y_sum_tresh:
                count += 1

    '''
    HERE WE DO THE SAME LOOP BUT WE FIRST ZEROED THE VARIABLE count AND CREATE THE 2 TENSORS X AND Y.
    THEN WE ASSIGN TO EACH SLICE OF X AND Y THE RIGHT VALUES
    '''
    X = np.full( (count, img_size, img_size, 1), -1, dtype=np.float32) # tensor of original images
    Y = np.full( (count, img_size, img_size, 1), -1, dtype=np.float32) # tensor of annotation images
    count = 0

    for i in np.arange(0, height, step_i_j):
        
        # Iterate through columns of the image
        for j in np.arange(0, width, step_i_j):

            # choosing indexes which don't go above the image dimensions
            row = min(i+img_size, height)
            col = min(j+img_size, width)

            # the sub-square 
            new_X_array = img[i: row, j: col]
            new_Y_array = annot[i: row, j: col]
            
            # checking if the sub-square taken is the right shape since numpy doesn't do 'out of bounds check'
            if new_X_array.shape == (img_size, img_size) and np.mean(new_X_array) > signal_treshold and np.sum(new_Y_array[:, :]) > Y_sum_tresh:

                X[count, :, :, 0] = new_X_array 
                Y[count, :, :, 0] = new_Y_array

                count += 1

    dataset = [X, Y]
    return dataset

# 3D
def my_3Dtensor(img, annot,
              signal_treshold=None,
              step_i_j = None,
              count = 0,
              img_size = None, # n_of features wanted
              Y_sum_tresh = -1,
              ):
    '''
    This function takes 2 images as input (original_image(3D) and an annotation), see
    which sub-images in original_image of size (img_size, img_size) have a mean 
    value grater than signal_treshold, and put them in the X array; put the respective
    annotation sub-images in the Y array
    '''

    '''
    FIRST WE DO THIS 2 LOOPS TO HAVE THE NUMBER OF FINAL SUB-IMAGES TO CONSIDER
    TO CREATE THE TENSORS X AND Y OF THE RIGHT SHAPE

    img: is a 3D array (size1, size2, 2) for which we suppose that the first slice (:, :, 0) is 
         from MICROGLIA while the second one from NUCLEI
    count: will be the number of sub-images to consider (shape=(img_size, img_size), np.mean(sub-image) > signal_treshold)
    '''
    # check that the input image is 2D and print its shape
    print('Input image shape for my_3Dtensor: ', img.shape)
    assert len(img.shape) == 3
    assert img.shape[2] == 2

    # check divisibility
    if img.shape[0]%img_size == 0 and img.shape[1]%img_size == 0:
        print(f'Image divisible by img_size={img_size}')
    else:
        print(f'Image NOT divisible by img_size={img_size}')

    # set variables related to height and width of the image
    height = img.shape[0]
    width = img.shape[1]

    # Iterate through rows of the image
    for i in np.arange(0, height, step_i_j):
        
        # Iterate through columns of the image
        for j in np.arange(0, width, step_i_j):

            # choosing indexes which don't go above the image dimensions
            row = min(i+img_size, height)
            col = min(j+img_size, width)

            # the sub-square 
            new_X_array = img[i: row, j: col, :]
            new_Y_array = annot[i: row, j: col]
            
            # checking if the sub-square taken is the right shape since numpy doesn't do 'out of bounds check'
            if new_X_array.shape == (img_size, img_size, 2) and np.mean(new_X_array[:, :, 0]) > signal_treshold and np.sum(new_Y_array[:, :]) > Y_sum_tresh:
                count += 1

    '''
    HERE WE DO THE SAME LOOP BUT WE FIRST ZEROED THE VARIABLE count AND CREATE THE 2 TENSORS X AND Y.
    THEN WE ASSIGN TO EACH SLICE OF X AND Y THE RIGHT VALUES
    '''
    X = np.full( (count, img_size, img_size, 2), -1, dtype=np.float32) # tensor of original images
    Y = np.full( (count, img_size, img_size, 1), -1, dtype=np.float32) # tensor of annotation images
    count = 0

    for i in np.arange(0, height, step_i_j):
        
        # Iterate through columns of the image
        for j in np.arange(0, width, step_i_j):

            # choosing indexes which don't go above the image dimensions
            row = min(i+img_size, height)
            col = min(j+img_size, width)

            # the sub-square 
            new_X_array = img[i: row, j: col, :]
            new_Y_array = annot[i: row, j: col]
            
            # checking if the sub-square taken is the right shape since numpy doesn't do 'out of bounds check'
            if new_X_array.shape == (img_size, img_size, 2) and np.mean(new_X_array[:, :, 0]) > signal_treshold and np.sum(new_Y_array[:, :]) > Y_sum_tresh:

                X[count, :, :, :] = new_X_array 
                Y[count, :, :, 0] = new_Y_array

                count += 1

    dataset = [X, Y]
    return dataset

# 2D-3D chice function
def my_tensors(img, annot,
              signal_treshold=None,
              step_i_j = None,
              count = 0,
              img_size = None, # n_of features wanted
              Y_sum_tresh = -1,
              ):

      args = dict(img=img, 
            annot=annot,
            signal_treshold=signal_treshold,
            step_i_j=step_i_j,
            count=count,
            img_size=img_size,
            Y_sum_tresh=Y_sum_tresh,)


      # chosing the approriate function

      if len(img.shape) == 2:
            dataset = my_2Dtensor(**args)
      else:
            dataset = my_3Dtensor(**args)

      return dataset

# output image

# 2D input image
def predict_image(model, img, sub_img_size, y_sum_tresh):
    # compute number of pixels corresponding to percentage given in y_sum_tresh
    y_sum_tresh = sub_img_size*sub_img_size*y_sum_tresh

    count = 0
    # check that the image is exactly divisible by the sub-squres chosen 
    assert img.shape[0]%sub_img_size == 0
    assert img.shape[1]%sub_img_size == 0

    img = img/np.max(img)
    y_pred = np.full((img.shape), -1, dtype=np.float32)

    for i in np.arange(0, img.shape[0], sub_img_size):
        
        # Iterate through columns of the image
        for j in np.arange(0, img.shape[1], sub_img_size):

            # the sub-square 
            x = np.expand_dims(img[i:i+sub_img_size, j:j+sub_img_size], axis=(0, -1))
            pred = tf.squeeze(model.predict(x))

            # put to 0 predicted sub-square of background that after normalization
            # resulted having lot of signal
            if np.sum(pred) > y_sum_tresh:
                pred = pred * 0
            
            y_pred[ i:i+sub_img_size, j:j+sub_img_size] = pred 
    return y_pred

# 3D input image
def predict_image_3D(model, img, sub_img_size, y_sum_tresh):

    # compute number of pixels corresponding to percentage given in y_sum_tresh
    y_sum_tresh = sub_img_size*sub_img_size*y_sum_tresh

    count = 0
    # check that the image is exactly divisible by the sub-squres chosen 
    assert img.shape[-3]%sub_img_size == 0
    assert img.shape[-2]%sub_img_size == 0

    y_pred = np.full((img.shape[0:2]), -1, dtype=np.float32)

    for i in np.arange(0, img.shape[0], sub_img_size):
        
        # Iterate through columns of the image
        for j in np.arange(0, img.shape[1], sub_img_size):

            # the sub-square 
            x = np.expand_dims(img[i:i+sub_img_size, j:j+sub_img_size, :], axis=(0))
            pred = tf.squeeze(model.predict(x))
            
            # round to 0 or 1 the prediction
            pred = np.round(pred)
            
            # put to 0 predicted sub-square of background that after normalization
            # resulted having lot of signal
            if np.sum(pred) > y_sum_tresh:
                pred = pred * 0
            
            y_pred[ i:i+sub_img_size, j:j+sub_img_size] = pred 
    return y_pred

# normalization

# square-by-square
def normalize(img,
              step_i_j = None,
              img_size = None, 
              sign_tresh = 50,
              mean_coeff = 1.5
              ):  
    '''
    This function takes as input an image and apply, square-by-square, first a standardization
    then a min-max scaling and eventually an original modification that keeps
    only values greater than mean_coeff, which represents the ratio = value/mean.
    sign_tresh default value is taken assuming img values are between 0 and 4096
    '''

    # check that the input image is 2D and print its shape
    print('Input image shape for normalize: ', img.shape)
    assert len(img.shape) == 2


    # set variables related to height and width of the image
    height = img.shape[-2]
    width = img.shape[-1]
    norm_img = np.full(img.shape, -1, dtype=np.float32)
    # Iterate through rows of the image
    for i in np.arange(0, height, step_i_j):
        
        # Iterate through columns of the image
        for j in np.arange(0, width, step_i_j):

            # choosing indexes which don't go above the image dimensions
            row = min(i+img_size, height) 
            col = min(j+img_size, width)

            # the sub-square 
            new_X_array = img[i: row, j: col]
            
            values = new_X_array
            mean = values.mean()
            square = ((values - mean) ** 2).sum()
            std  = np.sqrt( square / (img_size**2))

            # standardization
            norm = (new_X_array - mean) / std

            # min-max scaling
            min_norm, max_norm = np.min(norm), np.max(norm)    
            scaled_img = (norm - min_norm)/ (max_norm - min_norm)
            
            # multiply by zero all sub-square with mean signal from original image,
            # lower than specified value
            if np.mean(new_X_array) < sign_tresh:
                scaled_img = scaled_img * 0
            
            # MYNORM_3
            if np.mean(scaled_img) > 0:
                coeff = scaled_img / np.mean(scaled_img)
                scaled_img = np.where(coeff<mean_coeff, 0, scaled_img)
            
            norm_img[i: row, j: col] = scaled_img

    return norm_img

# pixel-by-pixel
def normalize_px(img,
              step_i_j=None,
              img_size=None, 
              sign_tresh=50,
              mean_coeff=1.5
              ):  
    '''
    This function takes as input an image and apply modifications, pixel-by-pixel,
    considering the square around that pixel to compute mean and std; first a standardization
    then a min-max scaling and eventually an original modification that keeps
    only values greater than mean_coeff, which represents the ratio = value/mean
    '''

    # check that the input image is 2D and print its shape
    print('Input image shape for normalize_px: ', img.shape)
    assert len(img.shape) == 2


    # set variables related to height and width of the image
    height = img.shape[-2]
    width = img.shape[-1]
    norm_img = np.full(img.shape, -1, dtype=np.float32)
    
    # Iterate through rows of the image
    for i in np.arange(0, height-img_size, step_i_j):
        
        # Iterate through columns of the image
        for j in np.arange(0, width-img_size, step_i_j):

            # choosing indexes which don't go above the image dimensions
            row = min(i+img_size, height) 
            col = min(j+img_size, width)

            # the sub-square 
            sub_square = img[i: row, j: col]
            
            mean = sub_square.mean()
            sum_var= ((sub_square - mean) ** 2).sum()
            std  = np.sqrt( sum_var / (img_size**2))

            # standardization
            norm = (sub_square - mean) / std

            # min-max scaling
            min_norm, max_norm = np.min(norm), np.max(norm)    
            scaled_img = (norm - min_norm) / (max_norm - min_norm)
            
            # multiply by zero all sub-square with mean signal from original image,
            # lower than specified value
            if np.mean(sub_square) < sign_tresh:
                scaled_img = scaled_img * 0

            # my normalization
            if np.mean(scaled_img) > 0:
                coeff = scaled_img / np.mean(scaled_img)

                # MYNORM3
                scaled_img = np.where(coeff<mean_coeff, 0, scaled_img)

                # # MYNORM2
                # scaled_img = np.where(coeff<mean_coeff, scaled_img*coeff**5, scaled_img)

            norm_img[i + img_size//2, j+img_size//2] = scaled_img[img_size//2, img_size//2]
            
    return norm_img

# cell count evaluation

# confusion matrix adjusted
def cellcount_eval(ann, pred_bod):
    '''
    ann: binary image [0/1] representing annotated cells
    pred_bod: biinary image [0/1] of the predicted cells bodies
    '''

    # checks that the 2 images are binary
    assert np.array_equal(np.unique(ann), [0.0, 1.0])
    assert np.array_equal(np.unique(pred_bod), [0.0, 1.0])

    # predicted postives
    labels_pred = label(pred_bod, connectivity=2)
    pp = np.max(labels_pred)
    print(f'Predicted cells bodies: {pp}')

    # mask of bodies inside cells
    bod_in_cell = pred_bod * ann
    labels_tp = label(bod_in_cell, connectivity=2)
    tp_all = np.max(labels_tp)
    print(f'Predcited cells bodies inside annotations: {tp_all}', '\n')

    

    # summing annotations and bodies in cells to have 2 as max where the body cell
    # is predicted inside a cell
    summed = ann + bod_in_cell

    labels = label(ann, connectivity=2)
    props = regionprops(labels, intensity_image=summed)

    # list of indices to be deleted
    idxs = []

    # if in a region there's a 2, it means that a body cell (or multiple) was over 
    # the annotation of that region
    for obj in props:
        if np.max(obj.image_intensity) == 2:
            idxs.append(obj.label)
    
    idxs = np.array(idxs)
    tp = idxs.size
    fn = len(props) - idxs.size

    # number of cell bodies not inside annotations
    fp_out = pp - tp_all
    # nimber of cell bodies inside annotations but multiple in one cell
    fp_in = tp_all - tp

    # real false positives,considering as such all bodies outside annotations 
    # or multiples inside a single cell
    fp = fp_out + fp_in

    print('Number of cells detected (TP / P): ',f'{tp}/{len(props)} ({tp / len(props):.5f})')
    print('Number of cells not detected (FN / P): ',f'{fn}/{len(props)} ({fn / len(props):.5f})')
    print('Number of FP (FP): ',f'{fp} ({fp_in} inside, {fp_out} outside)', '\n')
    print(f'F1 score: {2*tp/(2*tp + fp + fn):.5f}' )
    print(f'Recall: {tp/len(props):.5f}')
    print(f'Precision: {tp/(tp + fp):.5f}')
    # mask of tp
    mask = np.isin(labels, idxs)

    # creating new prediction image
    fn_mask = np.where(mask == 1, 0, ann)

    return fn_mask

# cell count by erosion
def cellcount_erosion(y_pred, int_img, erosion_num=4,):
    eroded = y_pred
    for i in range(erosion_num):
        eroded = erosion(eroded)
    eroded_pp = postprocess(eroded, int_img, round_n_trsh=0, m_intensity_trsh=0, m_area_trsh=200)
    labels = label(eroded_pp)
    return np.max(labels), eroded_pp