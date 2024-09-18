import cv2 as cv
import numpy as np
import os

def flattening(filename):
    image=cv.cvtColor(cv.imread(filename),cv.COLOR_BGR2GRAY)
    rows,cols=image.shape
    vector=[]
    for row in range(rows):
        for col in range(cols):
            vector.append(image[row,col])
    return np.array(vector)

def image_matrix(directory):
    images=os.listdir(directory)
    matrix=[]
    for image in images:
        image=os.path.join(directory,image)
        matrix.append(flattening(image))
    return np.array(matrix)

def training():
    training_matrix=image_matrix('Training Set')
    mean=np.mean(training_matrix,axis=0)
    centered_matrix=training_matrix-mean
    U,V,Vector=np.linalg.svd(centered_matrix,full_matrices=False)
    projected_matrix=np.dot(centered_matrix,Vector.T) 
    return mean,Vector,projected_matrix

def image_recognition():
    mean,Vector,training_projected_matrix=training()
    validation_image=flattening('Validation Set/image.png')
    validation_image_centered=validation_image-mean
    validation_projected_matrix=np.dot(validation_image_centered,Vector.T)
    euclidean_distances=np.linalg.norm(training_projected_matrix-validation_projected_matrix,axis=1)
    print(f"Euclidean distances:{euclidean_distances}")

if __name__=='__main__':
    image_recognition()