import cv2 as cv
import numpy as np
import os

def flattening(filename):
    image=cv.cvtColor(cv.imread(filename),cv.COLOR_BGR2GRAY)
    return image.flatten()

def image_matrix(directory):
    images=os.listdir(directory)
    matrix=[]
    for image in images:
        if image.lower().endswith('.jpeg'):
            image_path=os.path.join(directory,image)
            matrix.append(flattening(image_path))
    return np.array(matrix)

def training():
    training_matrix=[]
    names=[]
    for person_name in os.listdir('Training Set'):
        person_directory=os.path.join('Training Set',person_name)
        if os.path.isdir(person_directory):
            person_matrix=image_matrix(person_directory)
            training_matrix.append(person_matrix)
            names+=[person_name]*person_matrix.shape[0]
    training_matrix=np.concatenate(training_matrix)
    mean=np.mean(training_matrix,axis=0)
    centered_matrix=training_matrix-mean
    U,V,Vector=np.linalg.svd(centered_matrix,full_matrices=False)
    projected_matrix=np.dot(centered_matrix,Vector.T)
    return mean,Vector,projected_matrix,names

def recognize_face(validation_matrix,mean,Vector,training_projected_matrix):
    centered_matrix=validation_matrix-mean
    validation_projected_matrix=np.dot(centered_matrix,Vector.T)
    euclidean_distances=np.linalg.norm(training_projected_matrix-validation_projected_matrix,axis=1)
    return euclidean_distances

def image_recognition():
    mean,Vector,training_projected_matrix,names=training()
    validation_images=os.listdir('Validation Set')
    for validation_image in validation_images:
        validation_matrix=flattening(os.path.join('Validation Set',validation_image))
        euclidean_distances=recognize_face(validation_matrix,mean,Vector,training_projected_matrix)
        nearest_index=np.argmin(euclidean_distances)
        recognized_name=names[nearest_index]
        print(f"Validation image: '{validation_image}', Recognized image: {recognized_name}")
            
if __name__=='__main__':
    image_recognition()