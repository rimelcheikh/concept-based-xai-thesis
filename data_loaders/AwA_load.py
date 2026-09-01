import numpy as np
import  pandas as pd
import os
import scipy.io
import random

from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array


random.seed(42)
    
concepts = ['ocean-s', 'desert-s', 'forest-s', 'water-s', 'cave-s', 'black-c', 'brown-c', 'white-c', 'blue-c', 'orange-c', 'red-c', 'yellow-c']
      



#------------------------------------------------------
"""concepts_list = [c.split('-')[0] for c in concepts]
#Keeping lines that are Broden concepts
for c in mat_pd.index:
    if c.lower() not in concepts_list:
        mat_pd = mat_pd.drop(c.lower(), axis=0)"""





def load_CIFAR_data_all(unbalanced):
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0


    with open('../data/CIFAR100/cifar100_fine_labels.txt') as f:
        targets_list = f.read().splitlines()
        
    
    #get classes that are common to both AwA and CIFAR and keep their CIFAR indexes in ind_list
    targets = []
    ind = 0
    ind_list = []
    ind_list_2 = []
    for t in targets_list:
        if t in classes:
            targets.append(t)
            ind_list.append(ind)
            ind_list_2.append(targets_list.index(t))
        ind +=1    
        
    #Making dict to map CIFAR class index to class name
    idx_to_label_cifar = {}
    j = 0
    for i in ind_list:
        idx_to_label_cifar[i] = targets[j]
        j+=1


    #Get CIFAR index of images that are in desired classes
    y_train_idx, y_test_idx = [], []   
    for i in ind_list:
        y_train_idx.append(np.where(train_labels == i)[0])
        y_test_idx.append(np.where(test_labels == i)[0])

    #Flattening the above result to a 17*500 list
    i_train_idx_c, i_test_idx_c = [], []
    for i in range(len(y_train_idx)):
        for j in range(np.shape(y_train_idx[i])[0]):
            i_train_idx_c.append(y_train_idx[i][j])
    for i in range(len(y_test_idx)):
        for j in range(np.shape(y_test_idx[i])[0]):
            i_test_idx_c.append(y_test_idx[i][j])


    #Keeping images and CIFAR labels of the desired classes
    X_train, X_test = train_images[i_train_idx_c], test_images[i_test_idx_c]
    y_train_0, y_test_0 = train_labels[i_train_idx_c], test_labels[i_test_idx_c]

    
    #Get AwA matrix by keeping desired classes as columns 
    mat_pd_cifar = mat_pd_all
    for t in mat_pd_all.columns:
        if t.lower() not in targets:
            mat_pd_cifar = mat_pd_cifar.drop(t.lower(), axis=1)
            
    GT_matrix = mat_pd_cifar
            
    #Make a dict to map AwA class names to their index in the matrix columns
    label_to_idx_awa = {}
    for t in mat_pd_cifar.columns:
        label_to_idx_awa[t] = list(mat_pd_cifar.columns).index(t)
    concepts = list(mat_pd_cifar.index)

    #Label images with AwA indexes
    y_train, y_test = np.array([[0]]*len(y_train_0)), np.array([[0]]*len(y_test_0))
    for i in range(len(y_train)):
        y_train[i][0] = label_to_idx_awa[idx_to_label_cifar[y_train_0[i][0]]]
    for i in range(len(y_test)):
        y_test[i][0] = label_to_idx_awa[idx_to_label_cifar[y_test_0[i][0]]]
        
        
    attr_label_train = []
    for i in range(len(y_train)):   
        attr_label_train.append((mat_pd_cifar.iloc[:,y_train[i]].values/100).tolist())
        
    attr_label_test = []
    for i in range(len(y_test)):   
        attr_label_test.append((mat_pd_cifar.iloc[:,y_test[i]].values/100).tolist())
    
    a_train = np.array(attr_label_train).squeeze()
    a_test = np.array(attr_label_test).squeeze()
    
    
    
    #unbalanced_ratios = {class_idx: round(random.uniform(0.1, 0.9), 2) for class_idx in range(len(targets))}
    
    """unbalanced_ratios = {0: 0.27, 1: 0.31,
                         2: 0.85, 3: 0.62,
                         4: 0.59, 5: 0.24,
                         6: 0.68, 7: 0.23,
                         8: 0.4, 9: 0.89,
                         10: 0.61, 11: 0.55,
                         12: 0.65, 13: 0.77,
                         14: 0.72, 15: 0.28,
                         16: 0.13}

    # Apply unbalanced distribution if provided
    if unbalanced:
        # Initialize lists for unbalanced data
        unbalanced_X_train, unbalanced_y_train, unbalanced_a_train = [], [], []
        unbalanced_X_test, unbalanced_y_test, unbalanced_a_test = [], [], []
        for class_idx, ratio in unbalanced_ratios.items():
            class_indices_train = np.where(y_train == class_idx)[0]
            class_indices_test = np.where(y_test == class_idx)[0]
            sample_size_train = int(ratio * len(class_indices_train))  # Number of samples for the class
            sample_size_test = int(ratio * len(class_indices_test)) 
            sampled_indices_train = random.sample(list(class_indices_train), sample_size_train)
            sampled_indices_test = random.sample(list(class_indices_test), sample_size_test)
            
            # Append sampled data
            unbalanced_X_train.extend(X_train[sampled_indices_train])
            unbalanced_y_train.extend(y_train[sampled_indices_train])
            unbalanced_a_train.extend(a_train[sampled_indices_train])
            unbalanced_X_test.extend(X_test[sampled_indices_test])
            unbalanced_y_test.extend(y_test[sampled_indices_test])
            unbalanced_a_test.extend(a_test[sampled_indices_test])
        
        X_train = np.array(unbalanced_X_train)
        y_train = np.array(unbalanced_y_train)
        a_train = np.array(unbalanced_a_train)
        X_test = np.array(unbalanced_X_test)
        y_test = np.array(unbalanced_y_test)
        a_test = np.array(unbalanced_a_test)"""
      
    
    
    
    
    
    """mat_pd_cifar = mat_pd_cifar.replace(0, 0.1)
    mat_pd_cifar = 100. / mat_pd_cifar.to_numpy()
    mat_pd_cifar = (mat_pd_cifar-np.min(mat_pd_cifar))/(np.max(mat_pd_cifar)-np.min(mat_pd_cifar))"""
    
    mat_pd_cifar = np.where(mat_pd_cifar != 0, 100. / mat_pd_cifar, 0)   
    mat_pd_cifar = (mat_pd_cifar-np.min(mat_pd_cifar))/(np.max(mat_pd_cifar)-np.min(mat_pd_cifar))

    return X_train, X_test, y_train, y_test, a_train, a_test, mat_pd_cifar, targets, concepts , label_to_idx_awa, GT_matrix






def load_AwA_2(cv, data_dir):
    
    
    weights_matrix = np.loadtxt(data_dir+'/AwA/AwA2base/Animals_with_Attributes2/predicate-matrix-continuous.txt').transpose()
    classes = np.loadtxt(data_dir+'/AwA/AwA2base/Animals_with_Attributes2/classes.txt',dtype=str, usecols=1)
    attributes = np.loadtxt(data_dir+'/AwA/AwA2base/Animals_with_Attributes2/predicates.txt',dtype=str, usecols=1)
    mat_pd_all = pd.DataFrame(weights_matrix,index=attributes,columns=classes)
    mat_pd = mat_pd_all
    
    
    #Make a dict to map AwA class names to their index in the matrix columns
    label_to_idx_awa = {}
    for t in mat_pd.columns[1:]:
        label_to_idx_awa[t] = list(mat_pd.columns).index(t)-1
    
    dataset_dir = data_dir+'/AwA/JPEGImages'
    # Get the list of classes
    classes = sorted(os.listdir(dataset_dir))[1:]  # Sort classes to ensure consistent order
    
    
    
    #Prepare knowledge matrix
    mat_pd_awa = mat_pd_all.drop('antelope',axis=1) #contains -1 -> missing GT for some concepts

    mat_GT = mat_pd_awa

    mat_pd_awa = np.where(mat_pd_awa != 0, 100. / mat_pd_awa, 0)   
    mat_pd_awa = (mat_pd_awa-np.min(mat_pd_awa))/(np.max(mat_pd_awa)-np.min(mat_pd_awa))
    
    
    # Prepare lists to hold image data and their corresponding class labels
    images = []
    labels = []
    annotations = []
    
    # Load images and labels
    for c in classes:
        counter = 0
        class_dir = os.path.join(dataset_dir, c)
        for file in os.listdir(class_dir):
            if counter < 100 :
                counter += 1
                img_path = os.path.join(class_dir, file)
                img = load_img(img_path, target_size=(224, 224))  # Load image and resize to target size
                img_array = img_to_array(img)  # Convert image to numpy array
                images.append(img_array)
                labels.append(label_to_idx_awa[c])  # Use integer label
                annotations.append(mat_pd_all[c].values/100)

    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    annotations = np.array(annotations)
    
    
    
    if cv == False:
        X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
                                    images, labels, annotations, test_size=0.2, stratify=labels, random_state=42)
        X_train = X_train.astype('float32') 
        X_test = X_test.astype('float32') 

        return  X_train, X_test, y_train, y_test, a_train, a_test, mat_pd_awa, classes, list(mat_GT.index), label_to_idx_awa, mat_GT

    
    else:
        return images.astype('float32')/255., None, labels, None, annotations, None, mat_pd_awa, classes, list(mat_GT.index), label_to_idx_awa, mat_GT
        
    
    
def load_AwA_data_17(cv):
    
    with open('../data/CIFAR100/cifar100_fine_labels.txt') as f:
        targets_list = f.read().splitlines()
        
    classes = []
    for c in sorted(os.listdir('../data/AwA/JPEGImages'))[1:]:
        if c in targets_list:
            classes.append(c)
        
    
    mat_pd_ = mat_pd[classes]
    #Make a dict to map AwA class names to their index in the matrix columns
    label_to_idx_awa = {}
    for t in mat_pd_:
        label_to_idx_awa[t] = list(mat_pd_.columns).index(t)
    
    
    
    
    #Prepare knowledge matrix
    mat_GT = mat_pd_

    mat_pd_awa = np.where(mat_pd_ != 0, 100. / mat_pd_, 0)   
    mat_pd_awa = (mat_pd_awa-np.min(mat_pd_awa))/(np.max(mat_pd_awa)-np.min(mat_pd_awa))
    
    
    # Prepare lists to hold image data and their corresponding class labels
    images = []
    labels = []
    annotations = []
    
    # Load images and labels
    for c in classes:
        counter = 0
        class_dir = os.path.join('../data/AwA/JPEGImages', c)
        for file in os.listdir(class_dir):
            if counter < 100 :
                counter += 1
                img_path = os.path.join(class_dir, file)
                img = load_img(img_path, target_size=(224, 224))  # Load image and resize to target size
                img_array = img_to_array(img)  # Convert image to numpy array
                images.append(img_array)
                labels.append(label_to_idx_awa[c])  # Use integer label
                annotations.append(mat_pd_all[c].values/100)

    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    annotations = np.array(annotations)
    
    
    
    if cv == False:
        X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
                                    images, labels, annotations, test_size=0.2, stratify=labels, random_state=42)
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        return  X_train, X_test, y_train, y_test, a_train, a_test, mat_pd_awa, classes, list(mat_GT.index), label_to_idx_awa, mat_GT

    
    else:
        return images.astype('float32') / 255.0, None, labels, None, annotations, None, mat_pd_awa, classes, list(mat_GT.index), label_to_idx_awa, mat_GT
        
    

def load_ImageNet_data_all():
    dataset_dir = '../data/ImageNet/classes/'
    
    # Get the list of classes
    targets_list = sorted(os.listdir(dataset_dir))  # Sort classes to ensure consistent order
    
    
    #get classes that are common to both AwA and ImageNet and keep their ImageNet indexes in ind_list
    targets = []
    ind = 0
    ind_list = []
    ind_list_2 = []
    for t in targets_list:
        if t in classes:
            targets.append(t)
            ind_list.append(ind)
            ind_list_2.append(targets_list.index(t))
        ind +=1 
    
    
    #Making dict to map ImageNet class index to class name
    idx_to_label_imagenet = {}
    j = 0
    for i in ind_list:
        idx_to_label_imagenet[i] = targets[j]
        j+=1
    
    
    #Get AwA matrix by keeping desired classes as columns 
    mat_pd_imagenet = mat_pd_all
    for t in mat_pd_all.columns:
        if t.lower() not in targets:
            mat_pd_imagenet = mat_pd_imagenet.drop(t.lower(), axis=1)
    
    
    # Create a dictionary mapping class names to integer labels
    label_to_idx_imagenet = {cls: idx for idx, cls in enumerate(targets)}
    
    # Prepare lists to hold image data and their corresponding class labels
    images = []
    labels = []
    
    # Load images and labels
    for c in targets_list:
        counter = 0
        class_dir = os.path.join(dataset_dir, c)
        for file in os.listdir(class_dir):
            if counter < 5000 :
                counter += 1
                img_path = os.path.join(class_dir, file)
                img = load_img(img_path, target_size=(224, 224))  # Load image and resize to target size
                img_array = img_to_array(img)  # Convert image to numpy array
                images.append(img_array)
                labels.append(label_to_idx_imagenet[c])  # Use integer label

    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert labels to one-hot encoded format
    """label_encoder = LabelBinarizer()
    labels = label_encoder.fit_transform(labels)"""
    
    X_train, X_test, y_train, y_test = train_test_split(
                                images, labels, test_size=0.2, stratify=labels, random_state=42)
    
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return  X_train, X_test, y_train, y_test, mat_pd_imagenet, classes, concepts, label_to_idx_imagenet






def load_ImageNet_data():
    
    dataset_dir = '../data/ImageNet/classes/'
    
    # Get the list of classes
    targets_list = sorted(os.listdir(dataset_dir))  # Sort classes to ensure consistent order
    
    
    #get classes that are common to both AwA and ImageNet and keep their ImageNet indexes in ind_list
    targets = []
    ind = 0
    ind_list = []
    ind_list_2 = []
    for t in targets_list:
        if t in classes:
            targets.append(t)
            ind_list.append(ind)
            ind_list_2.append(targets_list.index(t))
        ind +=1 
    
    
    #Making dict to map ImageNet class index to class name
    idx_to_label_imagenet = {}
    j = 0
    for i in ind_list:
        idx_to_label_imagenet[i] = targets[j]
        j+=1
    
    
    #Get AwA matrix by keeping desired classes as columns 
    mat_pd_imagenet = mat_pd
    for t in mat_pd.columns:
        if t.lower() not in targets:
            mat_pd_imagenet = mat_pd_imagenet.drop(t.lower(), axis=1)
    
    
    # Create a dictionary mapping class names to integer labels
    label_to_idx_imagenet = {cls: idx for idx, cls in enumerate(targets)}
    
    # Prepare lists to hold image data and their corresponding class labels
    images = []
    labels = []
    
    # Load images and labels
    for c in targets_list:
        counter = 0
        class_dir = os.path.join(dataset_dir, c)
        for file in os.listdir(class_dir):
            if counter < 5000 :
                counter += 1
                img_path = os.path.join(class_dir, file)
                img = load_img(img_path, target_size=(224, 224))  # Load image and resize to target size
                img_array = img_to_array(img)  # Convert image to numpy array
                images.append(img_array)
                labels.append(label_to_idx_imagenet[c])  # Use integer label

    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert labels to one-hot encoded format
    """label_encoder = LabelBinarizer()
    labels = label_encoder.fit_transform(labels)"""
    
    X_train, X_test, y_train, y_test = train_test_split(
                                images, labels, test_size=0.2, stratify=labels, random_state=42)
    
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return  X_train, X_test, y_train, y_test, mat_pd_imagenet, classes, concepts, label_to_idx_imagenet



def load_CIFAR_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0


    with open('../data/CIFAR100/cifar100_fine_labels.txt') as f:
        targets_list = f.read().splitlines()
        
    
    #get classes that are common to both AwA and CIFAR and keep their CIFAR indexes in ind_list
    targets = []
    ind = 0
    ind_list = []
    ind_list_2 = []
    for t in targets_list:
        if t in classes:
            targets.append(t)
            ind_list.append(ind)
            ind_list_2.append(targets_list.index(t))
        ind +=1    
        
    #Making dict to map CIFAR class index to class name
    idx_to_label_cifar = {}
    j = 0
    for i in ind_list:
        idx_to_label_cifar[i] = targets[j]
        j+=1


    #Get CIFAR index of images that are in desired classes
    y_train_idx, y_test_idx = [], []   
    for i in ind_list:
        y_train_idx.append(np.where(train_labels == i)[0])
        y_test_idx.append(np.where(test_labels == i)[0])

    #Flattening the above result to a 17*500 list
    i_train_idx_c, i_test_idx_c = [], []
    for i in range(len(y_train_idx)):
        for j in range(np.shape(y_train_idx[i])[0]):
            i_train_idx_c.append(y_train_idx[i][j])
    for i in range(len(y_test_idx)):
        for j in range(np.shape(y_test_idx[i])[0]):
            i_test_idx_c.append(y_test_idx[i][j])


    #Keeping images and CIFAR labels of the desired classes
    X_train, X_test = train_images[i_train_idx_c], test_images[i_test_idx_c]
    y_train_0, y_test_0 = train_labels[i_train_idx_c], test_labels[i_test_idx_c]

    
    #Get AwA matrix by keeping desired classes as columns 
    mat_pd_cifar = mat_pd
    for t in mat_pd.columns:
        if t.lower() not in targets:
            mat_pd_cifar = mat_pd_cifar.drop(t.lower(), axis=1)
            
    #Make a dict to map AwA class names to their index in the matrix columns
    label_to_idx_awa = {}
    for t in mat_pd_cifar.columns:
        label_to_idx_awa[t] = list(mat_pd_cifar.columns).index(t)

    #weights_matrix = np.array(mat_pd/100)

    #Label images with AwA indexes
    y_train, y_test = np.array([[0]]*len(y_train_0)), np.array([[0]]*len(y_test_0))
    for i in range(len(y_train)):
        y_train[i][0] = label_to_idx_awa[idx_to_label_cifar[y_train_0[i][0]]]
    for i in range(len(y_test)):
        y_test[i][0] = label_to_idx_awa[idx_to_label_cifar[y_test_0[i][0]]]
        
        
    attr_label_train = []
    for i in range(len(y_train)):   
        attr_label_train.append((mat_pd_cifar.iloc[:,y_train[i]].values/100).tolist())
        
    attr_label_test = []
    for i in range(len(y_test)):   
        attr_label_test.append((mat_pd_cifar.iloc[:,y_test[i]].values/100).tolist())
    
    a_train = np.array(attr_label_train).squeeze()
    a_test = np.array(attr_label_test).squeeze()
     
    
    mat_pd_cifar = mat_pd_cifar.replace(0, 0.1)
    mat_pd_cifar = 100. / mat_pd_cifar.to_numpy()
    mat_pd_cifar = (mat_pd_cifar-np.min(mat_pd_cifar))/(np.max(mat_pd_cifar)-np.min(mat_pd_cifar))
    
    
    return X_train, X_test, y_train, y_test, a_train, a_test, mat_pd_cifar, targets, concepts , label_to_idx_awa, None




def load_CIFAR_data_attr():
    
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0


    with open('../data/CIFAR100/cifar100_fine_labels.txt') as f:
        targets_list = f.read().splitlines()
        
    
    
    #get classes that are common to both AwA and CIFAR and keep their CIFAR indexes in ind_list
    targets = []
    ind = 0
    ind_list = []
    ind_list_2 = []
    for t in targets_list:
        if t in classes:
            targets.append(t)
            ind_list.append(ind)
            ind_list_2.append(targets_list.index(t))
        ind +=1    
        
    #Making dict to map CIFAR class index to class name
    idx_to_label_cifar = {}
    j = 0
    for i in ind_list:
        idx_to_label_cifar[i] = targets[j]
        j+=1


    #Get CIFAR index of images that are in desired classes
    y_train_idx, y_test_idx = [], []   
    for i in ind_list:
        y_train_idx.append(np.where(train_labels == i)[0])
        y_test_idx.append(np.where(test_labels == i)[0])

    #Flattening the above result to a 17*500 list
    i_train_idx_c, i_test_idx_c = [], []
    for i in range(len(y_train_idx)):
        for j in range(np.shape(y_train_idx[i])[0]):
            i_train_idx_c.append(y_train_idx[i][j])
    for i in range(len(y_test_idx)):
        for j in range(np.shape(y_test_idx[i])[0]):
            i_test_idx_c.append(y_test_idx[i][j])


    #Keeping images and CIFAR labels of the desired classes
    X_train, X_test = train_images[i_train_idx_c], test_images[i_test_idx_c]
    y_train_0, y_test_0 = train_labels[i_train_idx_c], test_labels[i_test_idx_c]

    
    #Get AwA matrix by keeping desired classes as columns 
    mat_pd_cifar = mat_pd
    for t in mat_pd.columns:
        if t.lower() not in targets:
            mat_pd_cifar = mat_pd_cifar.drop(t.lower(), axis=1)
    concepts_list = [c.split('-')[0] for c in concepts]
    #Keeping lines that are Broden concepts
    for c in mat_pd_cifar.index:
        if c.lower() not in concepts_list:
            mat_pd_cifar = mat_pd_cifar.drop(c.lower(), axis=0)
            
    #Make a dict to map AwA class names to their index in the matrix columns
    label_to_idx_awa = {}
    for t in mat_pd_cifar.columns:
        label_to_idx_awa[t] = list(mat_pd_cifar.columns).index(t)

    #weights_matrix = np.array(mat_pd/100)

    #Label images with AwA indexes
    y_train, y_test = np.array([[0]]*len(y_train_0)), np.array([[0]]*len(y_test_0))
    for i in range(len(y_train)):
        y_train[i][0] = label_to_idx_awa[idx_to_label_cifar[y_train_0[i][0]]]
    for i in range(len(y_test)):
        y_test[i][0] = label_to_idx_awa[idx_to_label_cifar[y_test_0[i][0]]]
        
    
    samples_per_class = 50
    selected_indices = []
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    
    for c in unique_classes:
        class_indices = np.where(y_train == c)[0]
        sampled_indices = class_indices[:samples_per_class]
        if c == 0:
            sampled_indices = class_indices[:25]
        selected_indices.extend(sampled_indices)
    
    selected_indices = np.array(selected_indices)    
    X_train = X_train[selected_indices]
    y_train = y_train[selected_indices]
    
    
        
    attr_label_train = []
    for i in range(len(y_train)):   
        attr_label_train.append((mat_pd_cifar.iloc[:,y_train[i]].values/100).tolist())
        
    attr_label_test = []
    for i in range(len(y_test)):   
        attr_label_test.append((mat_pd_cifar.iloc[:,y_test[i]].values/100).tolist())
    
    a_train = np.array(attr_label_train).squeeze()
    a_test = np.array(attr_label_test).squeeze()
     
    
    mat_pd_cifar = mat_pd_cifar.replace(0, 0.1)
    mat_pd_cifar = 100. / mat_pd_cifar.to_numpy()
    mat_pd_cifar = (mat_pd_cifar-np.min(mat_pd_cifar))/(np.max(mat_pd_cifar)-np.min(mat_pd_cifar))
    
    
    return X_train, X_test, y_train, y_test, a_train, a_test, mat_pd_cifar, targets, concepts , label_to_idx_awa, None


   
    
        
    
    


def load_AwA_data():
    #Make a dict to map AwA class names to their index in the matrix columns
    label_to_idx_awa = {}
    for t in mat_pd.columns:
        label_to_idx_awa[t] = list(mat_pd.columns).index(t)
    
    dataset_dir = '../data/AwA/JPEGImages'
    # Get the list of classes
    classes = sorted(os.listdir(dataset_dir))  # Sort classes to ensure consistent order
    
    # Create a dictionary mapping class names to integer labels
    class_to_label = {cls: idx for idx, cls in enumerate(classes)}
    
    # Prepare lists to hold image data and their corresponding class labels
    images = []
    labels = []
    
    # Load images and labels
    for c in classes:
        counter = 0
        class_dir = os.path.join(dataset_dir, c)
        for file in os.listdir(class_dir):
            if counter < 1000 :
                counter += 1
                img_path = os.path.join(class_dir, file)
                img = load_img(img_path, target_size=(224, 224))  # Load image and resize to target size
                img_array = img_to_array(img)  # Convert image to numpy array
                images.append(img_array)
                labels.append(label_to_idx_awa[c])  # Use integer label

    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Convert labels to one-hot encoded format
    """label_encoder = LabelBinarizer()
    labels = label_encoder.fit_transform(labels)"""
    
    X_train, X_test, y_train, y_test = train_test_split(
                                images, labels, test_size=0.2, stratify=labels, random_state=42)
    
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return  X_train, X_test, y_train, y_test, mat_pd, classes, concepts, label_to_idx_awa


def load_SUN_data():
    data_dir = '../data/SUN/'
    imgs_dir = data_dir+'classes/'
    
    
    
    attr_list = scipy.io.loadmat(data_dir+'/attributes.mat')['attributes']

    concepts = []
    for i in range(len(list(attr_list))):
        concepts.append(attr_list[i][0][0])
    
    label_to_idx_SUN = {}
    classes = list(np.load(data_dir+'classes_list.npy'))
    for t in classes:
        label_to_idx_SUN[t] = classes.index(t)
    
    
    images, labels = [], []
    for file in os.listdir(imgs_dir):
        img_path = os.path.join(imgs_dir, file)
        img = load_img(img_path, target_size=(150, 150))  # Load image and resize to target size
        img_array = img_to_array(img)  # Convert image to numpy array
        images.append(img_array)
        labels.append(label_to_idx_SUN[file.split('+')[1].split('.')[0]])
        
    images = np.array(images)
    labels = np.array(labels)
    
   
    indices = np.arange(len(images))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                                images, labels, indices, test_size=0.2, stratify=labels, random_state=42)
    
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    
    annotations_mat = scipy.io.loadmat(data_dir+'/attributeLabels_continuous.mat')['labels_cv']
    att_mat = pd.DataFrame(annotations_mat,columns=concepts)
    att_mat_train = att_mat.loc[np.sort(idx_train)]
    att_mat_test = att_mat.loc[np.sort(idx_test)]


    return  X_train, X_test, y_train, y_test, att_mat_train, att_mat_test, classes, concepts, label_to_idx_SUN