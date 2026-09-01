import numpy as np
import pandas as pd
import csv
import os
import pickle
from PIL import Image

from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array



   
def load_CIFAR_data_apy(unbalanced, config):
    
    data_dir = config.get("data_dir")
    apy_dir = data_dir+'/aPY/attribute_data' 
    
    apascal_mat_train = pd.read_csv(apy_dir+'/apascal_train.txt',sep=' ',header=None,index_col=0)
    apascal_mat_test = pd.read_csv(apy_dir+'/apascal_test.txt',sep=' ',header=None,index_col=0)
    ayahoo_mat_test = pd.read_csv(apy_dir+'/ayahoo_test.txt',sep=' ',header=None,index_col=0)
    
    apy_mat = pd.concat([apascal_mat_train, apascal_mat_test, ayahoo_mat_test])
    
    apy_mat = apy_mat.rename(columns={0:'image',1:'class',2:'xmin',3:'ymin',4:'xmax',5:'ymax'})
    
    attributes = pd.read_csv(apy_dir+'/attribute_names.txt',sep='\t',header=None)
    
    for i in range(len(attributes)):
        apy_mat = apy_mat.rename(columns={i+6:attributes[0][i]})
    apy_mat = apy_mat.drop(['xmax','xmin','ymax','ymin'], axis=1)    
    classes = pd.read_csv(apy_dir+'/class_names.txt',sep='\t',header=None).to_numpy().squeeze()



    (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0


    with open(data_dir+'/CIFAR100/cifar100_fine_labels.txt') as f:
        targets_list = f.read().splitlines()
        
        
    person_group = ['baby', 'boy', 'girl', 'man', 'woman']
    apy_to_cifar100 = {"bicycle": "bicycle",
                        "bottle": "bottle",
                        "bus": "bus",
                        "car": "streetcar",
                        "chair": "chair",
                        "diningtable": "table",
                        "motorbike": "motorcycle",
                        "sofa": "couch",
                        "train": "train",
                        "tvmonitor": "television",
                        "wolf": "wolf",
                        "mug": "cup",
                        "person": person_group}  
    
    # Flatten CIFAR classes to include all in person_group
    target_class_names = []
    for v in apy_to_cifar100.values():
        if isinstance(v, list):
            target_class_names.extend(v)
        else:
            target_class_names.append(v)

    # Find CIFAR indices of all target classes
    target_indices = [targets_list.index(cls) for cls in target_class_names]

    # Filter CIFAR data to include only these classes
    train_mask = np.isin(train_labels.flatten(), target_indices)
    test_mask = np.isin(test_labels.flatten(), target_indices)

    X_train = train_images[train_mask]
    filtered_train_labels = train_labels[train_mask]

    X_test = test_images[test_mask]
    filtered_test_labels = test_labels[test_mask]

    # Prepare attribute matrix
    mat_pd_cifar = apy_mat[apy_mat['class'].isin(['person'] + list(apy_to_cifar100.keys()))]
    mat_pd_cifar = mat_pd_cifar.groupby('class').mean().transpose()
    mat_pd_apy = mat_pd_cifar.applymap(lambda x: 1. / x if x != 0 else 0)
    mat_pd_apy = (mat_pd_apy - np.min(mat_pd_apy)) / (np.max(mat_pd_apy) - np.min(mat_pd_apy))

    # Assign separate integer labels to CIFAR classes
    label_to_idx_cifar = {name: idx for idx, name in enumerate(target_class_names)}
    idx_to_label_cifar = {v: k for k, v in label_to_idx_cifar.items()}

    # Assign CIFAR-style integer labels
    y_train = np.array([label_to_idx_cifar[targets_list[label[0]]] for label in filtered_train_labels]).reshape(-1, 1)
    y_test = np.array([label_to_idx_cifar[targets_list[label[0]]] for label in filtered_test_labels]).reshape(-1, 1)

    # All person-group classes map to the same aPY class: 'person'
    label_to_idx_apy = {k: i for i, k in enumerate(mat_pd_cifar.columns)}
    cifar_to_apy = {}
    for apy_class, cifar_class in apy_to_cifar100.items():
        if isinstance(cifar_class, list):
            for c in cifar_class:
                cifar_to_apy[c] = apy_class
        else:
            cifar_to_apy[cifar_class] = apy_class

    # Attribute labels based on CIFAR → aPY mapping
    attr_label_train = [mat_pd_cifar[cifar_to_apy[targets_list[label[0]]]].values for label in filtered_train_labels]
    attr_label_test = [mat_pd_cifar[cifar_to_apy[targets_list[label[0]]]].values for label in filtered_test_labels]

    a_train = np.array(attr_label_train)
    a_test = np.array(attr_label_test)
    
    return X_train, X_test, y_train, y_test, a_train, a_test, mat_pd_apy.to_numpy(), target_class_names, list(mat_pd_cifar.index), label_to_idx_apy, mat_pd_cifar






def load_aPY_data(cv, data_dir, config):
    unbalanced = config.get("unbalanced")
    
    apy_dir = data_dir+'/aPY/attribute_data'   #config.get("data_dir")+'/aPY/attribute_data'   
    apascal_mat_train = pd.read_csv(apy_dir+'/apascal_train.txt',sep=' ',header=None,index_col=0)
    apascal_mat_test = pd.read_csv(apy_dir+'/apascal_test.txt',sep=' ',header=None,index_col=0)
    ayahoo_mat_test = pd.read_csv(apy_dir+'/ayahoo_test.txt',sep=' ',header=None,index_col=0)
    
    apy_mat = pd.concat([apascal_mat_train, apascal_mat_test, ayahoo_mat_test])
    
    apy_mat = apy_mat.rename(columns={0:'image',1:'class',2:'xmin',3:'ymin',4:'xmax',5:'ymax'})
    apy_mat_not_numeric = apy_mat.copy()
    

    # Convert all columns except 'class' to numeric
    for col in apy_mat.columns:
        if col != 'class':
            apy_mat[col] = pd.to_numeric(apy_mat[col], errors='coerce')
    
    attributes = pd.read_csv(apy_dir+'/attribute_names.txt',sep='\t',header=None)
    
    for i in range(len(attributes)):
        apy_mat = apy_mat.rename(columns={i+6:attributes[0][i]})
    
    classes = pd.read_csv(apy_dir+'/class_names.txt',sep='\t',header=None).to_numpy().squeeze()
    
    
    ##### Remove objects without any concept score
    no_attributes_mask = (apy_mat.drop(['class','xmax','xmin','ymax','ymin'], axis=1).fillna(0) == 0).all(axis=1)
    apy_mat_cleaned = apy_mat[~no_attributes_mask]
    print(f"Number of rows removed: {no_attributes_mask.sum()}")

    apy_mat = apy_mat_cleaned
    
    
    
    if config.get("to_crop"):
        #### Crop images to have one image-class couple
        output_dir = apy_dir+'/../cropped_images'
        os.makedirs(output_dir, exist_ok=True)
        apascal_images_dir = apy_dir + "/../apascal_images/JPEGImages"
        ayahoo_images_dir = apy_dir+"/../ayahoo_test_images"
        
        cropped_images_info = []
        skipped_images_count = 0
        cropped_images_count = 0
        cropped_images_tracker = {}
    
        
        # Loop over each image in apy_mat 
        for _, row in apy_mat.iterrows():
            file = row.name  # The file name is the index of the DataFrame
            c = row['class']  # Class of the current image-bounding boxes
            
            # Get the bounding box coordinates (xmin, ymin, xmax, ymax)
            xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']].values
            
            # Check which image directory the file belongs to
            if file in os.listdir(apascal_images_dir):
                img_path = os.path.join(apascal_images_dir, file)
            elif file in os.listdir(ayahoo_images_dir):
                img_path = os.path.join(ayahoo_images_dir, file)
            else:
                skipped_images_count += 1
                print(f"Image {file} not found in any directory.")
                continue  
            
            # Open the image
            img = Image.open(img_path)
            
            # Check if the bounding box is valid (non-zero area) and crop the image using the bounding box
            if xmin >= xmax or ymin >= ymax:
                skipped_images_count += 1
                print(f"Invalid bounding box for image {file}. Skipping...")
                cropped_img = img        
            else:
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
            
            #cropped_img = cropped_img.resize((224, 224))
        
            # Create a directory for the class if it doesn't exist
            class_output_dir = os.path.join(output_dir, c)
            os.makedirs(class_output_dir, exist_ok=True)
        
            # Save the cropped image with a new name based on the file name and class
            if (file, c) not in cropped_images_tracker:
                cropped_images_tracker[(file, c)] = 1
                cropped_img_name = f"{os.path.splitext(file)[0]}_{c}.jpg"
            else:
                cropped_images_tracker[(file, c)] += 1
                cropped_img_name = f"{os.path.splitext(file)[0]}_{c}_{cropped_images_tracker[(file, c)]}.jpg"
        
            cropped_img_path = os.path.join(class_output_dir, cropped_img_name)
            cropped_img.save(cropped_img_path)
        
            # Add the cropped image info to the new DataFrame
            cropped_images_info.append({
                'original_image': file,
                'cropped_image': cropped_img_name,
                'class': c
            })
            cropped_images_count += 1
            #print(f"Cropped image saved: {cropped_img_path}")
        
        # Create a new DataFrame for the cropped images
        cropped_images_df = pd.DataFrame(cropped_images_info)
        
        # Save the new DataFrame to a CSV file (optional)
        cropped_images_df.to_csv(apy_dir+"/../cropped_images_info.csv", index=True)
        
        print("Cropped images DataFrame saved.")
        
    else:
        cropped_images_df = pd.read_csv(apy_dir+"/../cropped_images_info.csv",)
    
    
    apy_mat['cropped_image'] = cropped_images_df['cropped_image'].values
    
    cropped_images_df = apy_mat
    
    apy_mat = apy_mat.drop(['xmax','xmin','ymax','ymin','cropped_image'], axis=1)    
    
    
    """index_counts = apy_mat.index.value_counts()
    unique_indexes = index_counts[index_counts == 1].index
    df_unique_indexes = apy_mat.loc[unique_indexes]
    
    class_distribution = df_unique_indexes['class'].value_counts()
    classes_to_keep = class_distribution[class_distribution > 100].index"""
    
    #Make class-attribute matrix by averaging across class column
    mat_pd_apy = apy_mat.groupby('class').mean().transpose()
    mat_GT = mat_pd_apy
    mat_pd_apy = mat_pd_apy.map(lambda x: 1. / x if x != 0 else 0)
    mat_pd_apy = (mat_pd_apy-np.min(mat_pd_apy))/(np.max(mat_pd_apy)-np.min(mat_pd_apy))
    
    
    #Make a dict to map AwA class names to their index in the matrix columns
    label_to_idx_awa = {}
    for t in mat_pd_apy.columns:
        label_to_idx_awa[t] = list(mat_pd_apy.columns).index(t)
    
    

    images = []
    labels = []
    annotations = []
    
    
    if unbalanced:
        for c in np.unique(classes):
            counter = 0
            class_dir = os.path.join(config.get("data_dir")+'/aPY/_targets', c)
            for file in os.listdir(class_dir):
                if counter < 1000 and file in df_unique_indexes.index:
                    counter += 1
                    img_path = os.path.join(class_dir, file)
                    img = load_img(img_path, target_size=(224, 224))  # Load image and resize to target size
                    img_array = img_to_array(img)  # Convert image to numpy array
                    images.append(img_array)
                    try:
                        labels.append(label_to_idx_awa[c])  # Use integer label
                        ann = apascal_mat.loc[(apascal_mat.index == file) & (apascal_mat['class'] == c)]
                        ann = ann.drop(['class''xmax','xmin','ymax','ymin'],axis=1).values[0]
                        annotations.append(ann)
                    except:
                        print('image ',file, ' not in matrix')
    
    else:
        for c in np.unique(classes):
            counter = 0
            class_dir = os.path.join(data_dir+'/aPY/cropped_images', c)
            for file in os.listdir(class_dir):
                
                if config.get("n_samples_per_class") is not None:
                    if counter < config.get("n_samples_per_class") and file in cropped_images_df['cropped_image'].values:
                        counter += 1
                        img_path = os.path.join(class_dir, file)
                        img = load_img(img_path, target_size=(config.get("image_size"), config.get("image_size")))  # Load image and resize to target size
                        img_array = img_to_array(img)  # Convert image to numpy array
                        images.append(img_array)
                        try:
                            labels.append(label_to_idx_awa[c])  # Use integer label
                            ann = cropped_images_df.loc[(cropped_images_df['cropped_image'] == file) & (cropped_images_df['class'] == c)]
                            ann = ann.drop(['class','xmax','xmin','ymax','ymin','cropped_image'],axis=1).values[0]
                            """ann = mat_GT[c]
                            ann = ann.values"""
                            annotations.append(ann)
                        except:
                            print('image ',file, ' not in matrix')
                else:
                    if file in cropped_images_df['cropped_image'].values:
                        counter += 1
                        img_path = os.path.join(class_dir, file)
                        img = load_img(img_path, target_size=(config.get("image_size"), config.get("image_size")))  # Load image and resize to target size
                        img_array = img_to_array(img)  # Convert image to numpy array
                        images.append(img_array)
                        try:
                            labels.append(label_to_idx_awa[c])  # Use integer label
                            ann = cropped_images_df.loc[(cropped_images_df['cropped_image'] == file) & (cropped_images_df['class'] == c)]
                            ann = ann.drop(['class','xmax','xmin','ymax','ymin','cropped_image'],axis=1).values[0]
                            """ann = mat_GT[c]
                            ann = ann.values"""
                            annotations.append(ann)
                        except:
                            print('image ',file, ' not in matrix')
    
    
    """images = np.array(images)
    labels = np.array(labels)
    annotations = np.array(annotations)"""

    
    if cv == False:
        X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
                                    np.array(images), np.array(labels), np.array(annotations), test_size=config.get("test_split"), stratify=labels, random_state=42)
        """X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0"""
     
        return  X_train, X_test, y_train, y_test, a_train, a_test, mat_pd_apy.to_numpy(), classes, attributes.to_numpy().squeeze(), label_to_idx_awa, mat_GT


    
    else:
        return np.array(images).astype('float32'), None, np.array(labels), None, np.array(annotations), None, mat_pd_apy.to_numpy(), classes, attributes, label_to_idx_awa, mat_GT
        