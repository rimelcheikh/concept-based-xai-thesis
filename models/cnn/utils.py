import os
import json
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def set_seeds(seed=42):
    import random
    import os
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # DETERM FIX
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        print("Torch determinism not set:", e)

        
    
def numpy_from_dl(dl):
    """
    Convert a PyTorch-like DataLoader to numpy arrays (X, y, a).
    Handles (N, C, H, W) or (N, H, W, C) inputs.
    """
    X_list, y_list, a_list = [], [], []
    for batch in dl:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, y, a = batch
            else:
                x, y = batch
                a = None
        else:
            x, y, a = batch["images"], batch["labels"], batch.get("attributes")

        x = np.asarray(x)
        if x.ndim == 4 and x.shape[1] in (1, 3):  # (N,C,H,W) -> (N,H,W,C)
            x = np.transpose(x, (0, 2, 3, 1))

        X_list.append(x.astype("float32"))
        y_list.append(np.asarray(y).astype("int32"))
        if a is not None:
            a_list.append(np.asarray(a).astype("float32"))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    a = np.concatenate(a_list, axis=0) if a_list else None
    return X, y, a


def get_backbone_preprocess(backbone_name: str):
    """
    Return the correct tf.keras.applications.preprocess_input
    for the chosen backbone. If unknown, return identity.
    """
    name = backbone_name.lower()
    if name == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    if name == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    if name == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input
    if name == "inceptionv3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input

    return lambda x: x


def write_setting(save_dir, model_name, k_data, dataset, l_attr_CE, l_p_y_w, l_w, 
                  i, n_runs, optimizer, lr, batch_size, start_mse, end_mse,
                  val_split, n_epochs, N, V, T, M, K):
    
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    with open(save_dir+'/setting.txt', 'w') as f:
        f.write('CNN backbone : '+model_name+'\n')
        f.write('Knowledge data: '+k_data+'\n')
        f.write('Dataset : '+dataset+'\n')
        f.write('Weight for attributes CE loss : '+ str(l_attr_CE) +'\n')
        f.write('Weight for class score loss : '+ str(l_p_y_w) +'\n')
        f.write('Weight for class weights loss : '+ str(l_w) +'\n')
        f.write('Number of run : ' + str(i) + "/" + str(n_runs) +'\n')
        f.write('Optimizer : ' + optimizer +'\n')
        f.write('Learning rate : ' + str(lr) +'\n')
        f.write('Batch size : ' + str(batch_size) +'\n')
        f.write('Validation split : '+ str(val_split) +'\n')
        f.write('Number of epochs : ' + str(n_epochs) + '\n')
        f.write('Train/Val/Test : ' + str(N) + '/' + str(V) + '/' + str(T) + '\n')
        f.write('Number of classes : ' + str(K) + '\n')
        f.write('Number of attributes : ' + str(M) + '\n')
        f.write('Start MSE value for alpha update: ' + str(start_mse) + '\n')
        f.write('End MSE value for alpha update: ' + str(end_mse) + '\n')
        





def make_tfds(
    X, y, a,
    batch_size,
    shuffle=False,
    cache=False,
    preprocess_fn=None,
    data_augmentation=None,
    is_training=False,
    # DETERM FIX
    seed=42,
    reshuffle_each_iteration=False,
):
    ...
    ds = tf.data.Dataset.from_tensor_slices((X, y, a))

    # DETERM FIX: force deterministic dataset behavior
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    

    if cache:
        ds = ds.cache()
    if shuffle:
        # DETERM FIX: fixed seed + no reshuffle each epoch by default
        ds = ds.shuffle(min(10_000, len(y)), seed=seed, reshuffle_each_iteration=False)
        
    def _map(x, y, a):
        x = tf.cast(x, tf.float32)
        if is_training and data_augmentation is not None:
            x = data_augmentation(x, training=True)
        if preprocess_fn is not None:
            x = preprocess_fn(x)
        return x, y, a

    if (data_augmentation is not None) or (preprocess_fn is not None):
        # DETERM FIX: avoid AUTOTUNE parallel nondeterminism
        ds = ds.map(_map, num_parallel_calls=1, deterministic=True)
        

    # DETERM FIX: prefetch=1 is safest for determinism
    ds = ds.batch(batch_size).prefetch(1)
    return ds

