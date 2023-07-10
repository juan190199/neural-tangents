import numpy as np


def process_data(data_chunk, to_continuous=False):
    """
    Flatten the images and one-hot encode the labels.
    Args:
        data_chunk:

    Returns:

    """
    image, label = data_chunk['image'], data_chunk['label']

    samples = image.shape[0]
    image = np.array(np.reshape(image, (samples, -1)), dtype=np.float32)
    image = (image - np.mean(image)) / np.std(image)

    if to_continuous:
        label = label/np.max(label)
        label = (label - np.mean(label)) / np.std(label)
    else:
        label = np.eye(10)[label]

    return {'image': image, 'label': label}