from tensorflow import keras
import numpy as np
import loader

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.images = dataset['images']
        self.depth = dataset['depth']
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index+1)*self.batch_size]
        
        flip = np.random.choice([True, False])
        
        images = [self.images[k] for k in indexes]
        depth = [self.depth[k] for k in indexes]
        
        p_img = self.__preprocess_images(images, flip)
        p_depth = self.__preprocess_depth(depth, flip)
        
        return p_img, p_depth
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __preprocess_images(self, images, flip):
        result = []
        for img in images:
            image = loader.load_color(img, flip)
            scaled_img = (image - image.min()) / (image.max() - image.min())
            result.append(scaled_img)
        return np.array(result)
    
    def __preprocess_depth(self, images, flip):
        result = []
        for img in images:
            image = loader.load_depth(img, flip)
            scaled_img = (image - image.min()) / (image.max() - image.min())
            result.append(scaled_img)
        return np.array(result)