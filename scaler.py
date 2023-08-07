import numpy as np

class CustomScaler():
    
    def __init__(self):
        self.CHANNELS = 16
        self.means = np.zeros(self.CHANNELS)
        self.stds = np.zeros(self.CHANNELS)
    
    def fit(self, data):
        for channel in range(self.CHANNELS):
            self.means[channel] = np.mean(data[:,:,channel], axis=(0,1))
            self.stds[channel] = np.std(data[:,:,channel], axis=(0,1))
            
        
    def transform(self, data):
        new_data = data.copy()
        for channel in range(self.CHANNELS):
            new_data[:,:,channel] = (new_data[:,:,channel] - self.means[channel]) / self.stds[channel]
        return new_data
    
    def fit_and_transform(self, data):
        self.fit(data)
        return self.transform(data)
        
        
    def inverse_transform(self, data):
        new_data = data.copy()
        for channel in range(self.CHANNELS):
            new_data[:,:,channel] = self.stds[channel] * new_data[:,:,channel] + self.means[channel]
        return new_data

