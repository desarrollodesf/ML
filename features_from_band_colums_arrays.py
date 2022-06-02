import numpy as np


class SpectralFeatures:
    def __init__(self, data):
        self.data = data
    # predictors_names = ['COASTAL', 'B', 'G', 'Y', 'R', 'R_EDGE', 'NIR1', 'NIR2', 'AGL']
    
    # check: this functions add the result directly to the input
    # does not have return, just add columns to input data
    #self.data['NDVI_NIR1']
    
    def ndvi(self):
        # NDVI = ((NIR - Red)/(NIR + Red))
        self.data['NDVI_NIR1'] = np.divide((self.data['NIR1'] - self.data['R']), (self.data['NIR1'] + self.data['R']))
        self.data['NDVI_NIR2'] = np.divide((self.data['NIR2'] - self.data['R']), (self.data['NIR2'] + self.data['R']))

    def savi(self, l=0.5):
        # Soil Adjusted Vegetation Index (SAVI)
        # SAVI = ((NIR - Red) / (NIR + Red + L)) x (1 + L)
        self.data['SAVI_NIR1'] = (1 + l) * np.divide((self.data['NIR1'] - self.data['R']),
                                                     (self.data['NIR1'] + self.data['R'] + l))
        self.data['SAVI_NIR2'] = (1 + l) * np.divide((self.data['NIR2'] - self.data['R']),
                                                     (self.data['NIR2'] + self.data['R'] + l))

    def vari(self):
        # Visible Atmospherically Resistant Index (VARI)
        # VARI = (Green - Red)/ (Green + Red - Blue)
        self.data['VARI'] = np.divide((self.data['G'] - self.data['R']),
                                      (self.data['G'] + self.data['R'] + self.data['B']))

    def mndwi(self):
        # Modified Normalized Difference Water Index (MNDWI)
        # MNDWI = (Green - SWIR) / (Green + SWIR) # Note: using NIR2 instead of SWIR
        self.data['MNDWI'] = np.divide((self.data['G'] - self.data['NIR2']),
                                       (self.data['G'] + self.data['NIR2']))

    def ndmi(self):
        # Normalized Difference Moisture Index (NDMI)
        # Same as Normilized Difference Water Index (NDWI)
        # NDMI = (NIR - SWIR1)/(NIR + SWIR1)
        self.data['NDWI1'] = np.divide((self.data['NIR1'] - self.data['NIR2']),
                                       (self.data['NIR1'] + self.data['NIR2']))
        # If SWIR band is not available
        # NDMI = (G - NIR)/(GREEN + NIR)
        self.data['NDWI2'] = np.divide((self.data['G'] - self.data['NIR1']),
                                       (self.data['G'] + self.data['NIR1']))
        self.data['NDWI3'] = np.divide((self.data['G'] - self.data['NIR2']),
                                       (self.data['G'] + self.data['NIR2']))        
    # Geology Indices
    def cmr(self):
        # Clay Minerals Ratio = SWIR1 / SWIR2
        self.data['CMR'] = None  # np.divide(self.data[''], self.data[''])

    def fmr(self):
        # Ferrous Minerals
        # Ferrous Minerals Ratio = SWIR / NIR # Note: using NIR2 istead of SWIR
        self.data['FMR'] = np.divide(self.data['NIR2'], self.data['NIR1'])

