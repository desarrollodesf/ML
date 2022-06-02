# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:18:35 2021

@author: alexyshr
"""
import numpy as np


class SquareImagesFeatures:
    def __init__(self, raw_data):
        self.raw_data = raw_data


class SpatialFeatures(SquareImagesFeatures):
    def sf1(self):
        print("implement")

    
class GeometricFeatures(SquareImagesFeatures):
    def func_1(self):
        print("implement")
    
class SpectralFeaturesSQ(SquareImagesFeatures):
    # Principal Component Analysis
    # Note that PCA calculation receives rectangular image, but the calculation
    # is performed with a stacked by columns image.
    # The result is again a rectangular image
    def calculation_pca(self, sq_array_image, dims_rescaled_data=3): # bands, rows, cols
        """
        """
        #arr = np.arange(16).reshape((2, 2, 4))
        #arr2 = arr.transpose(1,2,0)
        #arr3 = arr2.reshape(-1, 2)
        #arr4 = arr3.reshape(arr.shape[1],arr.shape[2], arr.shape[0])
        #arr5 = arr4.transpose(2, 0, 1)
        
        # organize the image to 
        sq_array_image_mod = sq_array_image.transpose(1,2,0).reshape(-1, sq_array_image.shape[0]) #.transpose()
        from sklearn.decomposition import PCA
        #import earthpy.plot as ep
        #from matplotlib 00import pyplot as plt
        
        pca = PCA(dims_rescaled_data)
        converted_data = pca.fit_transform(sq_array_image_mod) #already in column format
        #lets return to square image
        converted_data = converted_data.reshape(sq_array_image.shape[1], sq_array_image.shape[2], dims_rescaled_data)
        converted_data = converted_data.transpose(2,0,1)
        
        #ep.plot_bands(converted_data, cmap="RdYlGn", cols=1, figsize=(10, 14))
        
        #plt.show()
        return converted_data
    
    # Independent Component Analysis
    # Note that ICA calculation receives rectangular image, but the calculation
    # is performed with a stacked by columns image.
    # The result is again a rectangular image
    def calculation_ica(self, sq_array_image, n_components=3): # bands, rows, cols
        """
        """
        #arr = np.arange(16).reshape((2, 2, 4))
        #arr2 = arr.transpose(1,2,0)
        #arr3 = arr2.reshape(-1, 2)
        #arr4 = arr3.reshape(arr.shape[1],arr.shape[2], arr.shape[0])
        #arr5 = arr4.transpose(2, 0, 1)
        
        # organize the image to 
        sq_array_image_mod = sq_array_image.transpose(1,2,0).reshape(-1, sq_array_image.shape[0]) #.transpose()
        from sklearn.decomposition import FastICA
        #import earthpy.plot as ep
        #from matplotlib import pyplot as plt
        
        ica = FastICA(n_components=n_components)
        converted_data = ica.fit_transform(sq_array_image_mod)  # Reconstruct signals

        #lets return to square image
        converted_data = converted_data.reshape(sq_array_image.shape[1], sq_array_image.shape[2], n_components)
        converted_data = converted_data.transpose(2,0,1)
        
        #ep.plot_bands(converted_data, cmap="RdYlGn", cols=1, figsize=(10, 14))
        
        #plt.show()
        return converted_data
    
    def calculation_glcm(self, sq_array_image, df_filters, array_distances=[10,20,60],
                         array_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4, 8*np.pi/4],
                         patch_size=10, number_of_bands=3):
        import math
        from skimage.feature import greycomatrix, greycoprops
        import numpy as np
        # array to store the result of type
        # the number 6 is the number of fixed texture features used:
        # contrast
        # dissimilarity
        # homogeneity
        # energy
        # correlation
        # ASM
        the_texture = np.full(shape=(int(6*number_of_bands*len(array_distances)*len(array_angles)), 
                                     sq_array_image.shape[1],
                                     sq_array_image.shape[2]),
                                     fill_value=None, dtype=None)

        # Loop bands, rows, then columns (with each cell loop the patch)
        index_output_array = 0
        for bands_idx in range(number_of_bands):
            #cell_location = [(row_idx, col_idx)]
            current_band = sq_array_image[bands_idx]
            cb_0_255 = (255*(current_band - np.min(current_band))/np.ptp(current_band)).astype(int)
            real_patch_size = patch_size
            if (patch_size % 2) == 0:
                real_patch_size += 1
            patch_border =  math.floor(real_patch_size / 2)   
            cb_0_255_padded = np.pad(cb_0_255, pad_width=patch_border, mode='edge')
            for (idx, row_idx) in enumerate(df_filters['xidx']):
            #for row_idx in range(sq_array_image.shape[1]):
                col_idx = df_filters['yidx'].iloc[idx]
                #for col_idx in range(sq_array_image.shape[2]):            
                index_to_write = index_output_array
                patch_image = cb_0_255_padded[row_idx:row_idx+real_patch_size, col_idx:col_idx+real_patch_size]
                for distance in array_distances:
                    for angle in array_angles:
                        glcm = greycomatrix(patch_image, levels=256, distances=[distance], angles=[angle])                            
                        # contrast
                        pixel_value = greycoprops(glcm, 'contrast')[0, 0]
                        the_texture[index_to_write, row_idx, col_idx] = pixel_value
                        # dissimilarity
                        pixel_value = greycoprops(glcm, 'dissimilarity')[0, 0]
                        the_texture[index_to_write+1, row_idx, col_idx] = pixel_value
                        # homogeneity
                        pixel_value = greycoprops(glcm, 'homogeneity')[0, 0]
                        the_texture[index_to_write+2, row_idx, col_idx] = pixel_value
                        # energy
                        pixel_value = greycoprops(glcm, 'energy')[0, 0]
                        the_texture[index_to_write+3, row_idx, col_idx] = pixel_value
                        # correlation
                        pixel_value = greycoprops(glcm, 'correlation')[0, 0]
                        the_texture[index_to_write+4, row_idx, col_idx] = pixel_value
                        # ASM
                        pixel_value = greycoprops(glcm, 'ASM')[0, 0]
                        the_texture[index_to_write+5, row_idx, col_idx] = pixel_value  
                    index_to_write += 6
            index_output_array = index_to_write
        return the_texture
                            
                                        
            
    def pca(self):
        self.raw_data['PC'] = np.nan
        self.raw_data['PC'] = self.raw_data['PC'].astype(object)
        
        #self.raw_data['PC2'] = np.nan
        #self.raw_data['PC2'] = self.raw_data['PC2'].astype(object)
        
        #self.raw_data['PC3'] = np.nan
        #self.raw_data['PC3'] = self.raw_data['PC3'].astype(object)
        
        for row_index in range(len(self.raw_data)):
            array_msi = self.raw_data['MSI'].iloc[row_index]
            pc = self.calculation_pca(sq_array_image = array_msi, dims_rescaled_data=3)
            #self.raw_data[row_index, 'PC1']= pc[0]
            self.raw_data.at[row_index, 'PC'] = pc
            #self.raw_data.at[row_index, 'PC2'] = pc[1]
            #self.raw_data.at[row_index, 'PC3'] = pc[2]
            
    def ica(self):
        self.raw_data['ICA'] = np.nan
        self.raw_data['ICA'] = self.raw_data['ICA'].astype(object)
        
        
        for row_index in range(len(self.raw_data)):
            array_msi = self.raw_data['MSI'].iloc[row_index]
            ica = self.calculation_ica(sq_array_image = array_msi, n_components=3)
            #self.raw_data[row_index, 'PC1']= pc[0]
            self.raw_data.at[row_index, 'ICA'] = ica

                        
    def glcm(self, in_col_name="MSI", 
             array_distances=[10,20,60],
             array_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4, 8*np.pi/4],
             patch_size=10, 
             number_of_bands=3,
             out_col_name="GLCM"):
        # Note: by default the system calculates GLSM in the first number_of_bands
        # of the input array (be sure to pass an appropiate value, i.e. less than
        # number of bands)
        ## type: contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'
                
        self.raw_data[out_col_name] = np.nan
        self.raw_data[out_col_name] = self.raw_data[out_col_name].astype(object)
        
        
        for row_index in range(len(self.raw_data)):
            the_array = self.raw_data[in_col_name].iloc[row_index]
            df_filters = self.raw_data["RND_IDX"].iloc[row_index]
            arr = self.calculation_glcm(sq_array_image = the_array,
                                        df_filters=df_filters,
                                        array_distances=array_distances,
                                        array_angles=array_angles,
                                        patch_size=patch_size,
                                        number_of_bands=number_of_bands)
            self.raw_data.at[row_index, out_col_name] = arr

    def calculation_morpho(self, sq_array_image, disk_size=6):
        #Morphology techniques to use
        from skimage.morphology import erosion, dilation, opening, closing, \
                                       white_tophat, black_tophat, skeletonize, \
                                       convex_hull, convex_hull_image
        # Importing 'square' and 'disk' modules for creating 
        # BINARY structuring elements
        # from skimage.morphology import square as sq
        from skimage.morphology import disk
        
        # 'img_as_ubyte()', can be converted to type : 
        # *numpy.ndarray* with *uint8* element. 
        # This is the input array for the morphological functions
        #from skimage.util import img_as_ubyte
        # sqa_array_uint8 = img_as_ubyte(sq_array_image)
        sqa_array_uint8 = (255*(sq_array_image - np.min(sq_array_image))/
                           np.ptp(sq_array_image)).astype(int)
        
        list_of_images = []
        selem = disk(disk_size)
        eroded = erosion(sqa_array_uint8, selem)
        list_of_images.append(eroded)
        dilate = dilation(sqa_array_uint8, selem)
        list_of_images.append(dilate)
        opened = opening(sqa_array_uint8, selem)
        list_of_images.append(opened)
        closed = closing(sqa_array_uint8, selem)
        list_of_images.append(closed)
        w_tophat = white_tophat(sqa_array_uint8, selem)
        list_of_images.append(w_tophat)
        b_tophat = black_tophat(sqa_array_uint8, selem)
        list_of_images.append(b_tophat)
        # skeleton = skeletonize(sqa_array_uint8)
        # hull2 = convex_hull_image(sqa_array_uint8)
        return np.stack(list_of_images)
        # arrays = []
        # arrays.append
        # stacked_array = np.stack(arrays)
        # output_array = numpy.zeros((14,4,3), dtype=np.float32)
        # for i in range(14):
        #     mat = numpy.random.rand(4,3)
        #     output_array[i] = mat
            
            
    def morpho(self, disk_size=3):
        ## type: 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'
        self.raw_data["MORPHO"] = np.nan
        self.raw_data["MORPHO"] = self.raw_data["MORPHO"].astype(object)

        for row_index in range(len(self.raw_data)):
            the_array = self.raw_data["MSI"].iloc[row_index]
            arr = self.calculation_morpho(sq_array_image = the_array,
                                        disk_size=disk_size)
            self.raw_data.at[row_index, "MORPHO"] = arr
            
            
            
            
            
            
            
            