import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
import random
import pandas as pd 
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
#import cv2

class GeomapGenerator():
    def __init__(self, dfname='dataset.csv'):
        self.dataframe = pd.DataFrame(columns=['Name'])
        self.dfname = dfname

    def generate_dataset(self, NSAMPLES, IMGW, IMGH, ISOCOLOR, GEOCOLOR):
        for sample in range(NSAMPLES):
            topogeomap, binary_isomap, binary_outlines = self.generate_image(IMGW, IMGH, ISOCOLOR, GEOCOLOR)
            topogeomap, binary_isomap, binary_outlines = self.augmentate(topogeomap, binary_isomap, binary_outlines)
            name = str(round(time.time()))

            self.dataframe = self.dataframe.append({'Name': name}, ignore_index=True)
            plt.imsave('images/' + name + '.png', np.uint8(topogeomap))
            #cv2.imwrite('images/' + name + '.png', topogeomap)
            np.save('isolines/' + name, binary_isomap)
            np.save('geolines/' + name, binary_isomap)

        self.dataframe.to_csv(dfname, index=False)



    def augmentate(self, topogeomap, binary_isomap, binary_outlines):
        return topogeomap, binary_isomap, binary_outlines #PASS


    def binarize_colors(self, image, binvalue=255):
        width, height = image.shape
        for x in range(width):
            for y in range(height):
                if image[x][y] != 0:
                    image[x][y] = binvalue
        return image

    def change_binarized_colors(self, image, color):
        width, height = image.shape[0:2]
        new_image = np.zeros((width, height, 3)).astype('int')
        for x in range(width):
            for y in range(height):
                if image[x][y][0] == 255 and image[x][y][1] == 255 and image[x][y][2] == 255:
                    new_image[x][y][0] = color[0]
                    new_image[x][y][1] = color[1]
                    new_image[x][y][2] = color[2]
        return new_image

    def create_channels(self, image):
        width, height = image.shape
        new_image = np.zeros((width, height, 3)).astype('int')
        for x in range(width):
            for y in range(height):
                new_image[x][y][0] = image[x][y]
                new_image[x][y][1] = image[x][y]
                new_image[x][y][2] = image[x][y]
        return new_image    

    def get_outline(self, image, color):
        width, height = image.shape
        outline_image = np.zeros((width, height))
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if (image[x][y] == color) and (image[x - 1][y] != color or image[x][y - 1] != color or image[x + 1][y] != color or image[x][y + 1] != color):
                    outline_image[x][y] = 1
                else:
                    outline_image[x][y] = 0
        return outline_image

    def generate_image(self, IMGW, IMGH, ISOCOLOR, GEOCOLOR):
        #Step 0: Generate noise
        heightmap = np.random.random_sample((IMGW, IMGH))

        #Step 1: Blur noise
        heightmap = gaussian_filter(heightmap, 32)

        #Step 2: Reduce heights count
        heightmap = np.round(heightmap, 3)

        #Step 3: Create isolines
        isomap = np.zeros((heightmap.shape[0], heightmap.shape[1]))
        for color in np.unique(heightmap):
            isomap = isomap + self.get_outline(heightmap, color)

        #Step 5: Create geological regions map

        #Variant 1: Works enough good?
        random_geomap = gaussian_filter(np.random.random_sample((IMGW, IMGH)) - np.random.random_sample((IMGW, IMGH)), 96)
        heightmap = np.round(gaussian_filter(random_geomap + heightmap, 3), 3)

        #Step 6: Create outlines for geolayers (It's important!!)
        geomap_outlines = np.zeros((heightmap.shape[0], heightmap.shape[1]))
        for color in np.unique(heightmap):
            geomap_outlines = geomap_outlines + self.get_outline(heightmap, color)

        #Step 7: Create colors for every geolayer
        geomap = np.zeros((heightmap.shape[0], heightmap.shape[1], 3)).astype('int')
        for layer in np.unique(heightmap):
            layer_color = np.random.randint(120, 256, size=3)
            for x in range(0, heightmap.shape[0]):
                for y in range(0, heightmap.shape[1]):
                    if heightmap[x][y] == layer:
                        geomap[x][y] = layer_color
                        #print(layer_color)

        #Step 8: Create final isomap and outlines
        binary_isomap = self.binarize_colors(isomap, 1)
        binary_outlines = self.binarize_colors(geomap_outlines, 1)

        #Step 8: Change layers colors to needed
        topogeomap = geomap.copy()
        for x in range(IMGW):
            for y in range(IMGH):
                if binary_isomap[x][y] == 1:
                    topogeomap[x][y] = ISOCOLOR
                elif binary_outlines[x][y] == 1:
                    topogeomap[x][y] = GEOCOLOR

        return topogeomap, binary_isomap, binary_outlines


if __name__ == '__main__':
    GeomapGenerator().generate_dataset(5, 1024, 1024, [150, 75, 0], [75, 75, 75])


import PIL
from PIL import Image, ImageDraw

class Section_Creator():

    def get_heights_by_pos(self, heights, isopos, position):
        #Нет мыслей кроме как перебор
        new_heights = np.array([])
        for pos in position:
            new_heights = np.append(new_heights, heights[np.argmin(np.array([abs(isopos[i] - pos) for i in range(isopos.size - 1)]))])
        return new_heights

    def interpolate_array(self, arr, interpolate_value=300):
        new_array = np.array([])
        for i in range(0, len(arr) - 1):
            new_array = np.concatenate((new_array, np.linspace(arr[i], arr[i + 1], interpolate_value, endpoint=False)))
        new_array = np.append(new_array, arr[arr.size - 1])                                               
        return new_array

    def create_section(self, isodata, geodata, vert_scale, horizontal_scale, interpolate_value=300, pixels_in_cm=118):
        """
        Создание геологического разреза по данным высот (isodata) и геологических слоев (geodata)
        Формат: isodata -> [np.array([H1, H2, ..., HN]), np.array([p1, p2, ..., pN])]
                geodata -> np.array([p1, p2, ..., pN])
                vert_scale, horizontal_scale: масштаб карты (например, 1:2000) в степени -1. Пример: Масштаб 1:2000 -> vert_scale = 2000
        Возвращает:  а ща глянем

        """
        min_height = isodata[0].max()
        max_height = isodata[0].max()

        #Шаг 1: Переводим позиции в вид [0..1]
        max_position = np.max(np.concatenate((isodata[1], geodata)))
        min_position = np.min(np.concatenate((isodata[1], geodata)))

        geodata, isodata[1] = geodata / max_position, isodata[1] / max_position

        #Шаг 2: Интерполируем позиции и высоты
        interpolated_isopos = self.interpolate_array(isodata[1], interpolate_value)
        interpolated_height = spline(isodata[1], isodata[0], interpolated_isopos) #self.interpolate_array(isodata[0], interpolate_value)

        #Шаг 3: Теперь нам нужно получить высоты для каждой geolayer позиции
        geoheight = self.get_heights_by_pos(interpolated_height, interpolated_isopos, geodata) #Готово

        #Шаг 4: Отрисовка самой 



