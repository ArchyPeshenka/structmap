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