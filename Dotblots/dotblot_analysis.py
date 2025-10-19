import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import pandas as pd

directory: str = sys.argv[1]
threshold: int = 0
if len(sys.argv) > 2:
	threshold = int(sys.argv[2])/100
elif len(sys.argv) < 1 or len(sys.argv) > 3:
	print("Incorrect use: python3 <scriptname.py> <path to folder with images> <optional intensity threshold percentage>")
	exit

try:
	folder = os.listdir(directory)
except:
	print("Input folder does not exist!")
	exit

if os.path.isdir(directory[:-1]+"_processed_files"):
	for f in os.listdir(directory[:-1]+"_processed_files"):
		os.remove(directory[:-1]+"_processed_files/"+f)
	os.rmdir(directory[:-1]+"_processed_files")

os.mkdir(directory[:-1]+"_processed_files")


def Boundary_Fill(posx: int, posy: int, mask, pntsx: list, pntsy: list, pntsi: list) -> None:	

	stack: list = []
	stack.append((posx, posy))

	while stack != []:
		current_pointx, current_pointy = stack.pop(-1)
		current_intensity: float = mask[current_pointy][current_pointx]
		if current_pointx >= 0 and current_pointx < mask.shape[1] and current_pointy >= 0 and current_pointy < mask.shape[0]:
			if current_intensity != 0:
				pntsx.append(current_pointx)
				pntsy.append(current_pointy)
				pntsi.append(current_intensity)
				mask[current_pointy][current_pointx] = 0

				stack.append((current_pointx-1, current_pointy-1))
				stack.append((current_pointx, current_pointy-1))
				stack.append((current_pointx+1, current_pointy-1))
				stack.append((current_pointx-1, current_pointy))
				stack.append((current_pointx+1, current_pointy))
				stack.append((current_pointx-1, current_pointy+1))
				stack.append((current_pointx-1, current_pointy+1))
				stack.append((current_pointx+1, current_pointy+1))
class Spot:
	def __init__(self, centerx: int, centery: int, size: int, intensity: int, error: float):
		self.centerx: int = centerx
		self.centery: int = centery
		self.size: int = size
		self.total_intensity: int = intensity
		self.mean_intensity: float = self.total_intensity / self.size
		self.error: float = error

for file in folder:

	spots: list = []

	'''
	# OPTIONAL: supplementary data for checking output
	all_pointsx = []
	all_pointsy = []
	'''

	im = cv2.imread(directory+file, cv2.IMREAD_GRAYSCALE)

	max_brightness = np.max(im)
	
	if threshold != 0:
		mask = np.where(im < threshold*max_brightness, 255-im, 0)
	else:
		mask = np.where(im < 0.8*max_brightness, 255-im, 0)

	'''
	# OPTIONAL: shows the thresholded image visually
	plt.imshow(mask, origin="lower")
	plt.show()
	'''

	'''
	# OPTIONAL: saves the thresholded image to the disk
	cv2.imwrite(directory[:-1]+"_processed_files/processed_file_"+file, mask)
	'''

	img_x_size: int = mask.shape[0]
	img_y_size: int = mask.shape[1]
	for y in range(img_x_size):
		for x in range(img_y_size):
			if mask[y][x] != 0:
				pointsx: list[int] = []
				pointsy: list[int] = []
				points_intensites: list[int] = []
				Boundary_Fill(x, y, mask, pointsx, pointsy, points_intensites)

				centerx: int = int((max(pointsx)+min(pointsx))/2 + 0.5)
				centery: int = int((max(pointsy)+min(pointsy))/2 + 0.5)
				total_intensity: int = np.sum(points_intensites)
				standard_deviation: float = np.std(points_intensites)
				
				'''
				# OPTIONAL: supplementary data for checking output
				for x in pointsx:
					all_pointsx.append(x)
				for y in pointsy:
					all_pointsy.append(y)
				'''

				spots.append(Spot(centerx, centery, len(points_intensites), total_intensity, standard_deviation))
	
	'''
	# OPTIONAL: shows all of the processed pixels as a scatter plot, to check woth the original image
	# REQUIRES all_pointsx and all_pointsy

	fig, ax = plt.subplots()
	plt.ylim(0,img_y_size)
	plt.xlim(0,img_x_size)
	plt.scatter(all_pointsx, all_pointsy)
	plt.show()
	'''

	'''
	# OPTIONAL: For plotting the center of the points, to double check output
	# Also uncomment the final two commented lines (140 and 154)
	fig, ax = plt.subplots()
	plt.ylim(0,img_y_size)
	plt.xlim(0,img_x_size)
	'''
	
	out_data = {"spot": [], "center x": [], "center y": [], "size": [], "total signal intensity": [], "mean signal intensity": [], "stdev": []}

	i = 0
	for spot in spots:
		# plt.scatter(spot.centerx, spot.centery)
		out_data["spot"].append(i)
		out_data["center x"].append(spot.centerx)
		out_data["center y"].append(spot.centery)
		out_data["size"].append(spot.size)
		out_data["total signal intensity"].append(spot.total_intensity)
		out_data["mean signal intensity"].append(spot.mean_intensity)
		out_data["stdev"].append(spot.error)
		i += 1

	dataframe = pd.DataFrame(out_data)
	dataframe.to_csv(directory[:-1]+"_processed_files/statistics_"+ os.path.splitext(file)[0]+".csv", index=False)
	print(dataframe)

	#plt.show()
