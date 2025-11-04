from tifffile import imread
from tifffile import imwrite
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.optimize import curve_fit

# Pixel Size: 117 nm
# Frame Rate = 33.3 fps

pixel_resolution = 117
frame_rate = 33.3

class LinkedList:
	def __init__(self, point, frame_number):
		self.start_frame = frame_number
		self.inactivity = 0
		self.children = 0
		self.active = True
		self.added = False
		self.first = point
		self.last = point
	def append(self, point):
		point.set_prev(self.last)
		self.last.set_next(point)
		self.last = point
		self.children += 1
	def set_inactive(self):
		self.active = False
	def increase_inactivity(self):
		self.inactivity += 1

class Point:
	def __init__(self, posx, posy, size):
		self.x = posx
		self.y = posy
		self.size = size
		self.prev = None
		self.next = None

	def set_next(self, point):
		self.next = point
	def set_prev(self, point):
		self.prev = point

def get_distance(point_one, point_two):
	return (point_one[0]-point_two[0])**2 + (point_one[1]-point_two[1])**2

def Boundary_Fill(posx, posy, mask, pntsx, pntsy):
	if posx >= 0 and posx < mask.shape[1] and posy >= 0 and posy < mask.shape[0]:
		if mask[posy][posx] == 0:
			return
		pntsx.append(posx)
		pntsy.append(posy)
		mask[posy][posx] = 0

		Boundary_Fill(posx-1, posy-1, mask, pntsx, pntsy)
		Boundary_Fill(posx, posy-1, mask, pntsx, pntsy)
		Boundary_Fill(posx+1, posy-1, mask, pntsx, pntsy)
		Boundary_Fill(posx-1, posy, mask, pntsx, pntsy)
		Boundary_Fill(posx+1, posy, mask, pntsx, pntsy)
		Boundary_Fill(posx-1, posy+1, mask, pntsx, pntsy)
		Boundary_Fill(posx, posy+1, mask, pntsx, pntsy)
		Boundary_Fill(posx+1, posy+1, mask, pntsx, pntsy)
	else:
		return

all_lists = []
distance_cutoff = 9

filename = "FlowCell41-12 MOVIE (Frames 259-1957) - Cropped.tif"

masked_frames = []

image_array = imread(filename)

print("File: " + filename)
print("Pixel Resolution: 1px = " + str(pixel_resolution) + " nm")
print("Framerate: " + str(frame_rate) + " fps")

print("TIFF dimenstions: " + str(image_array.shape))

intensities = []
for frame in image_array[-180:-1]:
	intensities.append(np.max(frame))
max_intensity = np.mean(intensities)
print("Lower Intensity Cutoff: " + str(max_intensity))

frame_number = 0
for frame in image_array:

	blur = cv.GaussianBlur(frame,(5,5),0)

	#plt.imshow(blur)
	#plt.show()
		
	mask = np.where(blur < max_intensity, 0, 255)
	#masked_frames.append(mask)

	#plt.imshow(mask.astype(np.uint8))
	#plt.show()

	'''
	new_im = Image.fromarray(mask.astype(np.uint8), mode='L')
	new_im.save("movie/"+str(frame_number)+".tif")
	'''

	hits = []
	img_x=mask.shape[1];
	img_y=mask.shape[0];

	image_np_array = np.zeros((img_y, img_x))

	for y in range(0, img_y, 1):
		for x in range(0, img_x, 1):
			if mask[y][x] > 0: #hit pixel on
				pointsx = []
				pointsy = []
				
				Boundary_Fill(x, y, mask, pointsx, pointsy)		
		
				centerx =  int(((min(pointsx) + max(pointsx))/2) + 0.5)
				centery = int(((min(pointsy)+max(pointsy))/2)+0.5)
				size = (max(pointsx)-min(pointsx)) * (max(pointsy)-min(pointsy))
			
				point = Point(centerx, centery, size)
				hits.append((centerx, centery))

				image_np_array[centery][centerx] = 255

				if all_lists != []:
					for List in all_lists:
						if List.active == True and List.added == False and get_distance((point.x, point.y),(List.last.x, List.last.y)) <= distance_cutoff:
							List.append(point)
							List.added = True
						else:
							pass

					if point.prev == None:
						NewList = LinkedList(point, frame_number)
						all_lists.append(NewList)
				else:
					NewList = LinkedList(point, frame_number)
					all_lists.append(NewList)

	for List in all_lists:
		if List.added == False and List.active == True:
			List.increase_inactivity()
		if List.inactivity > 2:
			List.set_inactive()
		List.added = False

	
	'''
	new_im = Image.fromarray(image_np_array.astype(np.uint8), mode='L')
	new_im.save("movie_centers/"+str(frame_number)+".tif")
	'''


	frame_number += 1
	'''
	# plot
	fig, ax = plt.subplots()

	for i,j in hits:
		ax.scatter(i,j)

	ax.set(ylim=(0, mask.shape[0]), yticks = (range(0,mask.shape[0], 100)), xlim=(0, mask.shape[1]), xticks = (range(0,mask.shape[1], 100)))
	#ax.set(ylim=(0, mask.shape[0]),xlim=(0, mask.shape[1]))

	plt.show()
	'''

#imwrite('temp.tif', masked_frames, photometric='minisblack')

def sort_param(e):
	return e.children

#all_lists.sort(reverse=True, key=sort_param)

total_kymograph = []
local_kymographs = []

timeframe =  1

global_timeseries = [[] for i in range(int(timeframe*frame_rate))]
total_plectonemes = 0

fig, ax = plt.subplots()
plt.xlabel("Time (s)")
plt.ylabel("Cumulative Distance Traveled (nm^2)")
frame = 0
for List in all_lists:
	if List.children > 1:#int(0.5*frame_rate) and List.start_frame < 1700:
		total_plectonemes += 1
		local_points = []
		coordinates = []
		pos = List.start_frame
		point = List.first
		coordinates.append((point.x, point.y))
		image_np_array = np.zeros((img_y, img_x))
		image_np_array[point.y][point.x] = 255
		new_im = Image.fromarray(image_np_array.astype(np.uint8), mode='L')
		new_im.save(str(frame)+".tif")
		frame+=1
		while point.next != None:
			point = point.next
			coordinates.append((point.x, point.y))
			total_kymograph.append((point.x*pixel_resolution, pos))
			local_points.append(point.x*pixel_resolution/1000)	
			
			image_np_array = np.zeros((img_y, img_x))
			image_np_array[point.y][point.x] = 255
			new_im = Image.fromarray(image_np_array.astype(np.uint8), mode='L')
			new_im.save(str(frame)+".tif")

			frame+=1
			pos += 1

		local_kymographs.append(local_points)
		cumulative_distance_travelled = [0]
		for index, value in enumerate(coordinates):
			if index > 0:
				distance = get_distance(coordinates[index], coordinates[index-1]) * pixel_resolution**2
				cumulative_distance_travelled.append(cumulative_distance_travelled[index-1] + distance)
		ax.plot([f/frame_rate for f in range(len(cumulative_distance_travelled))], cumulative_distance_travelled)

		timestep = 1
		while timestep <= int(timeframe*frame_rate):#1 seconds
			if timestep < len(coordinates)-1:
				offset = 0
				for i in range(len(coordinates)-timestep):
					index = 1
					for j in range(int((len(coordinates)-offset-1)/timestep)):
						distance = get_distance(coordinates[index*timestep+offset], coordinates[(index*timestep-timestep)+offset]) * pixel_resolution**2
						global_timeseries[timestep-1].append(distance)
						index += 1
					offset += 1
			timestep += 1

print("Number of Plectonemes: " + str(total_plectonemes))
plt.show()

'''
fig, ax = plt.subplots()
plt.ylabel("Time (s)")
plt.xlabel("Position (nm)")
for px, py in kymograph:
	ax.scatter(px*pixel_resolution, py/frame_rate)
plt.show()
'''

fig, ax = plt.subplots()
plt.ylabel("Time (s)")
plt.xlabel("Position (um)")
rolling_index = 0
for kymo in local_kymographs:
	ax.plot(kymo,[(rolling_index+i)/frame_rate for i in range(len(kymo))])
	rolling_index += len(kymo)	
plt.show()

mean_distances = []
errors = []
for series in global_timeseries:
	mean_distances.append(np.mean(series))
	errors.append(np.std(series))


fig, ax = plt.subplots(2,1,sharex=True,sharey=False)
ax[0].set_ylabel("Mean distance travelled squared (nm^2)")
ax[1].set_xlabel("Time interval (s)")
ax[1].set_ylabel("residuals")

time_in_seconds = [(f+1)/frame_rate for f in range(len(mean_distances))]
ax[0].errorbar(time_in_seconds, mean_distances, yerr=errors,fmt='o')

def straight_line(x,m,c):
	return m*x + c

fit_start_seconds = 0.0
fit_start_frames = int(fit_start_seconds*frame_rate)
fit_end_seconds = 0.6
fit_end_frames = int(fit_end_seconds*frame_rate)

parameters, parameters_covarience = curve_fit(straight_line, time_in_seconds[fit_start_frames:fit_end_frames], mean_distances[fit_start_frames:fit_end_frames], sigma=errors[fit_start_frames:fit_end_frames])

residuals = np.array(mean_distances[fit_start_frames:fit_end_frames]) - straight_line(np.array(time_in_seconds[fit_start_frames:fit_end_frames]), parameters[0], parameters[1])

print("Gradient = " + str(parameters[0]) + ", Intercept = " + str(parameters[1]) )
diffusion_constant_in_micro_meters = (parameters[0]/1000000)/2
print("Diffusion Constant = "+str(diffusion_constant_in_micro_meters) + " um^2/sec +- " + str(np.sqrt(np.diag(parameters_covarience))[0]/1000000))
ax[0].plot(time_in_seconds[fit_start_frames:fit_end_frames], straight_line(np.array(time_in_seconds[fit_start_frames:fit_end_frames]), *parameters))
ax[1].scatter(time_in_seconds[fit_start_frames:fit_end_frames], residuals)
ax[1].plot(time_in_seconds[fit_start_frames:fit_end_frames],[0 for i in range(len(time_in_seconds[fit_start_frames:fit_end_frames]))])
plt.show()

#Plot in uM
residuals = np.array(mean_distances[fit_start_frames:fit_end_frames]) - straight_line(np.array(time_in_seconds[fit_start_frames:fit_end_frames]), parameters[0], parameters[1])
residuals = residuals/1000000
fig, ax = plt.subplots(2,1,sharex=True,sharey=False)
ax[0].set_ylabel("Mean distance travelled squared (um^2)")
ax[1].set_xlabel("Time interval (s)")
ax[1].set_ylabel("residuals")
ax[0].errorbar(time_in_seconds, np.array(mean_distances)/1000000, yerr=np.array(errors)/1000000,fmt='o')
ax[0].plot(time_in_seconds[fit_start_frames:fit_end_frames], straight_line(np.array(time_in_seconds[fit_start_frames:fit_end_frames]), *parameters)/1000000)
ax[1].scatter(time_in_seconds[fit_start_frames:fit_end_frames], residuals)
ax[1].plot(time_in_seconds[fit_start_frames:fit_end_frames],[0 for i in range(len(time_in_seconds[fit_start_frames:fit_end_frames]))])
plt.show()
'''
plt.subplot(2,2,1)
plt.imshow(im2)
plt.title("Original")
plt.subplot(2,2,2)
plt.imshow(mask)
plt.title("80% of max brightness threshold")
plt.subplot(2,2,3)
plt.imshow(blur)
plt.title("Original with 5x5 gaussian blur")
plt.subplot(2,2,4)
plt.imshow(mask2)
plt.title("blur + 80% of max brighness threshold")
plt.show()
'''
