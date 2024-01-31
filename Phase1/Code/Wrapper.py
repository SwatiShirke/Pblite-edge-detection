#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
import imutils
from sklearn.cluster import KMeans

def Gaussian_filter(kernel_size, sigma):
	"""
	This is 2d gaussian filter, creates filter of given Kernel size and variance
	"""
	sigma_x, sigma_y = sigma
	size = int(kernel_size)
	gauss = np.zeros([kernel_size, kernel_size])
	if (kernel_size %2==0):
		center_index = kernel_size /2
	else:
		center_index = (kernel_size-1) /2

	x,y = np.meshgrid(np.linspace(-center_index,center_index,kernel_size),np.linspace(-center_index,center_index,kernel_size))
	part_1 = 0.5 / ( math.pi * sigma_x * sigma_y) 
	#print(part_1)
	part_2 = np.exp(-((x**2 + y**2)/ 2.0 * sigma_x * sigma_y))
	#print(part_2)
	out = part_1 * part_2
	return out

def new_gauu(kernel_size, sigma):
	
	sigma_x, sigma_y = sigma
	Gauss = np.zeros([kernel_size, kernel_size])
	# x = np.linspace(0,kernel_size)
	# y = np.linspace(0,kernel_size)
	if(kernel_size/2):
		index = kernel_size/2
	else:
		index = (kernel_size - 1)/2
	x,y= np.meshgrid(np.linspace(-index,index,kernel_size),np.linspace(-index,index,kernel_size))
	term1 = 0.5/(np.pi*sigma_x*sigma_y)
	
	term2 = np.exp(-((np.square(x)/(np.square(sigma_x))+(np.square(y)/(np.square(sigma_y))))))/2
	Gauss = term1*term2
	return Gauss


def DOG_filter_bank(no_orientation, scales, kernel_size):
	"""
	scales: no of scales for filter 
	orientation: no of different orientation angles
	ex - 2 * 16
	"""
	filter_bank = []
	##create sobel kernels
	Sobel_x = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	Sobel_y = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])

	for scale_value in scales:
		sigma = [scale_value,scale_value] 
		#gauss = Gaussian_filter(kernel_size,sigma)
		gauss = new_gauu(kernel_size, sigma)
		gauss_x = cv2.filter2D(gauss, -1,Sobel_x)
		#print(gauss_x)
		gauss_y = cv2.filter2D(gauss, -1,Sobel_y)
		for orient in range(no_orientation):
			gauss_orientation = 2 * math.pi * (orient / no_orientation)
			filter = (gauss_x * math.cos(gauss_orientation)) + (gauss_y * math.sin(gauss_orientation))
			filter_bank.append(filter)
		#print(filter_bank)

	return filter_bank

def print_filter_bank(filter_bank, file_path, columns):
	rows = math.ceil(len(filter_bank)/columns)
	plt.subplots(rows,columns, figsize=(100,100))
	for i in range(len(filter_bank)):
		plt.subplot(rows, columns,i+1)
		plt.axis('off')
		plt.imshow(filter_bank[i],cmap='gray')
	#plt.tight_layout(pad = 50)	
	plt.savefig(file_path)	
	plt.close()


def read_images(directory):
	image_list = []
	##images in the given directory have names in sequence 1,2,3
	list = os.listdir(directory)
	for i in range(len(list)):

		path = directory + "/" + str(i+1) + ".jpg"
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		##exception handling
		if img is not None:
			image_list.append(img)
		else:
			print("error in reading image: ", i)

	return image_list

def show_images(image_list):
	#show images in the list
	for img in image_list:
		#window_name = 'image'
		cv2.namedWindow("Show Image")
		#cv2.imshow("input",img)
		plt.imshow(img)
		plt.show()

def LM_filter_bank(no_orientation, scales, kernel_size):
	gaussian_scale = scales
	derivative_scale = scales[0:3]
	LOG_scale = scales + [i * 3 for i in scales]

	#LOG_scale = scales	
	#for i in range(len(scales)):
	#	log_scale = 3 * scales[i]
	#	LOG_scale.append(log_scale)

	##declare lists
	filter_bank = []
	gaussian_filter = []
	first_derivative = []
	second_derivative = []
	LOG_filter = []

	Sobel_x = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	Sobel_y = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])

	for scale in derivative_scale:
		sigma = [scale, scale]
		gauss = new_gauu(kernel_size,sigma)
		first_dx = cv2.filter2D(gauss,-1,Sobel_x) 
		first_dy = cv2.filter2D(gauss,-1,Sobel_y) 
		second_dx = cv2.filter2D(first_dx, -1, Sobel_x)
		second_dy = cv2.filter2D(first_dy, -1, Sobel_y)
		## first and second derivative of gaussian
		for i in range(no_orientation):
			orient = 2 * math.pi * (i / no_orientation)
			first_derivative_filter  =  (first_dx * math.cos(orient)) + (first_dy * math.sin(orient))
			second_derivative_filter = (second_dx * math.cos(orient)) + (second_dy * math.sin(orient))
			first_derivative.append(first_derivative_filter )
			second_derivative.append(second_derivative_filter)
			
	##Log of Gaussian filter 
	for scale in LOG_scale:
		sigma = [scale,scale]
		gauss = new_gauu(kernel_size,sigma)
		LOG_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
		LOG_filter.append(cv2.filter2D(gauss,-1,LOG_kernel))

	##Gaussian filter
	for scale in gaussian_scale:
		sigma = [scale, scale]
		gauss = new_gauu(kernel_size,sigma)
		gaussian_filter.append(gauss)

	#print(len(first_derivative))
	#print(len(second_derivative))
	#print(len(LOG_filter))
	#print(len(gaussian_filter))

	filter_bank = first_derivative + second_derivative + LOG_filter + gaussian_filter
	return filter_bank

def Gabor_filter(sigma,theta,v_lambda,psi,gamma, kernel_size):
	sigma_x = sigma
	sigma_y = float(sigma) / gamma

	if (kernel_size %2==0):
		center_index = kernel_size /2
	else:
		center_index = (kernel_size-1) /2

	x,y = np.meshgrid(np.linspace(-center_index,center_index,kernel_size),np.linspace(-center_index,center_index,kernel_size))

	x_theta = x * np.cos(theta) + y * np.sin(theta)
	y_theta = - x * np.sin(theta) + y * np.cos(theta)

	gabor = np.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * np.cos(2 * np.pi / v_lambda* x_theta + psi)

	return gabor

def Gabor_filter_bank(no_orientation, scales,v_lambda,psi,gamma, kernel_size):
	filter_bank = []	
	for scale_value in scales:			
		for orient in range(no_orientation):

			gauss_orientation = 2 * math.pi * (orient / no_orientation)
			gabor = Gabor_filter( scale_value , gauss_orientation ,v_lambda,psi,gamma, kernel_size)
			filter_bank.append(gabor)

	return filter_bank
	
def half_disk_mask(angle, radius):
	centre = radius
	size = 2 * radius + 1 
	half_disk = np.zeros([size, size])	

	for i in range(radius):
		for j in range(size):
			distance = (i-centre) **2 + (j - centre) ** 2
			if distance <= radius ** 2:
				half_disk[i,j] = 1

	half_disk = imutils.rotate(half_disk, angle)
	half_disk[half_disk<=0.5] = 0
	half_disk[half_disk>0.5] = 1
	return half_disk

def half_disk_bank(no_orientation, scale_list):
	filter_bank = []
	for radi in scale_list:
		paired_filter_bank = []
		unpaired_filter_bank = []
		for j in range(no_orientation):
			angle = (j / no_orientation) * 360 
			half_disk = half_disk_mask(angle, radi)
			unpaired_filter_bank.append(half_disk)
		print("chp 1")
		print(len(unpaired_filter_bank))

		##lets pair the masks now
		for k in range(int(no_orientation /2)):
			paired_filter_bank.append(unpaired_filter_bank[k])
			paired_filter_bank.append(unpaired_filter_bank[k + int (no_orientation /2)])
		print("chpt 2")	
		print(len(paired_filter_bank))
		#filter_bank.append(paired_filter_bank)
		filter_bank += paired_filter_bank
	print(len(filter_bank))
	return filter_bank

def chi_square_distance(input, bins, filter_bank):
	chi_sqr_distance_list = []
	no_filter = len(filter_bank)
	iter = 0
	while iter < no_filter:
		left_mask = filter_bank[iter]
		right_mask = filter_bank[iter + 1]
		temp_img = np.zeros(input.shape)
		chi_sqr_dist = np.zeros(input.shape)
		min_input = np.min(input)
		for bin in range(bins):
			temp_img[ input == bin ] = 1
			g_i = cv2.filter2D(temp_img,-1,left_mask)
			h_i = cv2.filter2D(temp_img,-1,right_mask)
			#print(g_i)
			#print(h_i)
			chi_sqr_dist += (g_i - h_i) ** 2 / 2 * (g_i + h_i )
		#chi_sqr_dist = chi_sqr_dist
		chi_sqr_distance_list.append(chi_sqr_dist)
		##jump to next pair of mask
		iter = iter + 2
	chi_sqr_distance_list = chi_sqr_distance_list



	#chi_sqr_fin_list = np.mean(chi_sqr_distance_list, axis =0)

	#chi_sqr_distance_list = []
	#no_filter = len(filter_bank)
	#iter = 0
	#while iter < no_filter:
	#	left_mask = filter_bank[iter]
	#	right_mask = filter_bank[iter + 1]
	#	chi_sqr_dist = np.zeros(input.shape)
	#
	#	for value in range(bins):
	#
	#		bin_img = (input == value)
	#		bin_img = np.float32(bin_img)
	#		g_i = cv2.filter2D(bin_img,-1,left_mask)
	#		h_i = cv2.filter2D(bin_img,-1,right_mask)
	#	
	#		chi_sqr_dist += (g_i - h_i) ** 2 / 2 * (g_i + h_i )
	#	
	#	chi_sqr_distance_list.append(chi_sqr_dist)
	#	##jump to next pair of mask
	#	iter = iter + 2
	#chi_sqr_distance_list = np.array(chi_sqr_distance_list)
	return chi_sqr_distance_list

def create_maps(n_clusters, n_init, img_list , x_size, y_size, bank, type_of_map, img_no):
		
		k_means = KMeans(n_clusters=n_clusters, init='k-means++',max_iter=300, n_init=n_init, random_state=0)
		k_means.fit(img_list) 
		labels= k_means.predict(img_list)
		print(len(labels))
		map = labels.reshape([x_size,y_size])
		count = 1
		file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/texton_maps"
		file_name = file_path + "/"+ type_of_map +str(img_no)+ ".png"
		#print(texton.shape)
		cv2.imwrite(file_name, map)
		D_grad = np.array(chi_square_distance(map, n_clusters ,bank))
		D_grad = np.mean(D_grad, axis =0)
		
		file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/texton_grad"
		file_name = file_path +"/"+ type_of_map + "_grad" +str(img_no) + ".png"
		cv2.imwrite(file_name, D_grad)

		return D_grad

def pblite_edge_detector(T_g, B_g, C_g, Canny_edge, Sobel_edges, weights):
	##conver baseline to gray scale
	Canny_edge = cv2.cvtColor(Canny_edge, cv2.COLOR_BGR2GRAY)
	Sobel_edges = cv2.cvtColor(Sobel_edges, cv2.COLOR_BGR2GRAY)
	term1 = (T_g + B_g + C_g) / 3
	term2 = (weights[0] * Canny_edge + weights[1] *Sobel_edges)

	pb_edge = np.multiply (term1 , term2 )
	return pb_edge

def apply_filter(filter_bank, image,file_path):
	filtered_image_list = []
	image_count = 0
	##convert image into gray scale
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_list =[]
	for filter in filter_bank:
		filtered_img = cv2.filter2D(gray_img,-1, filter)
		image_count +=1		
		#file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/Filter_banks/filtered_images"
		file_name = file_path + "/"+"img_"+ str(image_count )+ ".png"
		cv2.imwrite(file_name, filtered_img)
		image_list.append(filtered_img)
	filtered_image_list += image_list
	
	#print_filter_bank(DOG_bank, file_path, 16)

	return filtered_image_list

def main():

	

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	#guass = Gaussian_filter(5,1)
	#print(guass)
	DOG_bank = DOG_filter_bank(16,[7,8],128)
	file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/Filter_banks/DOG_filter_bank.png"
	print_filter_bank(DOG_bank, file_path, 16)

	DOG_bank2 = DOG_filter_bank(16,[4,5],128)
	file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/Filter_banks/DOG_filter_bank2.png"
	print_filter_bank(DOG_bank, file_path, 16)


	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LM_bank_small = LM_filter_bank(6, [1,np.sqrt(2),2,2 * np.sqrt(2)], 49)
	file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/Filter_banks/LM_small_bank.png"
	print_filter_bank(LM_bank_small, file_path, 6)
	LM_bank_large = LM_filter_bank(6, [np.sqrt(2),2,2 * np.sqrt(2),4], 49)	
	file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/Filter_banks/LM_large_bank.png"
	print_filter_bank(LM_bank_large, file_path, 6)



	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gabor_bank = Gabor_filter_bank(8, [6,10,14,18],2,0.5,0.5, 49)
	file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/Filter_banks/Gabor_bank.png"
	print_filter_bank(gabor_bank, file_path, 6)
	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disk_filter_bank = half_disk_bank(8,[8,9,10] )
	#print(len(half_disk_filter_bank))
	file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/Filter_banks/half_disk_bank.png"
	filtered_image_list = print_filter_bank(half_disk_filter_bank, file_path, 8)
	

	"""
	read images and created a filtered image list by applying all 3 filter LM_bank_small
	"""
	directory_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/BSDS500/Images"
	image_list = read_images(directory_path)
	#show_images(image_list)
	file_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/Filter_banks/filtered_images"
	#filtered_image_list = apply_filter(DOG_bank, image_list[1],file_path)	
	baseline_sobel_img_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/BSDS500/SobelBaseline"
	baseline_canny_img_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/BSDS500/CannyBaseline"
	output_path = "/home/swati/Documents/CV/YourDirectoryID_hw0/Phase1/Results/output"
	
	print("her@@@@@@@@@@@@@@@@@@@@@@@@")
	print(len(DOG_bank))
	print(len(LM_bank_small))
	print(len(LM_bank_large))
	print(len(DOG_bank2))
	##create a combined filter bank
	comibined_filter_bank = DOG_bank + LM_bank_small + LM_bank_large + DOG_bank2
	count = 0
	##get filtered img list 
	texton_grad = []	
	brightness_grad =[]
	colour_grad = []
	for i in range(len(image_list)):

		img = image_list[i]
		img_no = i +1 
		filtered_image_list = apply_filter(comibined_filter_bank, img, file_path)
		filtered_image_list  = np.array(filtered_image_list )
		##convert intp mat format and reshpae
		no_img,x,y= filtered_image_list.shape
		#print(no_img)
		mat_img = filtered_image_list.reshape([no_img, x *y])
		mat_img = mat_img.transpose()
		texton_grad = create_maps(64, 2 , mat_img , x, y, half_disk_filter_bank, "texton", img_no )
		brightness_grad = create_maps(16, 4 , mat_img , x, y, half_disk_filter_bank, "brightness", img_no  )

		cv2.namedWindow("Show Image")
		cv2.imshow("input",img)
		plt.imshow(img)
		plt.show()		

		x,y,c = img.shape
		input_mat = img.reshape([x*y,c])
		colour_grad = create_maps(16, 4 , input_mat , x, y, half_disk_filter_bank, "colour", img_no)

		##
		img_name = str (img_no) + ".png"
		sobel_pb = cv2.imread(baseline_sobel_img_path + "/" + img_name)
		canny_pb = cv2.imread(baseline_canny_img_path + "/" + img_name)
		cv2.imshow("input",sobel_pb)
		plt.imshow(sobel_pb)
		plt.show()		

		print("here")
		weights = [0.5,0.5]
		pb_edge = pblite_edge_detector(texton_grad, brightness_grad , colour_grad , canny_pb, sobel_pb, weights)
		print(len(pb_edge))
		output_file_name = output_path + "/detected_edge_"+ str(img_no) + ".png"
		cv2.imwrite(output_file_name, pb_edge)
		break
		if i != 0 and i% 2 ==  0:
			break		
		#Generate Texton Map
		#Filter image using oriented gaussian filter bank
		
		

		

	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    
if __name__ == '__main__':
    main()
 


