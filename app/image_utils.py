# import the necessary packages
import numpy as np
import imutils
import cv2
import time

def align_images(image, template, maxFeatures=500, keepPercent=0.2,
	debug=False):
	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)

	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)

	# sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)

	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]

	# check to see if we should visualize the matched keypoints
	if debug:
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, None)
		matchedVis = imutils.resize(matchedVis, width=1000)
		cv2.imshow("Matched Keypoints", matchedVis)
		cv2.waitKey(0)

	# allocate memory for the keypoints (x,y-coordinates) from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")

	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt

	# compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

	# use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))

	# return the aligned image
	return aligned

def get_image_similarity_score(image1, kp2, des2):
	"""
    Used to get the similarity score between a template and an image, to find the best
    template to use.

    The similarity score is based on the number of good matchs based on the Lowe Ratio
    """
	print('Inside get_image_similarity_score function')
	lowe_ratio = 0.70
	
	# convert both the input image and template to grayscale
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # create feature matcher
	finder = cv2.SIFT_create()
	start = time.time()
	kp1, des1 = finder.detectAndCompute(img1,None)
	end = time.time()
	print(f'finding key points took: {end - start} seconds')

	# finder = cv2.ORB_create()
    # # find the keypoints and descriptors with SIFT
	# kp1, des1 = finder.detectAndCompute(img1,None)
	# kp2, des2 = finder.detectAndCompute(img2,None)

    # BFMatcher with default params
	# print('create Flann Matcher')
	# FLANN_INDEX_KDTREE = 0
	# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
	# search_params = dict(checks=5)   # or pass empty dictionary

	# flann = cv2.FlannBasedMatcher(index_params,search_params)
	
	print('create Flann Matcher')
	start = time.time()
	matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
	matches = matcher.knnMatch(des1, des2, 2)
	end = time.time()
	print(f'matching key points took: {end - start} seconds')
	
	# matches = flann.knnMatch(des1,des2,k=2)
	# print('create BF Matcher')
	# start = time.time()
	# bf = cv2.BFMatcher()
	# matches = bf.knnMatch(des1,des2, k=2)
	# end = time.time()
	# print(f'matching key points took: {end - start} seconds')


    # Apply ratio test
	good = []
	# good_matches = []
	
	print('find good matches')
	# for i,(m,n) in enumerate(matches):
	# 	if m.distance < 0.7*n.distance:
	# 		good.append([m])
	start = time.time()
	for m,n in matches:
		if m.distance < lowe_ratio*n.distance:
			good.append([m])

	end = time.time()
	print(f'finding good points took: {end - start} seconds')


	# matchedVis = cv2.drawMatches(image1, kp1, image2, kp2,
	# 	good_matches, None)
	# matchedVis = imutils.resize(matchedVis, width=1000)
	# cv2.imshow("Template Matching", matchedVis)
	# cv2.waitKey(0)
	print(f'number of good matches: {len(good)}')
	return len(good)
