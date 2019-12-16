#!/usr/bin/env python
# coding: utf-8

# Project: Face Synthesis 

# Computational Photography - Fall 2019

# In[1]:


import cv2
import numpy as np
from numpy.linalg import svd, inv, pinv
import os
import math
import decimal
from IPython.display import clear_output
import utils

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[2]:


from PIL import Image, ImageDraw
import dlib
import face_recognition


# In[3]:


# setting up custom parameters

dir_faces     = './faces'
dir_ref       = './ref'

filn_ref   = 'ref/ref.jpeg'


# In[4]:


ref = cv2.cvtColor(cv2.imread(filn_ref), cv2.COLOR_BGR2RGB)
frameHeight, frameWidth, frameChannels = ref.shape

# plt.imshow(ref)
# print(ref.dtype)
# print(ref.shape)


# In[42]:


filenames = []
filesinfo = os.scandir(dir_faces)


# In[43]:


filenames = [f.path for f in filesinfo if f.name.endswith(".jpg") or f.name.endswith(".jpeg")]
filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


# In[44]:


frameCount = len(filenames)
faces = np.zeros((frameCount, frameHeight, frameWidth, frameChannels), dtype=np.uint8)


# In[45]:


for idx, file_i in enumerate(filenames):
    faces[idx] = cv2.cvtColor(cv2.imread(file_i), cv2.COLOR_BGR2RGB)


# In[ ]:





# In[11]:


def get_average_face(ref, faces, choices=range(faces.shape[0])):
    
    num_choices = len(choices)
    frameHeight, frameWidth, frameChannels = ref.shape
    img_stack = np.zeros((num_choices, frameHeight, frameWidth, frameChannels), dtype=np.float32)
    result = np.zeros(ref.shape, dtype=np.uint8)

    #PRINT COUNTER
    counter = 1

    # Copy warped img into stack
    for idx, choice in enumerate(choices):
        img_stack[idx,:,:,:] = faces[choice]

    # Reduce stack
    for h in range(frameHeight):
        for w in range(frameWidth):
            for f in range(num_choices):
                pixel = img_stack[f, h, w]
                flag_allNan = True

                if np.array_equal(pixel, np.array([0,0,0])):
                    img_stack[f, h, w] = np.array([np.nan, np.nan, np.nan])
                elif flag_allNan:
                    flag_allNan = False

            if flag_allNan:
                img_stack[0, h, w] = np.array([0,0,0]) # make sure not all nan

            med = np.nanmean(img_stack[:, h, w, :], axis=0)
            result[h, w, :] = med

            # Print Processing
            if 100*(h+1)/(frameHeight) >= counter:
                clear_output(wait=True)
                counter += 1
                print('Processing:', int(round(100*(h+1)/(frameHeight), 0)),"%")
    
    return result


# In[12]:


def swap_face(ref, new_face):
    frameHeight, frameWidth, frameChannels = ref.shape
    
    for h in range(frameHeight):
        for w in range(frameWidth):
            pixel = new_face[h, w]
            
            if np.array_equal(pixel, np.array([0,0,0])):
                ref[h, w] = new_face[h, w]
    
    return ref


# In[13]:


choices2 = [0,1]
choices3 = [0,1,2,3,4,5]

# avrg_face = get_average_face(ref, W_dict, choices2)
avrg_face3 = get_average_face(ref, faces, choices3) 
plt.imshow(avrg_face3) 


# In[ ]:





# In[ ]:


#### Triangulation Warp ####


# In[14]:


def get_raw_face_landmarks_lst(img):
    face_landmarks_list = face_recognition.face_landmarks(img)
    raw_face_landmarks_lst = []
    
    for face_landmarks in face_landmarks_list:
        for key, val in face_landmarks.items():
            if (key != "bottom_lip"):
                for pt in val:
                    raw_face_landmarks_lst.append(pt)
            else:
                skip = [0,6,7,11]
                for idx in range(len(val)):
                    if idx not in skip:
                        raw_face_landmarks_lst.append(val[idx])
    
    return raw_face_landmarks_lst


# In[15]:


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


# In[40]:


def get_triangle_indices(ref):
    
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    
    ref_FL_lst = get_raw_face_landmarks_lst(ref)
    ref_FL = np.array(ref_FL_lst, np.int32)
    
    convexhull = cv2.convexHull(ref_FL)
    
    mask = np.zeros_like(ref_gray)
    cv2.fillConvexPoly(mask, convexhull, 255)

    ref_masked = cv2.bitwise_and(ref, ref, mask=mask)
    
    # Triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(ref_FL_lst)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indices_triangles = []
    
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((ref_FL == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((ref_FL == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((ref_FL == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indices_triangles.append(triangle)
            
    return indices_triangles


# In[37]:


def print_test_imgs(ref, isRef=True, indices=None):
    
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    
    ref_FL_lst = get_raw_face_landmarks_lst(ref)
    ref_FL = np.array(ref_FL_lst, np.int32)
    
    # landmark pts
    temp = np.copy(ref)
    for idx in range(ref_FL.shape[0]):
        pt = ref_FL[idx]
        x = pt[0]
        y = pt[1]
        cv2.circle(temp, (x,y), 3, (255, 0, 0), -1)
    cv2.imwrite('./test/facial_pts.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))
    
    # Convex Hull
    temp = np.copy(ref)
    convexhull = cv2.convexHull(ref_FL)
    cv2.polylines(temp, [convexhull], True, (255, 0, 0), 3)
    cv2.imwrite('./test/convexhull.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))

    # Mask
    temp = np.copy(ref)
    mask = np.zeros_like(ref_gray)
    cv2.fillConvexPoly(mask, convexhull, 255)
    masked = cv2.bitwise_and(temp, temp, mask=mask)
    cv2.imwrite('./test/mask.jpg', cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

    # Triangulation
    temp = np.copy(ref)
    
    if isRef:
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(ref_FL_lst)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indices_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = np.where((ref_FL == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((ref_FL == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((ref_FL == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indices_triangles.append(triangle)

        for triangle_index in indices_triangles:

            pt1 = ref_FL_lst[triangle_index[0]]
            pt2 = ref_FL_lst[triangle_index[1]]
            pt3 = ref_FL_lst[triangle_index[2]]

            cv2.line(temp, pt1, pt2, (255,0,0), 1)
            cv2.line(temp, pt2, pt3, (255,0,0), 1)
            cv2.line(temp, pt1, pt3, (255,0,0), 1)
    else:
        for triangle_index in indices:

            pt1 = ref_FL_lst[triangle_index[0]]
            pt2 = ref_FL_lst[triangle_index[1]]
            pt3 = ref_FL_lst[triangle_index[2]]

            cv2.line(temp, pt1, pt2, (255,0,0), 1)
            cv2.line(temp, pt2, pt3, (255,0,0), 1)
            cv2.line(temp, pt1, pt3, (255,0,0), 1)
    cv2.imwrite('./test/triangulation.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))


# In[19]:


def get_warped_face(ref, src, triangle_indices):
    
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    
    ref_FL_lst = get_raw_face_landmarks_lst(ref)
    ref_FL = np.array(ref_FL_lst, np.int32)
    
    src_FL_lst = get_raw_face_landmarks_lst(src)
    src_FL = np.array(src_FL_lst, np.int32)

    output_face = np.zeros_like(ref)
    
    for triangle_index in triangle_indices:
        
        # Triangulation of ref face
        tr1_pt1 = ref_FL_lst[triangle_index[0]]
        tr1_pt2 = ref_FL_lst[triangle_index[1]]
        tr1_pt3 = ref_FL_lst[triangle_index[2]]
        
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1

        cropped1 = ref[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)
        
        points1 = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                            [tr1_pt2[0] - x, tr1_pt2[1] - y],
                            [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points1, 255)
        cropped_triangle1 = cv2.bitwise_and(cropped1, cropped1, mask=cropped_tr1_mask)


        # Triangulation of src face
        tr2_pt1 = src_FL_lst[triangle_index[0]]
        tr2_pt2 = src_FL_lst[triangle_index[1]]
        tr2_pt3 = src_FL_lst[triangle_index[2]]
        
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        cropped2 = src[y: y + h, x: x + w]
        cropped_tr2_mask = np.zeros((h, w), np.uint8)
        
        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
        cropped_triangle2 = cv2.bitwise_and(cropped2, cropped2, mask=cropped_tr2_mask)
        
        # Warp triangles
        points1 = np.float32(points1)
        points2 = np.float32(points2)
        
        M = cv2.getAffineTransform(points2, points1)
        (x, y, w, h) = rect1
        warped_triangle = cv2.warpAffine(cropped_triangle2, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr1_mask)
        
        # Reconstruct warped face
        triangle_area = output_face[y: y + h, x: x + w]
        triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
        
        triangle_area = cv2.add(triangle_area, warped_triangle)
        
        output_face[y: y + h, x: x + w] = triangle_area       
        
#     mask_inv = cv2.bitwise_not(lines_space_mask)  
#     output_face_masked = cv2.bitwise_and(output_face,output_face,mask = mask_inv)
#     output_face = cv2.add(output_face_masked, cropped_triangle1)   
#         plt.imshow(cropped_triangle1)
#     cv2.imwrite('./teststh1.jpg', cv2.cvtColor(cropped_triangle1, cv2.COLOR_RGB2BGR))
#     cv2.imwrite('./teststh2.jpg', lines_space_mask)
#         cv2.imwrite('./teststh3.jpg', cv2.cvtColor(warped_triangle, cv2.COLOR_RGB2BGR))
#     cv2.imwrite('./teststh4.jpg', cv2.cvtColor(output_face, cv2.COLOR_RGB2BGR))

    return output_face


# In[22]:


def blend(ref, new_face):
    
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    
    ref_FL_lst = get_raw_face_landmarks_lst(ref)
    ref_FL = np.array(ref_FL_lst, np.int32)
    
    convexhull = cv2.convexHull(ref_FL)
    
    ref_face_mask = np.zeros_like(ref_gray)
    ref_head_mask = cv2.fillConvexPoly(ref_face_mask, convexhull, 255)
    ref_face_mask = cv2.bitwise_not(ref_head_mask)
    
    ref_head_noface = cv2.bitwise_and(ref, ref, mask=ref_face_mask)
    add_result = cv2.add(ref_head_noface, new_face)
    
    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
    
    blend_res = cv2.seamlessClone(add_result, ref, ref_head_mask, center_face, cv2.NORMAL_CLONE)
    
    return blend_res


# In[ ]:





# In[ ]:





# In[41]:


global_triangle_indices = get_triangle_indices(ref)


# In[35]:


# src = np.copy(faces[0])
# global_triangle_indices = get_triangulation_indices(ref)
# print_test_imgs(src, False, global_triangle_indices)
# # print_test_imgs(src)


# In[20]:


# unfiltered_face = get_warped_face(ref, src, global_triangle_indices)
# new_face = cv2.GaussianBlur(unfiltered_face, (25,25), cv2.BORDER_DEFAULT)


# In[48]:


faces_array = np.array(faces, dtype=np.uint8)
new_faces_array = np.zeros_like(faces_array)

for idx in range(faces_array.shape[0]):
    unfiltered_face = get_warped_face(ref, faces_array[idx], global_triangle_indices)
    new_face = cv2.GaussianBlur(unfiltered_face, (25,25), cv2.BORDER_DEFAULT)
    new_faces_array[idx] = new_face

choices_female = [0,1,4,10]
choices_male = [2,3,5,6,7,8,9]

avrg = get_average_face(ref, new_faces_array)
cv2.imwrite('./avrg.jpg', cv2.cvtColor(avrg, cv2.COLOR_RGB2BGR))

avrg_female = get_average_face(ref, new_faces_array, choices_female)
cv2.imwrite('./avrg_female.jpg', cv2.cvtColor(avrg_female, cv2.COLOR_RGB2BGR))

avrg_male = get_average_face(ref, new_faces_array, choices_male)
cv2.imwrite('./avrg_male.jpg', cv2.cvtColor(avrg_male, cv2.COLOR_RGB2BGR))


# In[49]:


final = blend(ref, avrg)
final_female = blend(ref, avrg_female)
final_male = blend(ref, avrg_male)

cv2.imwrite('./final.jpg', cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
cv2.imwrite('./final_female.jpg', cv2.cvtColor(final_female, cv2.COLOR_RGB2BGR))
cv2.imwrite('./final_male.jpg', cv2.cvtColor(final_male, cv2.COLOR_RGB2BGR))

