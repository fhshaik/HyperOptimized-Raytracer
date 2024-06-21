import numpy as np
import math 
from PIL import Image
image_height = 256
image_width = 256
image_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


# Display the image

focal_length = 1.0
viewport_width = 2.0
viewport_height = 2.0

viewport_u = np.array([viewport_width,0,0])
viewport_v = np.array([0,-viewport_height,0])

pixel_delta_u = viewport_u/image_width
pixel_delta_v = viewport_v/image_width

viewport_upper_left = -np.array([0,0,focal_length])-viewport_u/2 - viewport_v/2
pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)
i_indices = np.arange(image_width)
j_indices = np.arange(image_height)
pixel00_locations = np.tile(pixel00_loc, (image_height, image_width, 1))
indices_i = np.repeat(np.tile(np.arange(image_width), (image_height, 1))[:, :, np.newaxis], 3, axis=2)
indices_j = np.repeat(np.tile(np.arange(image_height)[:, np.newaxis],  (1,image_width))[:, :, np.newaxis], 3, axis=2)


# Create a 3D array with shape (256, 256, 3)

print(indices_j.shape)
print(indices_i.shape)
pixel_centers = pixel00_locations+indices_i*pixel_delta_u+ indices_j*pixel_delta_v
print(pixel_centers)  # Output: (256, 256, 3)
#render

def hit_sphere(center, radius, vector):
    a = np.dot(vector,vector)
    b = -2*np.dot(vector, center)
    c= np.dot(center, center)-radius*radius
    discriminant = b*b - 4*a*c
    if discriminant<0:
        return -1.0
    else:
        return (-b - math.sqrt(discriminant)) / (2.0*a)

def ray(vector):
    t = hit_sphere(np.array([0,0,-1]), 0.5, vector)
    if(t>0):
        N = t*vector -np.array([0,0,-1])
        return (0.5*np.array((N[0]+1, N[1]+1, N[2]+1)))*255
    
    unit_direction = vector/np.linalg.norm(vector)
    a = 0.5*(unit_direction[1] + 1.0)
    return 255*((1.0-a)*np.array([1.0,1.0,1.0]) + a*np.array([0.5, 0.7, 1.0]))

pixel_array = np.apply_along_axis(ray, 2, pixel_centers.astype(np.float32))

normalized_array = np.clip(pixel_array, 0, 255).astype(np.uint8)

print(normalized_array)
image = Image.fromarray(normalized_array)
print(ray(np.array([0.98046875,0.98046875, -1])))
# Save the image
image.save('output_image.jpg')


np.vectorize()

image.show()