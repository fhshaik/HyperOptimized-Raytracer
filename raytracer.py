import numpy as np
import math 
from abc import ABC, abstractmethod
from PIL import Image


class Interval:
    def __init__(self, min_val=float('inf'), max_val=float('-inf')):
        self.min = min_val
        self.max = max_val

    def size(self):
        return self.max - self.min

    def contains(self, x):
        return self.min <= x <= self.max

    def surrounds(self, x):
        return self.min < x < self.max
    
    def clamp(self, x):
        if x>self.max:
            return self.max
        if x<self.min:
            return self.min
        return x

    def __str__(self):
        return f"Interval({self.min}, {self.max})"

    @classmethod
    def empty(cls):
        return cls(min_val=float('inf'), max_val=float('-inf'))

    @classmethod
    def universe(cls):
        return cls(min_val=float('-inf'), max_val=float('inf'))
    

class Hittable(ABC):
    @abstractmethod
    def hit(self, ray, interval, rec):
        pass




class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + t*self.direction

class HitRecord:
    def __init__(self, point=None, normal=None, t=None, front_face=None, material = None):
        self.point = point
        self.normal = normal
        self.t = t
        self.front_face = front_face
        self.material = material

    def set_face_normal(self, ray, outward_normal):
        self.front_face = np.dot(ray.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


class Material(ABC):
    @abstractmethod
    def scatter(ray: Ray, hitrec: HitRecord):
        return

class Glass(Material):
    def __init__(self,color, refractive_index):
        self.albedo = color
        self.refractive_index = refractive_index
    def scatter(self, ray, hitrec):
        r = (1.0/self.refractive_index) if hitrec.front_face else self.refractive_index
        direction = ray.direction/np.linalg.norm(ray.direction)
        cos_theta = min(np.dot(-direction, hitrec.normal),1)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        r0 = (1-r)/(1+r)
        r0=r0**2
        r0=r0 + (1-r0)*np.float_power(1-cos_theta, 5)
        if(r*sin_theta>1.0 or r0>np.random.rand()):
            direction = direction - 2*np.dot(ray.direction,hitrec.normal)
        else:
            
            r_perp = r*(direction+cos_theta*hitrec.normal)
            r_parallel = -np.sqrt(np.abs(1.0-r_perp**2))*hitrec.normal
            direction = r_perp+r_parallel

        return Ray(hitrec.point, direction)

    
class Lambertian(Material):
    def __init__(self,color):
        self.albedo = color
    def scatter(self, ray, hitrec):
        direction = np.random.rand(3)
        direction = direction/np.linalg.norm(direction)
        if np.dot(direction,hitrec.normal)<0:
            direction = -direction
        direction += hitrec.normal
        return Ray(hitrec.point, direction)

class Metal(Material):
    def __init__(self, color, fuzz):
        self.albedo = color
        self.fuzz = fuzz
    def scatter(self, ray, hitrec):
        direction = np.random.rand(3)
        direction = direction/np.linalg.norm(direction)
        if np.dot(direction,hitrec.normal)<0:
            direction = -direction
        direction += hitrec.normal

        reflected = ray.direction - 2*np.dot(ray.direction,hitrec.normal)*hitrec.normal
        reflected += self.fuzz*direction
        return Ray(hitrec.point, reflected)
    

class Sphere(Hittable):
    def __init__(self, center, radius, material=None):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, interval, rec):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        h = np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = h**2 - a*c
        
        if discriminant < 0:
            return None
        
        sqrt_discriminant = np.sqrt(discriminant)
        
        root = (-h - sqrt_discriminant) / a
        if not interval.surrounds(root):
            root = (-h + sqrt_discriminant) / a
            if not interval.surrounds(root):
                return None
        
        point = ray.origin + root * ray.direction
        outward_normal = (point - self.center) / self.radius
        
        # Initialize the HitRecord instance with intersection details
        
        rec.point = point
        rec.normal = outward_normal
        rec.t = root
        rec.material = self.material
        
        # Set face normal
        rec.set_face_normal(ray, outward_normal)
        
        return rec
    
class HittableList(Hittable):
    def __init__(self, list_of_hittable):
        self.objects = np.array(list_of_hittable)
    def add(self, hittable):
        self.objects.append(hittable)
    
    def hit(self, ray: Ray, interval:Interval, hitrecord: HitRecord):

        if not self.objects.any():
            return None
        
        # Vectorize the hit method
        vectorized_hit = np.vectorize(lambda obj: obj.hit(ray, interval, HitRecord()))

        # Check for hits
        hits = vectorized_hit(self.objects)

        none_mask = hits == None
        hits = hits[~none_mask]

        if not np.any(hits):
            return None
        # Vectorize to get 't' values
        vectorized_t = np.vectorize(lambda rec: rec.t if rec else np.inf)
        hitvalues = vectorized_t(hits)

        # Find index of minimum 't' value
        min_index = np.argmin(hitvalues)


        return hits[min_index]


class Camera:
    maxdepth = 10
    def __init__(self):
        self.image_height = 720
        self.image_width = 720

        focal_length = 1.0
        viewport_width = 2.0
        viewport_height = 2.0

        viewport_u = np.array([viewport_width,0,0])
        viewport_v = np.array([0,-viewport_height,0])

        self.pixel_delta_u = viewport_u/self.image_width
        self.pixel_delta_v = viewport_v/self.image_width

        viewport_upper_left = -np.array([0,0,focal_length])-viewport_u/2 - viewport_v/2
        pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)
        pixel00_locations = np.tile(pixel00_loc, (self.image_height, self.image_width, 1))
        indices_i = np.repeat(np.tile(np.arange(self.image_width), (self.image_height, 1))[:, :, np.newaxis], 3, axis=2)
        indices_j = np.repeat(np.tile(np.arange(self.image_height)[:, np.newaxis],  (1,self.image_width))[:, :, np.newaxis], 3, axis=2)
        #finally setting up our camera location through this
        self.pixel_centers = pixel00_locations+indices_i*self.pixel_delta_u+ indices_j*self.pixel_delta_v
    
    def aliased_ray(self,vector, hitabbleObjects):
        samples_per_pixel = 10
        a = np.random.uniform(-0.5, 0.5, size=samples_per_pixel)
        b = np.random.uniform(-0.5, 0.5, size=samples_per_pixel)
        c = np.zeros_like(a)
        samples = np.column_stack((a, b, c))*(self.pixel_delta_u + self.pixel_delta_v)

        
        vector = np.tile(vector, (samples_per_pixel,1))
        samples+=vector
        return np.sum(np.apply_along_axis(lambda x: self.raytrace(Ray(np.array([0,0,0]),x),hitabbleObjects,self.maxdepth), 1,samples),0)/samples_per_pixel



        
    
    def raytrace(self, ray: Ray, hitabbleObjects, depth):
        if depth<=0:
            return np.array([0,0,0])
        depth-=1
        hitlist = HittableList(hitabbleObjects)
        hit = hitlist.hit(ray,  Interval(0.01,1000), HitRecord())

        if(hit):
            t = hit.t
            if(t>0):
                scattered = hit.material.scatter(ray, hit)
                return hit.material.albedo * self.raytrace(scattered, hitabbleObjects, depth)
                # direction = np.random.rand(3)
                # direction = direction/np.linalg.norm(direction)
                # if np.dot(direction,hit.normal)<0:
                #     direction = -direction
                # direction += hit.normal
                #N = t*vector -np.array([0,0,-1])
                #N = (0.5*np.array((N[0]+1, N[1]+1, N[2]+1)))*256
                #return 0.5 * self.ray(direction,hitabbleObjects, depth) 
                
                return 

        unit_direction = ray.direction/np.linalg.norm(ray.direction)
        a = 0.5*(unit_direction[1] + 1.0)
        return ((1.0-a)*np.array([1.0,1.0,1.0]) + a*np.array([0.5, 0.7, 1.0]))
    
    def render(self, hittableObjects):
        pixel_array = np.apply_along_axis(lambda vector: self.aliased_ray(vector, hittableObjects), 2, self.pixel_centers.astype(np.float32))
        np.sqrt(pixel_array)
        return np.clip(pixel_array, 0, 1)

    


#render

  
# def ray(vector):
#     #t = hit_sphere(np.array([0,0,-1]), 0.5, vector)
#     if(t>0):
#         N = t*vector -np.array([0,0,-1])
#         return (0.5*np.array((N[0]+1, N[1]+1, N[2]+1)))*255
    
#     unit_direction = vector/np.linalg.norm(vector)
#     a = 0.5*(unit_direction[1] + 1.0)
#     return 255*((1.0-a)*np.array([1.0,1.0,1.0]) + a*np.array([0.5, 0.7, 1.0]))

raytracer = Camera()

image_array = raytracer.render([Sphere(np.array([0,0,-2]), 0.5, Metal(np.array([0.7,0.9,0.6]), 0.1)), Sphere(np.array((0,-1000.5,-1)), 1000, Metal(np.array([0.5,0.7,0.8]), 0.5)), Sphere(np.array([0.2,0.7,-1]), 0.2,Metal(np.array([0.7,0.9,0.5]), 0.6)), Sphere(np.array([-0.3,0.4,-3]), 2,Glass(np.array([1.2,0.8,0.8]), 2.5)), Sphere(np.array([-0.3,0.4,-1.3]), 0.5,Glass(np.array([1,1,1]), 0.2))])
image_array*=255
print(image_array)
image = Image.fromarray(image_array.astype(np.uint8))

# Save the image
image.save('output_image.jpg')



image.show()