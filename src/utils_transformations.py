import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import random
import imutils
import torchvision
from transforms import RGBTransform 


#### The predefined transformations

c_src_list = np.array([
    [[0, 0.2],
    [1, 0],
    [1, 1],
    [0.8, 0.4],
    [0.4, 0.9],
    [0, 1]], 
    
    [[0, 0],
    [1, 0],
    [1, 1],
    [0.7, 0.7],
    [0.5, 0.6],
    [0, 1]], 
    
    [[0, 0.1],
    [1, 0],
    [1, 1],
    [0.3, 0.3],
    [0.7, 0.8],
    [0, 1]],
    
    [[0, 0],
    [1, 0],
    [1, 1],
    [0.2, 0.7],
    [0.7, 0.2],
    [0, 1]],
    
    [[0, 0],
    [1, 0],
    [1, 1],
    [0.3, 0.4],
    [0.8, 0.7],
    [0, 1]], 
    
    [[0, 0],
    [1, 0],
    [1, 1],
    [0.7, 0.7],
    [0.5, 0.6],
    [0, 1]] 
    
])

c_dst_list = np.array([
    [[0.1, 0.2],
    [0.9, 0.2],
    [0.86, 0.9],
    [0.6, 0.6],
    [0.4, 0.5],
    [0.1, 1]],
    
    [[0, 0.2],
    [1, 0.2],
    [1, 0.9],
    [0.8, 0.8],
    [0.5, 0.7],
    [0.3, 1]], 
    
    [[0.1, 0],
    [0.9, 0.2],
    [0.7, 0.9],
    [0.2, 0.4],
    [0.5, 0.6],
    [0, 0.9]], 

    [[0, 0],
    [1, 0],
    [0.8, 1],
    [0.3, 0.7],
    [0.5, 0.3],
    [0, 0.8]],
    
    [[0.1, 0.2],
    [1, 0.1],
    [0.8, 0.9],
    [0.2, 0.5],
    [0.6, 0.8],
    [0.1, 0.9]],
    
    [[0, 0],
    [1, 0.1],
    [1, 0.9],
    [0.8, 0.7],
    [0.5, 0.6],
    [0.1, 1]] 
])

## Source code from : https://github.com/cheind/py-thin-plate-spline
class TPS:
    @staticmethod
    def fit(c, lambd=0., reduced=False):
        n = c.shape[0]

        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32) * lambd

        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]

        v = np.zeros(n + 3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n + 3, n + 3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = np.linalg.solve(A, v)  # p has structure w,a
        return theta[1:] if reduced else theta

    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r ** 2 * np.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = np.concatenate((-np.sum(w, keepdims=True), w))
        b = np.dot(U, w)
        return a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + b

def uniform_grid(shape):
    '''Uniform grid coordinates.

    Params
    ------
    shape : tuple
        HxW defining the number of height and width dimension of the grid
    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range.
    '''

    H, W = shape[:2]
    c = np.ones((H, W, 2))
    c[..., 0] = np.linspace(0, 1, W, dtype=np.float32)
    c[..., 1] = np.expand_dims(np.linspace(0, 1, H, dtype=np.float32), -1)

    return c

def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst

    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))

    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)

    return np.stack((theta_dx, theta_dy), -1)


def tps_grid(theta, c_dst, dshape):
    ugrid = uniform_grid(dshape)
    reduced = c_dst.shape[0] + 2 == theta.shape[0]

    dx = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 0]).reshape(dshape[:2])
    dy = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 1]).reshape(dshape[:2])
    dgrid = np.stack((dx, dy), -1)

    grid = dgrid + ugrid

    return grid  # H'xW'x2 grid[i,j] in range [0..1]

def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.

    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    sshape : tuple
        Height and width of source image in pixels.
    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''

    mx = (grid[:, :, 0] * sshape[1]).astype(np.float32)
    my = (grid[:, :, 1] * sshape[0]).astype(np.float32)

    return mx, my

def warp_image(img, c_src, c_dst, angle, border_value=0, dshape=None):
    rows = img.shape[0]
    cols = img.shape[1]
    
    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, angle, 1)
    rotated_image = cv2.warpAffine(img.numpy(), M, (cols, rows), borderValue=(border_value,border_value,border_value))         
    dshape = dshape or img.shape
    theta = tps_theta_from_points(c_src  , c_dst, reduced=True)
    grid = tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps_grid_to_remap(grid, img.shape)
    
    return  cv2.remap(rotated_image, mapx, mapy, cv2.INTER_CUBIC, borderValue=(border_value,border_value,border_value))

# appearance transformations
## HSV shift + RGB shift  
import colorsys

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = h + hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(np.clip(shift_hue(arr, hue/360.).astype('uint8'), 0, 255), 'RGBA')
    
    new_img = new_img.convert("RGB")
    new_img = RGBTransform().mix_with((np.random.randint(255), np.random.randint(255), np.random.randint(255)),factor=.50).applied_to(new_img)  
    return new_img

def change_appearance(img):
    img = img.numpy()
    brightness = np.random.randint(30)
    contrast = np.random.randint(50)
    hue = np.random.randint(150, 300)
    
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)    
    img = Image.fromarray(img, 'RGB')
    img = colorize(img, hue)
    return  torchvision.transforms.ToTensor()(img)