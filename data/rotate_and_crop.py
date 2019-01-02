import numpy as np
from PIL import Image

def perp(a):
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2):
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1

def rotate_and_crop(img, deg, same_size=False, interp=Image.BICUBIC):
    # let the four corners of a rectangle to be ABCD, clockwise
    if deg == 0:
        return img

    w, h = img.size

    A = np.array([-w/2, h/2])
    B = np.array([w/2, h/2])
    C = np.array([w/2, -h/2])
    D = np.array([-w/2, -h/2])

    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]]).T

    Arot = np.dot(A, R)
    Brot = np.dot(B, R)
    if deg > 0:
        X = seg_intersect(A, C, Arot, Brot)
        offset = X - A
        offset[1] = -offset[1]
    else:
        X = seg_intersect(B, D, Arot, Brot)
        offset = B - X
    
    if same_size:
        wh_org = np.array([w, h])
        wh = np.ceil(np.divide(np.square(wh_org), wh_org - 2*offset)).astype(np.int32)
        offset = (wh - wh_org)/2
        img = img.resize(wh, interp)
        w = wh[0]
        h = wh[1]
    else:
        offset = np.ceil(offset)
    img = img.rotate(deg, interp)
    return img.crop(
            (offset[0], 
                offset[1],
                w - offset[0],
                h - offset[1])
        )

