import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

def random_poster(img, p=0.3):
    """
    randomly posterize the image
    
    Parameters
    ----------
    img: PIL image
    p: float
        the event probability, from 0 to 1
    
    Returns
    ----------
    img: PIL image
    """
    if (p < np.random.rand()):
        return img
    
    return ImageOps.posterize(img, np.random.randint(3, 5))


def random_grayscale(img, p=0.2):
    """
    randomly grayscale the image
    
    Parameters
    ----------
    img: PIL image
    p: float
        the event probability, from 0 to 1
    
    Returns
    ----------
    img: PIL image
    """
    if (p < np.random.rand()):
        return img
    
    return img.convert("L").convert("RGB")


def random_enhance(img, p=0.4, max_strength=2.):
    if (p < np.random.rand()):
        return img
    
    strength = max_strength * np.random.rand()
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(strength)
    

def random_erasing(img, p=0.5, max_width=36, max_num=4):
    if (p < np.random.rand()):
        return img
    #determine the number of erasing randomly
    num = np.random.randint(1, max_num)
    
    for i in range(num):
        x_size, y_size = img.size
        x_width = np.random.randint(max_width / 4, max_width)
        y_width = np.random.randint(max_width / 4, max_width)

        x_pos = np.random.randint(0, x_size - max_width)
        y_pos = np.random.randint(0, x_size - max_width)
        noise = Image.new('RGB', (x_width, y_width), (0, 0, 0))
        img.paste(noise, (x_pos, y_pos))
        
    return img


def random_resize(img, bnd_xywhs, p=0.5, max_shrink=0.3):
    """
    randomly resize the image
    
    Parameters
    ----------
    img: PIL image
    bnd_xywhs: list
        [{'x':, 'y':, 'w':, 'h': }, {'x':, 'y':, 'w':, 'h': }..]
    p: float
        the event probability, from 0 to 1
    max_shrink: float
        maximum rate of shrinking
    
    Returns
    ----------
    img: PIL image
    bnd_xywhs: list
    """
    if (p < np.random.rand()):
        return img, bnd_xywhs
    
    x_size, y_size = img.size
    
    #determine the ratio of the image after resizing
    x_resize = 1 - (max_shrink * np.random.rand())
    y_resize = 1 - (max_shrink * np.random.rand())
    
    img = img.resize((int(x_resize * x_size),
                      int(y_resize * y_size)))
    
    #generate gray image and paste resized image
    img_res = Image.new('RGB', (x_size, y_size), (127, 127, 127))
    img_res.paste(img, (0, 0))
    
    for i in range(len(bnd_xywhs)):
        bnd_xywhs[i]["x"] *= x_resize
        bnd_xywhs[i]["y"] *= y_resize
        bnd_xywhs[i]["w"] *= x_resize
        bnd_xywhs[i]["h"] *= y_resize
    
    return img_res, bnd_xywhs


def random_horizontal_flip(img, bnd_xywhs, p=0.5):
    """
    randomly mirror the image
    
    Parameters
    ----------
    img: PIL image
    bnd_xywhs: list
        [{'x':, 'y':, 'w':, 'h': }, {'x':, 'y':, 'w':, 'h': }..]
    p: float
        the event probability, from 0 to 1
    
    Returns
    ----------
    img: PIL image
    bnd_xywhs: list
    """
    if (p < np.random.rand()):
        return img, bnd_xywhs
    #image width
    size = img.size[0]
    img = ImageOps.mirror(img)

    for i in range(len(bnd_xywhs)):
        bnd_xywhs[i]["x"] = size - bnd_xywhs[i]["x"]
    return img, bnd_xywhs


def random_vertical_flip(img, bnd_xywhs, p=0.5):
    """
    randomly flip the image
    
    Parameters
    ----------
    img: PIL image
    bnd_xywhs: list
        [{'x':, 'y':, 'w':, 'h': }, {'x':, 'y':, 'w':, 'h': }..]
    p: float
        the event probability, from 0 to 1
        
    Returns
    ----------
    img: PIL image
    bnd_xywhs: list
    """
    if (p < np.random.rand()):
        return img, bnd_xywhs
    #image height, y...axis_0
    size = img.size[1]
    img = ImageOps.flip(img)

    for i in range(len(bnd_xywhs)):
        bnd_xywhs[i]["y"] = size - bnd_xywhs[i]["y"]
    return img, bnd_xywhs


def random_gaussian_blur(img, p=0.5, s=1.):
    """
    Randomly blur the image
    
    Parameters
    ----------
    img: PIL image
    bnd_xywhs: list
        [{'x':, 'y':, 'w':, 'h': }, {'x':, 'y':, 'w':, 'h': }..]
    p: float
        the event probability, from 0 to 1
    s: float
        maximum blur strength, greater than 0
    Returns
    ----------
    img: PIL image
    """
    if (p < np.random.rand()):
        return img
    filter = ImageFilter.GaussianBlur(s * np.random.rand())
    return img.filter(filter)


def random_sharpness(img, p=0.5):
    if (p < np.random.rand()):
        return img
    return img.filter(ImageFilter.SHARPEN)


def data_aug(img, bnd_xywhs):
    """
    Data Augmentation
    
    Parameters
    ----------
    img: PIL image
    bnd_xywhs: list
        [{'x':, 'y':, 'w':, 'h': }, {'x':, 'y':, 'w':, 'h': }..]

    Returns
    ----------
    img: PIL image
    bnd_xywhs: list
    """
    img = random_grayscale(img)
    img = random_enhance(img)
    img = random_erasing(img, p=0.5, max_width=32)
    img, bnd_xywhs = random_resize(img, bnd_xywhs, p=0.5)
    img, bnd_xywhs = random_horizontal_flip(img, bnd_xywhs, p=0.5)
    img, bnd_xywhs = random_vertical_flip(img, bnd_xywhs, p=0.5)
    img = random_gaussian_blur(img,p=0.5)
    img = random_sharpness(img,p=0.5)
    img = random_poster(img)
    return img, bnd_xywhs
