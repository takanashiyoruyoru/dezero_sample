import numpy as np
import matplotlib.pyplot as plt

def convert_img_for_mat(img):
    """
    Reverse process of VGG16 preprocess
    we can show the returned image of this function by matplotlib
    
    Parameters
    ----------
    img : ndarray
        3dim, channel first
    
    Returns
    ----------
    img : ndarray
        3 dim, channel last
    """
    #channel first to channel last
    img = np.transpose(img, (1, 2, 0))
    img += np.array([103.939, 116.779, 123.68])
    
    #Swap the order of RGB
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255)
    
    return np.int32(img)


def write_bndbox(img, bnd_xyxys, scores=None, clip=True, ticks=True):
    """
    Overlay bouding boxes on a image
    
    Parameters
    ----------
    img : ndarray or PIL
    bnd_xyxys : list of dict
        [{xmin:, xmax: , ymin: , ymax: }, ...]
    scores : list of float
        the confidences of bounding boxes
    clip : bool
        True ...Displayed so that the bouding boxes fit within the image
        False...Show original bouding boxes
    ticks : bool
        whether xy coordinates are displayed or not
    """
    plt.imshow(img)
    img = np.array(img)
    
    for i, bnd in enumerate(bnd_xyxys):
        if clip is True:
            xmin = np.clip(bnd['xmin'], 1, img.shape[0] - 1)
            ymin = np.clip(bnd['ymin'], 1, img.shape[1] - 1)
            xmax = np.clip(bnd['xmax'], 1, img.shape[0] - 1)
            ymax = np.clip(bnd['ymax'], 1, img.shape[1] - 1)
        else:
            xmin = bnd['xmin']
            ymin = bnd['ymin']
            xmax = bnd['xmax']
            ymax = bnd['ymax']
        
        #Drawing a bouding box
        plt.plot([xmin, xmin], [ymin, ymax], c="red")
        plt.plot([xmax, xmax], [ymin, ymax], c="red")
        plt.plot([xmin, xmax], [ymin, ymin], c="red")
        plt.plot([xmin, xmax], [ymax, ymax], c="red")
        
        if scores is not None:
            #Drowing a score within the bounding box
            plt.text(s=str((np.round(100 * scores[i], 1))),
                     x=xmin + 3, y=ymax - 4, 
                     fontsize=9, c="white",
                     horizontalalignment="left",
                     verticalalignment="bottom",
                     bbox=dict(facecolor="red", edgecolor="red"))
    
    if clip is True:
        plt.xlim(0, img.shape[0] - 1)
        plt.ylim(img.shape[1] - 1, 0)
    
    if ticks is not True:
        plt.xticks([])
        plt.yticks([])
    plt.show()

    
def show_heatmap(img, pred, ticks=True):
    """
    Overlay heatmap on image
    If the resolution of the pred is smaller than the img, upsampling the pred.
    
    Parameters
    ----------
    img : ndarray or PIL
    pred : ndarray
        heatmap (2-d image for heatmap)
    ticks : bool
        whether xy coordinates are displayed or not
    """ 
    #convert ndarray
    img = np.array(img)
    pred = np.array(pred)
    
    #axis_1...x, axis_0...y
    x_repeat = int(img.shape[1] / pred.shape[1])
    y_repeat = int(img.shape[0] / pred.shape[0])
    
    #Upsampling the heatmap
    pp = pred.repeat(y_repeat, axis=0).repeat(x_repeat, axis=1)
    
    plt.imshow(img[:, :, 0], cmap="gray")
    plt.imshow(pp, cmap="Reds", alpha=0.6, vmax=1., vmin=0)
    
    if ticks is not True:
        plt.xticks([])
        plt.yticks([])
    plt.show()