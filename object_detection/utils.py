import numpy as np


def train_test_split(xs, thres=0.8, seed=42):
    """
    Randomly split xs(list)
    
    Parameters
    ----------
    xs : list
    thres: float
        0 to 1, the ratio of train
    seed : int
        greater than 0, numpy random seed
    
    Returns
    ----------
    train : list
    test : list
    """
    np.random.seed(int(seed))
    randoms = np.random.rand(len(xs))
    
    train, test = list(), list()
    
    for i in range(len(xs)):
        if randoms[i] < thres:
            train.append(xs[i])
        else:
            test.append(xs[i])
            
    return train, test


def resize_xyxy_bnd(bnd_xyxy, ratio=416./224):
    bnd_xyxy = {
       "xmin": bnd_xyxy['xmin'] / ratio,
       "ymin": bnd_xyxy['ymin'] / ratio,
       "xmax": bnd_xyxy['xmax'] / ratio,
       "ymax": bnd_xyxy['ymax'] / ratio
      }
    return bnd_xyxy

def resize_xyxys_bnd(target_obj, ratio=416/224):
    return [resize_xyxy_bnd(bnd['bndbox'], ratio) for bnd in target_obj]


def convert_xyxy_to_xywh(bnd_xyxy):
    bnd_xywh = {
      "x": (bnd_xyxy['xmax'] + bnd_xyxy['xmin']) / 2,
      "y": (bnd_xyxy['ymax'] + bnd_xyxy['ymin']) / 2,
      "w": (bnd_xyxy['xmax'] - bnd_xyxy['xmin']),
      "h": (bnd_xyxy['ymax'] - bnd_xyxy['ymin'])
      }
    return bnd_xywh

def convert_xyxys_to_xywhs(bnd_xyxys):
    """
    from {xmin, xmax, ymin, ymax} to {x, y, w, h}
    
    Parameters
    ----------
    bnd_xyxys : list of dict
        [{xmin:, xmax: , ymin: , ymax: }, ...]
    
    Returns
    ----------
    bnd_xywhs : list of dict
        [{x:, y: , w: , h: }, ...]
    """
    return [convert_xyxy_to_xywh(bnd) for bnd in bnd_xyxys]


def convert_xywh_to_xyxy(bnd_xywh):
    bnd_xyxy = {
      "xmin": bnd_xywh['x'] - bnd_xywh['w'] / 2,
      "ymin": bnd_xywh['y'] - bnd_xywh['h'] / 2,
      'xmax': bnd_xywh['x'] + bnd_xywh['w'] / 2,
      'ymax': bnd_xywh['y'] + bnd_xywh['h'] / 2,      
      }
    return bnd_xyxy

def convert_xywhs_to_xyxys(bnd_xywhs):
    """
    from {x, y, w, h} to {xmin, xmax, ymin, ymax}
    
    Parameters
    ----------
    bnd_xywhs : list of dict
        [{x:, y: , w: , h: }, ...]
    
    Returns
    ----------
    bnd_xyxys : list of dict
        [{xmin:, xmax: , ymin: , ymax: }, ...]
    """
    return [convert_xywh_to_xyxy(bnd) for bnd in bnd_xywhs]