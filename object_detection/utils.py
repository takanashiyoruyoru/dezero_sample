import numpy as np
import matplotlib.pyplot as plt


def train_test_split(xs, thres=0.8, seed=42):
    """
    xs(list)をランダムに分割して出力
    
    Parameters
    ----------
    xs: list
    thres: float
        0 to 1, the ratio of train
    seed: int
        greater than 0, numpy random seed
    """
    np.random.seed(seed)
    randoms = np.random.rand(len(xs))
    
    train, test = list(), list()
    
    for i in range(len(xs)):
        if randoms[i] < thres:
            train.append(xs[i])
        else:
            test.append(xs[i])
            
    return train, test


def convert_img_for_mat(img):
    """
    VGG16のpreprocessの逆の処理,
    pltで表示可能な画像に変換
    
    Parameters
    ----------
    img: ndarray
        channel first
    
    Returns
    ----------
    img: ndarray
        channel last
    """
    #channel first to channel last
    img = np.transpose(img, (1, 2, 0))
    img += np.array([103.939, 116.779, 123.68])
    
    #RGBの順番を入れ替える
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255)
    
    return np.int32(img)


def write_bndbox(img, bnd_xyxys, scores=None, clip=True, ticks=True):
    """
    画像にbouding boxに重ねて表示
    
    Parameters
    ----------
    img: ndarray or PIL
    bnd_xyxys: list
    scores: list
        bounding boxの信頼度を表示
    clip: bool
        True ...bouding boxが画像内に収まるように表示
        False...本来のbouding boxを表示
    ticks: bool
        xyの座標を表示するか否か
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
        
        #bouding boxの描画
        plt.plot([xmin, xmin], [ymin, ymax], c="red")
        plt.plot([xmax, xmax], [ymin, ymax], c="red")
        plt.plot([xmin, xmax], [ymin, ymin], c="red")
        plt.plot([xmin, xmax], [ymax, ymax], c="red")
        
        if scores is not None:
            plt.text(s=str((np.round(100 * scores[i], 1))),
                     x=xmin + 3, y=ymax - 4, 
                     fontsize=9, c="white",
                     horizontalalignment="left",
                     verticalalignment="bottom",
                     bbox=dict(facecolor="red", edgecolor="red"))
            
    plt.xlim(0, img.shape[0] - 1)
    plt.ylim(img.shape[1] - 1, 0)
    
    if ticks is not True:
        plt.xticks([])
        plt.yticks([])
    plt.show()

    
def show_heatmap(img, pred, ticks=True):
    """
    画像にheatmapを重ねて表示
    
    Parameters
    ----------
    img: ndarray or PIL
    pred: ndarray
        2 dim
    ticks: bool
        xyの座標を表示するか否か
    """ 
    img = np.array(img)
    pred = np.array(pred)
    
    #axis_1...x, axis_0...y
    x_repeat = int(img.shape[1] / pred.shape[1])
    y_repeat = int(img.shape[0] / pred.shape[0])
    
    #heatmapの作成
    pp = pred.repeat(y_repeat, axis=0).repeat(x_repeat, axis=1)
    
    plt.imshow(img[:, :, 0], cmap="gray")
    plt.imshow(pp, cmap="Reds", alpha=0.6, vmax=1., vmin=0)
    
    if ticks is not True:
        plt.xticks([])
        plt.yticks([])
    plt.show()