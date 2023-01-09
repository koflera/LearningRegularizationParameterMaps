from matplotlib import animation
from typing import Optional
import matplotlib.pyplot as plt
import tempfile
import numpy as np


def save_animation(what, filename:Optional[str]=None, cmap:str="inferno", vmin:float=None, vmax:float=None)->str:
    """
    Creates an animation and saves it as gif.

    Args:
        what (ArrayLike): Data. Will be animated along last dimension
        filename (Optional[str], optional): if set, output will be saved to filename, else to a tempfile
        cmap (str, optional): coloarmap. Defaults to "inferno".
        vmin (float, optional): Defaults to 1%-percentile.
        vmax (float, optional): Defaults to 99%-percentile.

    Returns:
        str: filename of saved animation
    """

    if filename is None:
        filename = tempfile.mkstemp(".gif", "anim_")[1]

    fig = plt.figure(dpi=100, figsize=(8, 8))
    ims = []
    if vmin is None:
        vmin = np.percentile(np.asarray(what[what.shape[0]//4:-what.shape[0]//4,what.shape[1]//4:-what.shape[1]//4]),1)
    if vmax is None:
        vmax = np.percentile(np.asarray(what[what.shape[0]//4:-what.shape[0]//4,what.shape[1]//4:-what.shape[1]//4]),99)
    for i in range(what.shape[-1]):
        img = what[..., i]
        im = plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.axis("off")
        plt.tight_layout()
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,)
    ani.save(filename, dpi=100)
    plt.close()
    return filename
