import os
import pickle
import random

import gymnasium as gym
import imageio
import numpy as np
import tqdm
from pyvirtualdisplay.display import Display
from tqdm.notebook import tqdm

if __name__ == "__main__":
    virtual_display = Display(visible=False, size=(1400, 900))
    virtual_display.start()

    virtual_display.stop()
