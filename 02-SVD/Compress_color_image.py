import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

color_images = { 
    "cat": img_as_float(data.chelsea()),
    "astronaout": img_as_float(data.astronaut()),
    "coffee": img_as_float(data.coffee())
}

