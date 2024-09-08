#Using for running really "Vision" Web for presentation 
#pakai def ... (): untuk tiap isi page
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from typing import Union, List, Tuple
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import joblib
import os
import glob

#Pre processing page
image = 'image.png'