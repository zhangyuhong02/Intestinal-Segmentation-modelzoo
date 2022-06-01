import sys
sys.path.append('/home/aistudio/external-libraries')
if './external-libraries' not in sys.path:
    sys.path.append('./external-libraries')
import numpy as np
import nibabel as nib
from util import * 
from tqdm import tqdm
import os
if './external-libraries' not in sys.path:
    sys.path.append('./external-libraries')
import matplotlib.pyplot as plt
from util import * 
from PIL import Image
from paddle.vision.transforms import functional as F
