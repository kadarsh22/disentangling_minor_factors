import random
from utils import *
import torch.nn as nn
import numpy as np


class Trainer(object):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_ours(self, generator, deformator, deformator_opt):
        generator.zero_grad()
        deformator.zero_grad()
        raise NotImplementedError

