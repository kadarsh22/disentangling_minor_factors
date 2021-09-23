import random
from utils import *
import torch
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

    def train_ours(self, generator, deformator, deformator_opt,eps_predictor, eps_predictor_opt):
        generator.zero_grad()
        deformator.zero_grad()
        eps_predictor_loss = 0
        deformator_ranking_loss = 0
        deformator.ortho_mat = torch.nn.Parameter(deformator.ortho_mat + torch.randn((512, 512)))
        return deformator, deformator_opt, eps_predictor, eps_predictor_opt, eps_predictor_loss, deformator_ranking_loss

