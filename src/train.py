import os.path
import random
from utils import *
import torch
import numpy as np
from models.attribute_predictors import attribute_utils, attribute_predictor
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import wandb
import logging


class Trainer(object):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.all_attr_list = ['Bald', 'Bangs', 'Goatee', 'Mustache', 'Pale_Skin',
                              'Wearing_Lipstick']
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_ours(self, target_generator, target_deformator, target_deformator_opt, transformation_learning_net):

        target_deformator_opt.zero_grad()
        z = torch.randn(self.config.batch_size, self.config.latent_dim).to(self.config.device)
        target_indices, shifts, z_shift = self.make_shifts(self.config.latent_dim)
        ref_images = target_generator(z)
        z_shift = target_deformator(z_shift)
        peturbed_images = target_generator(z, shift= z_shift)
        logits, shift_prediction = transformation_learning_net(ref_images,  peturbed_images)
        logit_loss = self.config.label_weight * self.cross_entropy(logits, target_indices)
        shift_loss = self.config.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))
        loss = logit_loss + shift_loss
        loss.backward()
        target_deformator_opt.step()

        return target_deformator, target_deformator_opt, logit_loss.item(), shift_loss.item()

    def train_transformation_learning_net(self, source_generator, source_deformator, transformation_learning_net,
                                          transformation_learning_net_opt):
        transformation_learning_net.train().to(self.config.device)
        source_generator.eval()
        source_deformator.eval()
        training_logit_loss = []
        training_shift_loss = []

        for step in range(self.config.num_steps):
            transformation_learning_net_opt.zero_grad()
            z = torch.randn(self.config.batch_size, self.config.latent_dim).to(self.config.device)
            target_indices, shifts, z_shift = self.make_shifts(self.config.latent_dim)
            ref_images = source_generator(z)
            z_shift = source_deformator(z_shift)
            peturbed_images = source_generator(z, shift= z_shift)
            logits, shift_prediction = transformation_learning_net(ref_images,  peturbed_images)
            logit_loss = self.config.label_weight * self.cross_entropy(logits, target_indices)
            shift_loss = self.config.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))
            loss = logit_loss + shift_loss
            training_logit_loss.append(logit_loss.item())
            training_shift_loss.append(shift_loss.item())
            loss.backward()
            transformation_learning_net_opt.step()

            if step % 50 == 0:
                training_shift_loss_avg = sum(training_shift_loss) / len(training_shift_loss)
                training_logit_loss_avg = sum(training_logit_loss) / len(training_logit_loss)
                percent = self.validate_transformation_learning_net(source_generator, source_deformator, transformation_learning_net)
                logging.info("step : %d / %d shift loss : %.3f logit loss  %.3f " % (
                    step, self.config.num_steps, training_shift_loss_avg, training_logit_loss_avg))
                wandb.log({'step': step + 1, 'accuracy': percent, 'shift_loss': training_shift_loss_avg,
                           'logit_loss': training_logit_loss_avg})

        return transformation_learning_net

    def validate_transformation_learning_net(self, source_generator, source_deformator, transformation_learning_net):

        n_steps = 100
        percents = torch.empty([n_steps])
        for step in range(n_steps):
            z = torch.randn(self.config.batch_size, self.config.latent_dim).to(self.config.device)
            target_indices, shifts, basis_shift = self.make_shifts(self.config.latent_dim)

            imgs = source_generator(z)
            imgs_shifted = source_generator(z,shift = source_deformator(basis_shift))

            logits, _ = transformation_learning_net(imgs, imgs_shifted)
            percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

        return percents.mean()

    def make_shifts(self, latent_dim):
        target_indices = torch.randint(0, self.config.num_directions, [self.config.batch_size]).to(self.config.device)
        if self.config.shift_distribution == 'normal':
            shifts = torch.randn(target_indices.shape).to(self.config.device)
        elif self.config.shift_distribution == 'uniform':
            shifts = 2.0 * torch.rand(target_indices.shape).to(self.config.device) - 1.0

        shifts = self.config.epsilon * shifts
        shifts[(shifts < self.config.min_shift) & (shifts > 0)] = self.config.min_shift
        shifts[(shifts > -self.config.min_shift) & (shifts < 0)] = -self.config.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        z_shift = torch.zeros([self.config.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift
