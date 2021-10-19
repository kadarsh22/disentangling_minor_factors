import os.path
import random
from utils import *
import torch
import numpy as np
import torch.nn as nn
from models.attribute_predictors import attribute_utils, attribute_predictor
import torchvision
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.all_attr_list = ['Bald', 'Bangs', 'Goatee', 'Mustache', 'Pale_Skin',
                              'Wearing_Lipstick', 'Pose']
        self.classifier_loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_ours(self, generator, supervision_images, deformator, deformator_opt, classifier, classifier_opt, seed):

        classifier.train()
        classifier_opt.zero_grad()
        attribute_idx = torch.randint(0, len(self.all_attr_list), (self.config.batch_size, 1)).view(-1)
        type_idx = torch.randint(0, 2, (self.config.batch_size, 1)).view(-1)
        image_idx = attribute_idx * 2 + type_idx
        one_shot_images = supervision_images[image_idx.view(-1)]
        pred = classifier(one_shot_images)
        pred_logits = torch.gather(pred.view(self.config.batch_size, -1), dim=1,
                                   index=torch.LongTensor(attribute_idx).view(-1, 1).to(self.config.device)).view(-1)
        classifier_loss = self.classifier_loss(pred_logits, type_idx.float().to(self.config.device))
        classifier_loss.backward()
        classifier_opt.step()

        classifier.eval()
        generator.generator.zero_grad()
        classifier.zero_grad()
        deformator_opt.zero_grad()
        z = torch.randn(self.config.batch_size, self.config.latent_dim).cuda()
        target_indices, type_idx, w_shift = self.make_shifts(deformator.input_dim)
        w = generator.generator.gen.style(z)
        images = generator.generator(w + deformator(w_shift))
        pred = classifier(images)
        pred_logits = torch.gather(pred.view(-1, self.config.num_directions), dim=1,
                                   index=torch.LongTensor(target_indices).view(-1, 1).to(self.config.device))
        deformator_loss = self.classifier_loss(pred_logits.view(-1), type_idx.float().to(self.config.device))
        deformator_loss.backward()
        deformator_opt.step()

        return deformator, deformator_opt, classifier, classifier_opt, deformator_loss, classifier_loss

    def train_classifier(self, generator, discriminator, classifier, classifier_opt):

        classifier.zero_grad()

        return generator, discriminator, classifier, classifier_opt

    def get_initialisations(self, generator, seed):

        z_full = torch.randn(self.config.supervision_pool_size, self.config.latent_dim).to(self.config.device)
        os.makedirs(os.path.join(self.config.result_path, "generated_images"), exist_ok=True)
        torch.save(z_full, os.path.join(self.config.result_path, "generated_images", "z_generated.pth"))
        new_dataset = NoiseDataset(z_full)
        z_loader = torch.utils.data.DataLoader(new_dataset, batch_size=self.config.batch_size,
                                               num_workers=0,
                                               pin_memory=False, shuffle=False, drop_last=False)
        initialisation_artifact = wandb.Artifact(str(wandb.run.name) + 'initialisation', type="initialisations")
        extreme_ = wandb.Table(columns=['image_grid', 'direction_idx'])

        ordered_idx = {}
        if not self.config.load_pretrained_z:
            for predictor_idx, classifier_name in enumerate(self.all_attr_list):
                predictor = attribute_utils.ClassifierWrapper(classifier_name, ckpt_path=self.config.nvidia_cls_path,
                                                              device=self.config.device)
                predictor.to(self.config.device).eval()
                classifier_scores = []
                for batch_idx, z in enumerate(z_loader):
                    images = torch.clamp(F.avg_pool2d(generator(z), 4, 4), min=-1, max=1)
                    scores = torch.softmax(predictor(images.to(self.config.device)), dim=1)[:, 1]
                    classifier_scores = classifier_scores + scores.detach().tolist()
                print(len(classifier_scores))
                classifier_scores_array = np.array(classifier_scores)
                ordered_idx[str(classifier_name)] = classifier_scores_array
                smallest_idx = classifier_scores_array.argsort()[:20]
                largest_idx = classifier_scores_array.argsort()[-20:][::-1]
                print(classifier_name)
                print("-------smallest_idx-------")
                print(smallest_idx)
                print("-------largest_idx --------")
                print(largest_idx)
                indx = smallest_idx.tolist() + largest_idx.tolist()
                image_array = torch.stack([torch.clamp(F.avg_pool2d(generator.generator(
                    generator(z_full[idx].view(-1, self.config.latent_dim)).cuda()).detach(), 4, 4),
                                                       min=-1, max=1) for idx in indx])
                image_array = image_array.view(-1, 3, 256, 256)
                grid = torchvision.utils.make_grid(image_array, nrow=10, scale_each=True, normalize=True)
                extreme_.add_data(wandb.Image(grid), self.all_attr_list[predictor_idx])
            os.makedirs(os.path.join(self.config.result_path, "sorted_images"), exist_ok=True)
            torch.save(ordered_idx, os.path.join(self.config.result_path, "sorted_images", "ordered_idx.pth"))
            initialisation_artifact.add(extreme_, "initialisations")
            wandb.run.log_artifact(initialisation_artifact, aliases=str(0))
        else:
            supervised_z = torch.load('pretrained_models/supervision_images_pool/z_generated.pth')
            supervised_idx = [1369, 4016, 1897, 3659, 4614, 4570, 2384, 3535, 829, 2019, 1352, 3041, 4406, 1959]
            image_array = torch.stack([torch.clamp(F.avg_pool2d(generator.generator(
                generator.generator.gen.style(supervised_z[idx].view(-1, self.config.latent_dim)).cuda()).detach(), 4,
                                                                4),
                                                   min=-1, max=1) for idx in supervised_idx])
            image_array = image_array.view(-1, 3, 256, 256)
            grid = torchvision.utils.make_grid(image_array, nrow=2, scale_each=True, normalize=True)
            images = wandb.Image(grid, caption="Supervision images")
            wandb.log({"supervision_images": images})

            supervised_images = torch.stack([generator(supervised_z[idx].view(-1, self.config.latent_dim)).cuda().detach() for
                                             idx in supervised_idx])
            return supervised_images.detach().squeeze(1)

    def make_shifts(self, latent_dim):
        target_indices = torch.randint(0, len(self.all_attr_list), (self.config.batch_size,))
        type_idx = torch.randint(0, 2, (self.config.batch_size,))

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
            if type_idx[i] == 0:
                z_shift[i][index] += -abs(val)
            else:
                z_shift[i][index] += abs(val)

        return target_indices, type_idx, z_shift
