import random
from utils import *
import torch
import numpy as np
from models.attribute_predictors import attribute_utils,attribute_predictor
from torchvision import transforms
import torchvision
import wandb


class Trainer(object):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.all_attr_list = ['Bald', 'Bangs', 'Goatee', 'Mustache', 'Blurry', 'Pale_Skin',
                              'Wearing_Lipstick']

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
        self.get_initialisations()
        return deformator, deformator_opt, eps_predictor, eps_predictor_opt, eps_predictor_loss, deformator_ranking_loss

    def get_initialisations(self):
        celeba_dataset = CelebADataset(self.config.image_path, transforms.Compose([transforms.ToTensor()]))
        pool_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=self.config.batch_size, num_workers=0,
                                                  pin_memory=True, shuffle=False, drop_last=True)
        initialisation_artifact = wandb.Artifact(str(wandb.run.name) + 'initialisation', type="initialisations")
        extreme_ = wandb.Table(columns=['image_grid', 'direction_idx'])

        for predictor_idx, classifier_name in enumerate(self.all_attr_list):
            predictor = attribute_utils.ClassifierWrapper(classifier_name, ckpt_path=self.config.nvidia_cls_path,
                                                          device=self.config.device)
            predictor.to(self.config.device).eval()
            classifier_scores = []
            for batch_idx, images in enumerate(pool_loader):
                scores = torch.softmax(predictor(images), dim=1)[:, 1]
                classifier_scores.append(scores[0].item())
                classifier_scores.append(scores[1].item())
            classifier_scores_array = np.array(classifier_scores)
            smallest_idx = classifier_scores_array.argsort()[:10]
            largest_idx = classifier_scores_array.argsort()[-10:][::-1]
            indx = smallest_idx.tolist() + largest_idx.tolist()
            image_array = torch.stack([celeba_dataset.__getitem__(idx) for idx in indx])
            grid = torchvision.utils.make_grid(image_array, nrow=3, scale_each=True, normalize=True)
            extreme_.add_data(wandb.Image(grid), self.all_attr_list[predictor_idx])
        initialisation_artifact.add(extreme_, "initialisations")
        wandb.run.log_artifact(initialisation_artifact, aliases=str(0))





