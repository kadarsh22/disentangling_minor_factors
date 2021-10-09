import torch
from models.closedform.utils import load_generator as load_cf_generator
from models.closedform.utils import load_deformator as load_cf_deformator
from models.latentdiscovery.utils import load_generator as load_ld_generator
from models.latentdiscovery.utils import load_deformator as load_ld_deformator
from models.epsilon_predictor import Classifier
from models import domain_generator


def get_model(config):
    if config.initialisation == 'closed_form':
        generator = load_cf_generator(config)
        deformator = load_cf_deformator(config)

    elif config.initialisation == 'latent_discovery':
        generator = load_ld_generator(config)
        deformator = load_ld_deformator(config)

    generator = domain_generator.define_generator('stylegan2', 'celebahq')

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=config.deformator_lr)
    classifier = Classifier(num_dirs=config.num_directions).to(config.device)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=config.eps_predictor_lr)

    return generator, deformator, deformator_opt, classifier, classifier_opt
