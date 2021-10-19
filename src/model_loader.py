import torch
from models.closedform.utils import load_generator as load_cf_generator
from models.closedform.utils import load_deformator as load_cf_deformator
from models.latentdiscovery.utils import load_generator as load_ld_generator
from models.latentdiscovery.utils import load_deformator as load_ld_deformator
from models.epsilon_predictor import Classifier


def get_model(config):
    if config.initialisation == 'closed_form':
        generator, discriminator = load_cf_generator(config)
        deformator = load_cf_deformator(config)

    elif config.initialisation == 'latent_discovery':
        generator = load_ld_generator(config)
        deformator = load_ld_deformator(config)

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=config.deformator_lr)
    classifier = Classifier(num_dirs=config.num_directions).to(config.device)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=config.eps_predictor_lr)

    return generator, deformator, deformator_opt, discriminator, classifier, classifier_opt
