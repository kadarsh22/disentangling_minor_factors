from model_loader import get_model
from train import Trainer
from saver import Saver
from utils import *
import wandb
from evaluation import Evaluator
from visualisation import Visualiser


def run_training_wrapper(config, seed, logger, perf_logger):
    directories = list_dir_recursively_with_ignore('.',
                                                   ignores=['checkpoint.pt', '__pycache__', 'wandb', 'venv', '.idea',
                                                            'data', 'pretrained_models', 'data',
                                                            'results', '.git', '.gitignore', 'artifacts',
                                                            'pretrained_models.tar.xz', 'pretrained_models'])
    filtered_dirs = []
    for file in directories:
        x, y = file
        filtered_dirs.append((x, y))
    files = [(f[0], os.path.join(os.getcwd(), 'results', wandb.run.name, "src_copy", f[1])) for f in filtered_dirs]

    copy_files_and_create_dirs(files)

    models = get_model(config)
    model_trainer = Trainer(config)
    saver = Saver(config)
    evaluator = Evaluator(config)
    visualiser = Visualiser(config)

    generator, deformator, deformator_opt, classifier, classifier_opt = models
    deformator.train()
    generator.generator.eval()
    classifier_loss_list = []
    deformator_loss_list = []
    supervision_images = model_trainer.get_initialisations(generator, seed)
    # saver.load_model((deformator,deformator_opt))
    for iteration in range(config.num_iterations):
        deformator, deformator_opt, classifier, classifier_opt, deformator_loss, classifier_loss = \
            model_trainer.train_ours(generator, supervision_images, deformator, deformator_opt, classifier,
                                     classifier_opt)
        classifier_loss_list.append(classifier_loss.item())
        deformator_loss_list.append(deformator_loss.item())

        if iteration % config.logging_freq == 0 and iteration != 0:
            classifier_loss_avg = sum(classifier_loss_list) / len(classifier_loss_list)
            deformator_loss_avg = sum(deformator_loss_list) / len(deformator_loss_list)
            logger.info("step : %d / %d eps predictor loss : %.4f Deformator_loss  %.4f " % (
                iteration, config.num_iterations, classifier_loss_avg, deformator_loss_avg))
            wandb.log({'iteration': iteration + 1, 'deformator_loss': deformator_loss_avg,
                       'classifier_loss': classifier_loss_avg})

        if iteration % config.saving_freq == 0 and iteration != 0:
            params = (deformator, deformator_opt, classifier, classifier_opt)
            perf_logger.start_monitoring("Saving Model for iteration :" + str(iteration))
            saver.save_model(params, iteration)
            perf_logger.stop_monitoring("Saving Model for iteration :" + str(iteration))

        if iteration % config.visualisation_freq == 0:
            perf_logger.start_monitoring("Visualising model for iteration :" + str(iteration))
            visualiser.visualise_directions(generator, deformator, iteration)
            visualiser.generate_latent_traversal(generator, deformator, iteration)
            perf_logger.stop_monitoring("Visualising model for iteration :" + str(iteration))


        ## need  to be changed
        # if iteration % config.evaluation_freq == 0 and iteration != 0:
        #     perf_logger.start_monitoring("Evaluating model for iteration :" + str(iteration))
        #     evaluator.evaluate_directions(generator, deformator.ortho_mat, iteration,
        #                                   resume_dir=config.resume_direction)
        #     perf_logger.stop_monitoring("Evaluating model for iteration :" + str(iteration))
