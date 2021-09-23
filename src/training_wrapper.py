from model_loader import get_model
from train import Trainer
from saver import Saver
from utils import *
import wandb
from evaluation import Evaluator
from visualisation import Visualiser


def run_training_wrapper(config, logger, perf_logger):
    directories = list_dir_recursively_with_ignore('.',
                                                   ignores=['checkpoint.pt', '__pycache__', 'wandb', 'venv', '.idea',
                                                            'data', 'pretrained_models',
                                                            'results', '.git', '.gitignore', 'artifacts'])
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

    generator, deformator, deformator_opt, eps_predictor, eps_predictor_opt = models
    generator.eval()
    deformator.train()
    eps_predictor_loss_list = []
    deformator_ranking_loss_list = []
    # saver.load_model((deformator,deformator_opt))
    for iteration in range(config.num_iterations):
        deformator, deformator_opt, eps_predictor, eps_predictor_opt, eps_predictor_loss, deformator_ranking_loss = \
            model_trainer.train_ours(generator, deformator, deformator_opt, eps_predictor, eps_predictor_opt)
        eps_predictor_loss_list.append(eps_predictor_loss)
        deformator_ranking_loss_list.append(deformator_ranking_loss)

        if iteration % config.logging_freq == 0 and iteration != 0:
            eps_predictor_loss_avg = sum(eps_predictor_loss_list) / len(eps_predictor_loss_list)
            deformator_ranking_loss_avg = sum(deformator_ranking_loss_list) / len(deformator_ranking_loss_list)
            logger.info("step : %d / %d eps predictor loss : %.4f Deformator_loss  %.4f " % (
                iteration, config.num_iterations, eps_predictor_loss_avg, deformator_ranking_loss_avg))
            wandb.log({'iteration': iteration + 1, 'loss': deformator_ranking_loss_avg})

        if iteration % config.saving_freq == 0 and iteration != 0:
            params = (deformator, deformator_opt, eps_predictor, eps_predictor_opt)
            perf_logger.start_monitoring("Saving Model for iteration :" + str(iteration))
            saver.save_model(params, iteration)
            perf_logger.stop_monitoring("Saving Model for iteration :" + str(iteration))

        if iteration % config.evaluation_freq == 0 and iteration != 0:
            perf_logger.start_monitoring("Evaluating model for iteration :" + str(iteration))
            evaluator.evaluate_directions(generator, deformator.ortho_mat, iteration, resume_dir=config.resume_direction)
            perf_logger.stop_monitoring("Evaluating model for iteration :" + str(iteration))

        if iteration % config.visualisation_freq == 0:
            perf_logger.start_monitoring("Visualising model for iteration :" + str(iteration))
            visualiser.visualise_directions(generator, deformator, iteration)
            visualiser.generate_latent_traversal(generator, deformator,iteration)
            perf_logger.stop_monitoring("Visualising model for iteration :" + str(iteration))
