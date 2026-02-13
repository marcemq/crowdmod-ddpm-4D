import gc,logging,re
import torch
import wandb
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.cuda import amp
from torchmetrics import MeanMetric

from models.diffusion.forward import ForwardSampler
from models.backbones.unet import UNet
from models.diffusion.forward import get_from_idx
from models.guidance import sparsityGradient, preservationMassNumericalGradientOptimal
from utils.utils import save_checkpoint, init_wandb, create_directory
from utils.plot.plot_sampled_mprops import setup_predictions_plot
from utils.metrics.metricsGenerator import MetricsGenerator, compute_metrics

class DDPM(ForwardSampler):
    # This will implement one step back in the reverse process
    def step(self, predicted_noise:torch.Tensor, xnoise:torch.Tensor, timestep: int):
        # Noise from normal distribution
        z  = torch.randn_like(xnoise) if timestep > 0 else torch.zeros_like(xnoise)
        beta_t                     = self.beta[timestep].reshape(-1, 1, 1)
        one_by_sqrt_alpha_t        = self.one_by_sqrt_alpha[timestep].reshape(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[timestep].reshape(-1, 1, 1)
        # Use the formula above to sample a denoised version from the noisy one
        # equation at Algorithm 2 Sampling pseudocode
        xdenoised = (
            one_by_sqrt_alpha_t
            * (xnoise - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )
        return xdenoised, torch.sqrt(beta_t), 1-beta_t

class DDPM_model:
    def __init__(self, cfg, arch, mprops_count, output_dir=None):
        self.cfg  = cfg
        self.arch = arch
        self.mprops_count = mprops_count
        self.output_dir = output_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.denoiser = self._get_denoiser()
        self.denoiser.to(self.device)

        self.optimizer = torch.optim.Adam(self.denoiser.parameters(),
                                          lr = self.cfg.MODEL.DDPM.TRAIN.SOLVER.LR,
                                          betas=cfg.MODEL.DDPM.TRAIN.SOLVER.BETAS,
                                          weight_decay=cfg.MODEL.DDPM.TRAIN.SOLVER.WEIGHT_DECAY)

    def _get_denoiser(self):
        denoiser = None
        if self.arch == "DDPM-UNet":
            denoiser = UNet(input_channels  = self.mprops_count,
                            output_channels = self.mprops_count,
                            num_res_blocks  = self.cfg.MODEL.DDPM.UNET.NUM_RES_BLOCKS,
                            base_channels           = self.cfg.MODEL.DDPM.UNET.BASE_CH,
                            base_channels_multiples = self.cfg.MODEL.DDPM.UNET.BASE_CH_MULT,
                            apply_attention         = self.cfg.MODEL.DDPM.UNET.APPLY_ATTENTION,
                            dropout_rate            = self.cfg.MODEL.DDPM.UNET.DROPOUT_RATE,
                            time_multiple           = self.cfg.MODEL.DDPM.UNET.TIME_EMB_MULT,
                            condition               = self.cfg.MODEL.DDPM.UNET.CONDITION)

        elif self.arch == "DDPM-ViT":
            denoiser = None # Placeholder for ViT architecture
        else:
            logging.info("Architecture not supported.")

        return denoiser

    # Apply one training step
    def _train_step(self, future:torch.Tensor, past:torch.Tensor, forward_sampler:DDPM):
        # Sample a timestep uniformly
        t = torch.randint(low=0, high=forward_sampler.timesteps, size=(future.shape[0],), device=future.device)
        # Apply forward noising process on original images, up to step t (sample from q(x_t|x_0))
        future_macroprops_noisy, eps_true = forward_sampler(future, t)
        with amp.autocast():
            # Our prediction for the denoised macropros sequence
            eps_predicted = self.denoiser(future_macroprops_noisy, t, past)
            # Deduce the loss
            loss          = F.mse_loss(eps_predicted, eps_true)
        return loss

    def _train_one_epoch(self, forward_sampler:DDPM, loader, epoch):
        total_epochs = self.cfg.MODEL.DDPM.TRAIN.EPOCHS
        loss_record = MeanMetric()
        # Set in training mode
        self.denoiser.train()

        with tqdm(total=len(loader), dynamic_ncols=True) as tq:
            tq.set_description(f"DDPM Train :: Epoch: {epoch}/{total_epochs}")
            # Scan the batches
            for batched_train_data in loader:
                tq.update(1)
                # Take a batch of macropros sequences
                past_train, future_train = batched_train_data
                past_train, future_train = past_train.float(), future_train.float()
                past_train, future_train = past_train.to(device=self.device), future_train.to(device=self.device)

                loss = self._train_step(future_train, past_train, forward_sampler)

                # Backpropagation and update
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                loss_value = loss.detach().item()
                loss_record.update(loss_value)

                tq.set_postfix_str(s=f"DDPM Loss: {loss_value:.4f}")

            mean_loss = loss_record.compute().item()

            tq.set_postfix_str(s=f"DDPM Epoch Loss: {mean_loss:.4f}")

        return mean_loss

    def train(self, batched_train_data):
        forward_sampler = DDPM(timesteps=self.cfg.MODEL.DDPM.TIMESTEPS, scale=self.cfg.MODEL.DDPM.SCALE)
        forward_sampler.to(self.device)

        best_loss      = 1e6
        consecutive_nan_count = 0

        low = int(self.cfg.MODEL.DDPM.TRAIN.EPOCHS * 0.75)
        high = self.cfg.MODEL.DDPM.TRAIN.EPOCHS + 1  # randint upper bound is exclusive
        epochs_cktp_to_save = np.random.randint(low, high, size=self.cfg.MODEL.DDPM.CHECKPOINTS_TO_KEEP)
        # Training loop
        for epoch in range(1, self.cfg.MODEL.DDPM.TRAIN.EPOCHS + 1):
            torch.cuda.empty_cache()
            gc.collect()

            # One epoch of training
            epoch_loss = self._train_one_epoch(forward_sampler, batched_train_data, epoch)
            wandb.log({"train_loss": epoch_loss})
            # NaN handling / early stopping
            if np.isnan(epoch_loss):
                consecutive_nan_count += 1
                logging.warning(f"Epoch {epoch}: loss is NaN ({consecutive_nan_count} consecutive)")
                if consecutive_nan_count >= 3:
                    logging.error("Loss has been NaN for 3 consecutive epochs; terminating training early.")
                    wandb.finish()
                    break
            else:
                consecutive_nan_count = 0  # reset on valid loss

            # Save best checkpoint from all training
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(self.optimizer, self.denoiser, "000", self.cfg, self.arch)

            # Save model samples at stable loss
            if epoch in epochs_cktp_to_save:
                logging.info(f"Epoch {epoch}: in checkpoints_to_keep set, saving model.")
                save_checkpoint(self.optimizer, self.denoiser, epoch, self.cfg, self.arch)

        logging.info(f"Trained model {self.arch} saved in {self.output_dir}")

    @torch.inference_mode()
    def _generate_ddpm(self, past:torch.Tensor, backward_sampler:DDPM, nsamples, history=False):
        # Set the model in evaluation mode
        self.denoiser.eval()
        # Noise from a normal distribution
        xnoisy = torch.randn((nsamples, self.mprops_count, self.cfg.MACROPROPS.ROWS, self.cfg.MACROPROPS.COLS, self.cfg.DATASET.FUTURE_LEN), device=self.device)
        xnoisy_over_time = [xnoisy]
        # Now, to reverse the diffusion process, use a sequence of denoising steps
        for t in tqdm(iterable=reversed(range(0, backward_sampler.timesteps)),
                            dynamic_ncols=False,total=backward_sampler.timesteps,
                            desc="DDPM Sampling :: ", position=0):
            t_tensor = torch.as_tensor(t, dtype=torch.long, device=self.device).reshape(-1).expand(xnoisy.shape[0])
            # Estimate the noise
            eps_pred = self.denoiser(xnoisy, t_tensor, past)
            # Denoise with the sampler and the estimation of the noise
            xnoisy, sigma, alpha_t = backward_sampler.step(eps_pred, xnoisy, t)
            if self.cfg.MODEL.DDPM.GUIDANCE == "sparsity":
                # Update the noisy image with the sparsity guidance
                sparsity_grad = sparsityGradient(xnoisy, self.cfg, self.device)
                xnoisy-= 0.004*sigma*sparsity_grad # 0.004*sqrt(1-alpha_t)
            if self.cfg.MODEL.DDPM.GUIDANCE == "mass_preservation":
                mass_preserv_grad = preservationMassNumericalGradientOptimal(xnoisy, self.device, delta_t=1.0, delta_l=1.0, eps=0.1)
                xnoisy-= (1-alpha_t)*mass_preserv_grad
            if history:
                xnoisy_over_time.append(xnoisy)

        if not history:
            xnoisy_over_time.append(xnoisy)

        return xnoisy, xnoisy_over_time

    @torch.inference_mode()
    def _generate_ddim(self, past:torch.Tensor, taus, backward_sampler:DDPM, nsamples, history=False):
        # Set the model in evaluation mode
        self.denoiser.eval()
        # Noise from a normal distribution
        xnoisy = torch.randn((nsamples, self.mprops_count, self.cfg.MACROPROPS.ROWS, self.cfg.MACROPROPS.COLS, self.cfg.DATASET.FUTURE_LEN), device=self.device)
        last_t                     = torch.ones(xnoisy.shape[0], dtype=torch.long, device=self.device) * (backward_sampler.timesteps-1)
        beta_t                     = get_from_idx(backward_sampler.beta, last_t)
        sqrt_alpha_bar_t           = get_from_idx(backward_sampler.sqrt_alpha_bar, last_t)
        sqrt_one_minus_alpha_bar_t = get_from_idx(backward_sampler.sqrt_one_minus_alpha_bar, last_t)
        xnoisy_over_time = [xnoisy]
        # Now, to reverse the diffusion process, use a sequence of denoising steps
        for t in tqdm(iterable=reversed(taus), dynamic_ncols=False,total=len(taus), desc="DDIM Sampling :: ", position=0):
            # Time vectors
            ts = torch.ones(xnoisy.shape[0], dtype=torch.long, device=self.device) * t
            # Estimate the noise
            predicted_noise = self.denoiser(xnoisy, ts, past)
            # The betas, alphas etc.
            beta_t_prev                     = get_from_idx(backward_sampler.beta, ts)
            sqrt_alpha_bar_t_prev           = get_from_idx(backward_sampler.sqrt_alpha_bar, ts)
            sqrt_one_minus_alpha_bar_t_prev = get_from_idx(backward_sampler.sqrt_one_minus_alpha_bar, ts)
            # Predicted x0
            predicted_x0                    = (xnoisy-sqrt_one_minus_alpha_bar_t*predicted_noise)/sqrt_alpha_bar_t
            # AR: Generating images for t-1 (deterministic way). Review this step, can we do it no deterministic?
            # AR: redo eq 65, 67 that depends on sigma and test, with sigma=0, and sigma=1
            xnoisy = sqrt_alpha_bar_t_prev * predicted_x0 + sqrt_one_minus_alpha_bar_t_prev * predicted_noise
            if self.cfg.MODEL.DDPM.GUIDANCE == "sparsity":
                # Update the noisy image with the sparsity guidance
                sparsity_grad = sparsityGradient(xnoisy, self.cfg, self.device)
                sigma = torch.sqrt(beta_t)
                xnoisy-= 0.004*sigma*sparsity_grad

            beta_t                          = beta_t_prev
            sqrt_alpha_bar_t                = sqrt_alpha_bar_t_prev
            sqrt_one_minus_alpha_bar_t      = sqrt_one_minus_alpha_bar_t_prev
            if history:
                xnoisy_over_time.append(xnoisy)

        if not history:
            xnoisy_over_time.append(xnoisy)

        return xnoisy, xnoisy_over_time

    def sampling(self, batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, macropropPlotter):
        logging.info(f'model full name:{model_fullname}')
        create_directory(self.output_dir)

        self.denoiser.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'), weights_only=True)['model'])
        self.denoiser.to(self.device)

        timesteps = self.cfg.MODEL.DDPM.TIMESTEPS
        backward_sampler = DDPM(timesteps=self.cfg.MODEL.DDPM.TIMESTEPS, scale=self.cfg.MODEL.DDPM.SCALE)
        backward_sampler.to(self.device)

        for batch in batched_test_data:
            past_test, future_test = batch
            past_test, future_test = past_test.float(), future_test.float()
            past_test, future_test = past_test.to(device=self.device), future_test.to(device=self.device)
            random_past_idx = torch.randperm(past_test.shape[0])[:self.cfg.MODEL.NSAMPLES4PLOTS]
            # Predict different sequences for the same past sequence
            if samePastSeq:
                fixed_past_idx = random_past_idx[0]
                random_past_idx.fill_(fixed_past_idx)

            random_past_samples = past_test[random_past_idx]
            random_future_samples = future_test[random_past_idx]

            if self.cfg.MODEL.DDPM.SAMPLER == "DDPM":
                predictions, _  = self._generate_ddpm(random_past_samples, backward_sampler, self.cfg.MODEL.NSAMPLES4PLOTS) # AR review .cpu() call here
                if self.cfg.MODEL.DDPM.GUIDANCE == "sparsity" or self.cfg.MODEL.DDPM.GUIDANCE == "None":
                    l1 = torch.mean(torch.abs(predictions[:,0,:,:,:])).cpu().detach().numpy()
                    logging.info('L1 norm {:.2f}'.format(l1))
            elif self.cfg.MODEL.DDPM.SAMPLER == "DDIM":
                taus = np.arange(0, timesteps, self.cfg.MODEL.DDPM.DDIM_DIVIDER)
                logging.info(f'Shape of subset taus:{taus.shape}')
                predictions, _ = self._generate_ddim(random_past_samples, taus, backward_sampler, self.cfg.MODEL.NSAMPLES4PLOTS) # AR review .cpu() call here
            else:
                logging.info(f"{self.cfg.MODEL.DDPM.SAMPLER} sampler not supported")

            setup_predictions_plot(predictions, random_past_idx, random_past_samples, random_future_samples, model_fullname, plotType, plotMprop, plotPast, macropropPlotter)
            logging.info(f"All sampling macroprops seqs saved in {self.output_dir}")
            break

    def generate_metrics(self, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir):
        logging.info(f'model full name:{model_fullname}')
        create_directory(self.output_dir)

        self.denoiser.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'), weights_only=True)['model'])
        self.denoiser.to(self.device)

        match = re.search(r'TE\d+_PL\d+_FL\d+_CE\d+_NA', model_fullname)
        timesteps = self.cfg.MODEL.DDPM.TIMESTEPS
        backward_sampler = DDPM(timesteps=self.cfg.MODEL.DDPM.TIMESTEPS, scale=self.cfg.MODEL.DDPM.SCALE)
        backward_sampler.to(self.device)

        count_batch = 0
        pred_seq_list, gt_seq_list = [], []
        # cicle over batched test data
        for batch in batched_test_data:
            logging.info("===" * 20)
            logging.info(f'Computing sampling on batch:{count_batch+1}')
            past_test, future_test = batch
            past_test, future_test = past_test.float(), future_test.float()
            past_test, future_test = past_test.to(device=self.device), future_test.to(device=self.device)
            # Compute the idx of the past sequences to work on
            if past_test.shape[0] < samples_per_batch:
                random_past_idx = torch.randperm(past_test.shape[0])
            else:
                random_past_idx = torch.randperm(past_test.shape[0])[:samples_per_batch]

            expanded_random_past_idx = torch.repeat_interleave(random_past_idx, chunkRepdPastSeq)
            random_past_idx = expanded_random_past_idx[:samples_per_batch]
            random_past_samples = past_test[random_past_idx]
            random_future_samples = future_test[random_past_idx]

            if self.cfg.MODEL.DDPM.SAMPLER == "DDPM":
                x, _  = self._generate_ddpm(random_past_samples, backward_sampler, samples_per_batch) # AR review .cpu() call here
                if self.cfg.MODEL.DDPM.GUIDANCE == "sparsity" or self.cfg.MODEL.DDPM.GUIDANCE=="mass_preservation" or self.cfg.MODEL.DDPM.GUIDANCE == "None":
                    l1 = torch.mean(torch.abs(x[:,0,:,:,:])).cpu().detach().numpy()
                    logging.info(f'L1 norm {l1:.2f} using {self.cfg.MODEL.DDPM.GUIDANCE} guidance')
            elif self.cfg.MODEL.DDPM.SAMPLER == "DDIM":
                taus = np.arange(0, timesteps, self.cfg.MODEL.DDPM.DDIM_DIVIDER)
                logging.info(f'Shape of subset taus:{taus.shape}')
                x, _ = self._generate_ddim(random_past_samples, taus, backward_sampler, samples_per_batch) # AR review .cpu() call here
            else:
                logging.info(f"{self.cfg.MODEL.DDPM.SAMPLER} sampler not supported")

            future_samples_pred = x
            for i in range(len(random_past_idx)):
                pred_seq_list.append(future_samples_pred[i])
                gt_seq_list.append(random_future_samples[i])

            count_batch += 1
            if count_batch == batches_to_use:
                break

        logging.info("===" * 20)
        logging.info(f'Computing metrics on predicted mprops sequences with DDPM model.')
        metricsGenerator = MetricsGenerator(pred_seq_list, gt_seq_list, self.cfg.METRICS, output_dir)
        compute_metrics(self.cfg, metricsGenerator, metric, chunkRepdPastSeq, match, batches_to_use, samples_per_batch, self.arch)