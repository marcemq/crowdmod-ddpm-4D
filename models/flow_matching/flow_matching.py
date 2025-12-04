import gc,logging,re
import torch
import tqdm
import wandb
import numpy as np
from torchmetrics import MeanMetric

from utils.utils import save_checkpoint, init_wandb
from utils.plot.plot_sampled_mprops import setup_predictions_plot
from utils.metrics.metricsGenerator import MetricsGenerator, compute_metrics
from models.unet import UNet

class FM_model:
    def __init__(self, cfg, arch, mprops_count):
        self.cfg  = cfg
        self.arch = arch
        self.mprops_count = mprops_count

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.u_predictor = self._get_u_predictor()
        self.u_predictor.to(self.device)

        self.optimizer = torch.optim.Adam(self.u_predictor.parameters(),
                                          lr = self.cfg.MODEL.FLOW_MATCHING.TRAIN.SOLVER.LR,
                                          betas=cfg.MODEL.FLOW_MATCHING.TRAIN.SOLVER.BETAS,
                                          weight_decay=cfg.MODEL.FLOW_MATCHING.TRAIN.SOLVER.WEIGHT_DECAY)

        self.w_type_fns = {
            "Linear": self.w_linear,
            "Conic": self.w_conic,
        }

        self.integrators = {
            "Euler": self.sampling_with_euler,
            "Heun": self.sampling_with_euler,
        }

    def _get_u_predictor(self):
        u_predictor = None
        if self.arch == "FM-UNet":
            u_predictor = UNet(input_channels  = self.mprops_count,
                               output_channels = self.mprops_count,
                               num_res_blocks  = self.cfg.MODEL.FLOW_MATCHING.UNET.NUM_RES_BLOCKS,
                               base_channels           = self.cfg.MODEL.FLOW_MATCHING.UNET.BASE_CH,
                               base_channels_multiples = self.cfg.MODEL.FLOW_MATCHING.UNET.BASE_CH_MULT,
                               apply_attention         = self.cfg.MODEL.FLOW_MATCHING.UNET.APPLY_ATTENTION,
                               dropout_rate            = self.cfg.MODEL.FLOW_MATCHING.UNET.DROPOUT_RATE,
                               time_multiple           = self.cfg.MODEL.FLOW_MATCHING.UNET.TIME_EMB_MULT,
                               condition               = self.cfg.MODEL.FLOW_MATCHING.UNET.CONDITION)

        elif self.arch == "FM-ViT":
            u_predictor = None # Placeholder for ViT architecture
        else:
            logging.info("Architecture not supported.")

        return u_predictor

    def w_linear(self, x0, x1, t):
        #Sampling x: x0 + t * (x1 - x0)
        xt = x0 + t * (x1 - x0)
        # This is the reference we take for u
        u_target = x1 - x0
        return xt, u_target

    def w_conic(self, x0, x1, t):
        # Sampling x: x \sim N(tx1,(1-t)I)
        xt = t * x1 + (1 - t) * x0
        # This is the reference we take for u
        u_target = (x1 - xt)/(1-t)
        return xt, u_target

    def _train_one_epoch_fm(self, loader, epoch):
        total_epochs = self.cfg.MODEL.FLOW_MATCHING.TRAIN.EPOCHS
        time_max_pos = self.cfg.MODEL.FLOW_MATCHING.TIME_MAX_POS
        # Set in training mode
        self.u_predictor.train()

        wtype = self.cfg.MODEL.FLOW_MATCHING.W_TYPE
        try:
            w_fn = self.w_type_fns[wtype]
        except KeyError:
            raise ValueError(f"Unsupported W_TYPE '{wtype}'. "
                             f"Available: {list(self.w_type_fns.keys())}")

        with tqdm(total=len(loader), dynamic_ncols=True) as tq:
            loss_record = MeanMetric()
            tq.set_description(f"FM Train :: Epoch: {epoch}/{total_epochs}")
            # Scan the batches
            for batched_train_data in loader:
                tq.update(1)
                # Take a batch of macropros sequences
                past_train, future_train = batched_train_data
                past_train, future_train = past_train.float(), future_train.float()
                past_train, future_train = past_train.to(device=self.device), future_train.to(device=self.device)

                # 1. Sampling w
                x1 = future_train
                x0 = torch.randn_like(x1, device=self.device)

                # 2. Sampling random time step t for each sample in the batch
                t = torch.rand(x1.size(0), device=self.device)
                t = t.view(-1, 1, 1, 1, 1)

                # 3. Compute xt and target velocity via selected W mapping
                xt, u_target = w_fn(x0, x1, t)

               # 4. Predict velocity
                u_pred  = self.u_predictor(xt, (t * time_max_pos).long().view(-1), past_train)

                # 5. Evaluating the loss
                loss = ((u_target - u_pred) ** 2).mean()

                # Backpropagation
                loss.backward()
                # Apply optimization step
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_value = loss.detach().item()
                loss_record.update(loss_value)

                tq.set_postfix_str(s=f"FM Loss: {loss_value:.4f}")

                mean_loss = loss_record.compute().item()

            tq.set_postfix_str(s=f"FM Epoch Loss: {mean_loss:.4f}")

        return mean_loss

    def train(self, batched_train_data):
        best_loss      = 1e6
        consecutive_nan_count = 0

        low = int(self.cfg.MODEL.FLOW_MATCHING.TRAIN.EPOCHS * 0.75)
        high = self.cfg.MODEL.FLOW_MATCHING.TRAIN.EPOCHS + 1  # randint upper bound is exclusive
        epochs_cktp_to_save = np.random.randint(low, high, size=self.cfg.MODEL.FLOW_MATCHING.CHECKPOINTS_TO_KEEP)
        # Training loop
        for epoch in range(1, self.cfg.MODEL.FLOW_MATCHING.TRAIN.EPOCHS + 1):
            torch.cuda.empty_cache()
            gc.collect()

            # One epoch of training
            epoch_loss = self._train_one_epoch_fm(batched_train_data, epoch=epoch)
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
                save_checkpoint(self.optimizer, self.u_predictor, "000", self.cfg, self.arch)

            # Save model samples at stable loss
            if epoch in epochs_cktp_to_save:
                logging.info(f"Epoch {epoch}: in checkpoints_to_keep set, saving model.")
                save_checkpoint(self.optimizer, self.u_predictor, epoch, self.cfg, self.arch)

    @torch.inference_mode()
    def sampling_with_euler(self, past:torch.Tensor, nsamples):
        # Set the model in evaluation mode
        self.u_predictor.eval()
        # Noise from a normal distribution
        xt = torch.randn((nsamples, self.mprops_count, self.cfg.MACROPROPS.ROWS, self.cfg.MACROPROPS.COLS, self.cfg.DATASET.FUTURE_LEN), device=self.device)
        time_max_pos = self.cfg.MODEL.FLOW_MATCHING.TIME_MAX_POS
        delta = 1 / self.cfg.MODEL.FLOW_MATCHING.INTEGRATOR_STEPS.EULER

        pbar = tqdm(range(1, self.cfg.MODEL.FLOW_MATCHING.INTEGRATOR_STEPS.EULER + 1), desc="Sampling (Euler)")

        # Cycle over the integration steps
        for i, t in enumerate(torch.linspace(0, 1, self.cfg.MODEL.FLOW_MATCHING.INTEGRATOR_STEPS.EULER, device=self.device), start=1):
            time_indices = (t * time_max_pos).clamp(0, time_max_pos-1).long()
            time_indices = time_indices.to(self.device).expand(xt.size(0))
            # Apply the velocity to get the velocity
            u = self.u_predictor(xt, time_indices, past)
            # Integration step
            xt = xt + delta * u
            pbar.update(1)

        pbar.close()
        return xt

    @torch.inference_mode()
    def sampling_with_heun(self, past:torch.Tensor, nsamples):
        return 1

    def sampling(self, batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, macropropPlotter):
        logging.info(f'model full name:{model_fullname}')
        self.u_predictor.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'), weights_only=True)['model'])
        self.u_predictor.to(self.device)

        intg_type = self.cfg.FLOW_MATCHING.INTEGRATOR
        try:
            integrator = self.integrators[intg_type]
        except KeyError:
            raise ValueError(f"Unsupported INTEGRATOR '{intg_type}'. "
                             f"Available: {list(self.integrators.keys())}")

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

            predictions = integrator(random_past_samples, self.cfg.MODEL.NSAMPLES4PLOTS)

            setup_predictions_plot(predictions, random_past_idx, random_past_samples, random_future_samples, model_fullname, plotType, plotMprop, plotPast, macropropPlotter)
            break

    def generate_metrics(self, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir):
        logging.info(f'model full name:{model_fullname}')
        self.u_predictor.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'), weights_only=True)['model'])
        self.u_predictor.to(self.device)

        match = re.search(r'TE\d+_PL\d+_FL\d+_CE\d+_VN[FT]', model_fullname)

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

            integrator = self.integrators[self.cfg.FLOW_MATCHING.INTEGRATOR]
            x = integrator(random_past_samples, self.cfg.MODEL.NSAMPLES4PLOTS)
            future_samples_pred = x
            for i in range(len(random_past_idx)):
                pred_seq_list.append(future_samples_pred[i])
                gt_seq_list.append(random_future_samples[i])

            count_batch += 1
            if count_batch == batches_to_use:
                break

        logging.info("===" * 20)
        logging.info(f'Computing metrics on predicted mprops sequences with FM-UNet model.')
        metricsGenerator = MetricsGenerator(pred_seq_list, gt_seq_list, self.cfg.METRICS, output_dir)
        compute_metrics(self.cfg, metricsGenerator, metric, chunkRepdPastSeq, match, batches_to_use, samples_per_batch)