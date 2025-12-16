import gc,logging,re
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torchmetrics import MeanMetric

from models.convGRU.forecaster import Forecaster
from utils.loss import evaluate_loss
from utils.utils import create_directory, save_checkpoint
from utils.plot.plot_sampled_mprops import setup_predictions_plot
from utils.metrics.metricsGenerator import MetricsGenerator, compute_metrics

class ConvGRU_model:
    def __init__(self, cfg, arch, mprops_count=4, output_dir=None):
        self.cfg  = cfg
        self.arch = arch
        self.mprops_count = mprops_count
        self.output_dir = output_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convGRU = Forecaster(input_size  = (cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS),
                                  input_channels       = mprops_count,
                                  enc_hidden_channels  = cfg.MODEL.CONVGRU.ENC_HIDDEN_CH,
                                  forc_hidden_channels = cfg.MODEL.CONVGRU.FORC_HIDDEN_CH,
                                  enc_kernels          = cfg.MODEL.CONVGRU.ENC_KERNELS,
                                  forc_kernels         = cfg.MODEL.CONVGRU.FORC_KERNELS,
                                  device               = self.device,
                                  bias                 = False)

        self.convGRU.to(self.device)
        self.optimizer = torch.optim.Adam(self.convGRU.parameters(),
                                          lr = self.cfg.MODEL.CONVGRU.TRAIN.SOLVER.LR,
                                          betas=cfg.MODEL.CONVGRU.TRAIN.SOLVER.BETAS,
                                          weight_decay=cfg.MODEL.CONVGRU.TRAIN.SOLVER.WEIGHT_DECAY)

    def _train_one_epoch(self, train_data_loader, val_data_loader, epoch):
        self.convGRU.train()
        train_loss_record = MeanMetric()
        val_loss_record = MeanMetric()

        total_epochs = self.cfg.MODEL.CONVGRU.TRAIN.EPOCHS
        teacher_forcing=self.cfg.MODEL.CONVGRU.TEACHER_FORCING

        with tqdm(total=len(train_data_loader), dynamic_ncols=True) as tq:
            tq.set_description(f"Train :: Epoch: {epoch}/{total_epochs}")
            # Scan the training batches
            for batched_train_data in train_data_loader:
                tq.update(1)
                # Take a batch of macropros sequences
                past_train, future_train = batched_train_data
                past_train, future_train = past_train.float(), future_train.float()
                past_train, future_train = past_train.to(device=self.device), future_train.to(device=self.device)
                # Evaluate losses
                rloss, vloss = evaluate_loss(self.convGRU, past_train, future_train, teacher_forcing)
                # Total loss
                loss = rloss + vloss
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # Update weights
                self.optimizer.step()
                train_loss_value = loss.detach().item()
                train_loss_record.update(train_loss_value)
                tq.set_postfix_str(s=f"ConvGRU Training Loss: {train_loss_value:.4f}")

            train_mean_loss = train_loss_record.compute().item()
            tq.set_postfix_str(s=f"ConvGRU Epoch Loss: {train_mean_loss:.4f}")

        with torch.no_grad():
            with tqdm(total=len(val_data_loader), dynamic_ncols=True) as tq:
                tq.set_description(f"Val   :: Epoch: {epoch}/{total_epochs}")
                # Scan the validation batches
                for batched_val_data in val_data_loader:
                    tq.update(1)
                    # Take a batch of macropros sequences
                    past_val, future_val = batched_val_data
                    past_val, future_val = past_val.float(), future_val.float()
                    past_val, future_val = past_val.to(device=self.device), future_train.to(device=self.device)
                    rloss, vloss = evaluate_loss(self.convGRU, past_val, future_val, teacher_forcing)
                    val_loss = rloss + vloss
                    # Total loss
                    val_loss = rloss + vloss
                    val_loss_value = val_loss.detach().item()
                    val_loss_record.update(val_loss_value)
                    tq.set_postfix_str(s=f"ConvGRU Val Loss: {val_loss_value:.4f}")

            val_mean_loss = val_loss_record.compute().item()
            tq.set_postfix_str(s=f"ConvGRU Epoch Loss: {val_mean_loss:.4f}")

        return train_mean_loss, val_mean_loss

    def train(self, batched_train_data, batched_val_data):
        best_loss      = 1e6
        consecutive_nan_count = 0

        for epoch in range(1, self.cfg.MODEL.CONVGRU.TRAIN.EPOCHS + 1):
            torch.cuda.empty_cache()
            gc.collect()
            epoch_train_loss, epoch_val_loss = self._train_one_epoch(batched_train_data, batched_val_data, epoch=epoch)
            wandb.log({
                "train_loss": min(epoch_train_loss, 10),
                "val_loss": min(epoch_val_loss, 10)
            }, step=epoch)
            # NaN handling / early stopping
            if np.isnan(epoch_train_loss):
                consecutive_nan_count += 1
                logging.warning(f"Epoch {epoch}: loss is NaN ({consecutive_nan_count} consecutive)")
                if consecutive_nan_count >= 3:
                    logging.error("Loss has been NaN for 3 consecutive epochs; terminating training early.")
                    wandb.finish()
                    break
            else:
                consecutive_nan_count = 0  # reset on valid loss
            # Save best checkpoint from all training
            if epoch_train_loss < best_loss:
                best_loss = epoch_train_loss
                save_checkpoint(self.optimizer, self.convGRU, "000", self.cfg, self.arch)

    @torch.inference_mode()
    def _generate_convGRU(self, x_test, y_test, teacher_forcing):
        # Set the model in evaluation mode
        self.convGRU.eval()
        predictions = self.convGRU(x_test, y_test, teacher_forcing)
        #AR: check if exp() is still needed for rho, vx and vy
        predictions[:,0,:,:,:] = torch.exp(predictions[:,0,:,:,:])
        predictions[:,3,:,:,:] = torch.exp(predictions[:,3,:,:,:])

        return predictions

    def sampling(self, batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, macropropPlotter):
        logging.info(f'model full name:{model_fullname}')
        create_directory(self.output_dir)

        self.convGRU.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'), weights_only=True)['model'])
        self.convGRU.to(self.device)

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
            predictions = self._generate_convGRU(random_past_samples, random_future_samples, teacher_forcing=False)
            setup_predictions_plot(predictions, random_past_idx, random_past_samples, random_future_samples, model_fullname, plotType, plotMprop, plotPast, macropropPlotter)
            break

    def generate_metrics(self, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir):
        logging.info(f'model full name:{model_fullname}')
        create_directory(self.output_dir)

        self.convGRU.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'), weights_only=True)['model'])
        self.convGRU.to(self.device)

        match = re.search(r'TE\d+_PL\d+_FL\d+_CE\d+_NA', model_fullname)

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
            predictions = self._generate_convGRU(random_past_samples, random_future_samples, teacher_forcing=False)

            logging.info(f'***** Shape of seq PRED and GT BEFORE:{predictions.shape}, {random_future_samples.shape}')
            # mprops setup for metrics compute
            random_future_samples = random_future_samples[:, :self.cfg.METRICS.MPROPS_COUNT, :, :, :]
            predictions = predictions[:, :self.cfg.METRICS.MPROPS_COUNT, :, :, :]
            logging.info(f'***** Shape of seq PRED and GT AFTER:{predictions.shape}, {random_future_samples.shape}')

            for i in range(len(random_past_idx)):
                pred_seq_list.append(predictions[i])
                gt_seq_list.append(random_future_samples[i])

            count_batch += 1
            if count_batch == batches_to_use:
                break

        logging.info("===" * 20)
        logging.info(f'Computing metrics on predicted mprops sequences with ConvGRU model.')
        logging.info(f'Shape of seq PRED and GT:{pred_seq_list[0].shape}, {gt_seq_list[0].shape}')
        metricsGenerator = MetricsGenerator(pred_seq_list, gt_seq_list, self.cfg.METRICS, output_dir)
        compute_metrics(self.cfg, metricsGenerator, metric, chunkRepdPastSeq, match, batches_to_use, samples_per_batch)