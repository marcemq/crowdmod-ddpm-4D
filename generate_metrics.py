import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os, re
from models.generate import generate_ddpm, generate_ddim

from utils.myparser import getYamlConfig
from utils.dataset import getDataset
from utils.computeMetrics import psnr_mprops_seq, ssim_mprops_seq, motion_feature_by_mse, motion_feature_by_bhattacharyya
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM

def save_metric_data(cfg, match, data, metric, header):
    file_name = f"metrics/mpSampling_{metric}_NS{cfg.DIFFUSION.NSAMPLES}_{match.group()}.csv"
    np.savetxt(file_name, data, delimiter=",", header=header, comments="")

def save_all_metrics(match, metrics_dict):
    # Stack metrics by epoch into an array
    for metric_name, (metric_data_list, _) in metrics_dict.items():
        metrics_dict[metric_name][0] = np.stack(metric_data_list)

    # Save each non-empty metric with its required data
    for metric_name, (metric_data, metric_header) in metrics_dict.items():
        if len(metric_data) != 0:
            save_metric_data(cfg, match, metric_data, metric_name, metric_header)

def get_metrics_dict():
    metrics_dict = {"PSNR" : ([], "rho,vx,vy,unc"),
                    "MAX-PSNR" : ([], "rho,vx,vy,unc"),
                    "SSIM" : ([], "rho,vx,vy,unc"),
                    "MAX-SSIM" : ([], "rho,vx,vy,unc"),
                    "MOTIONFEAT_MSE" : ([], "MSE_Hist_2D_Based,MSE_Hist_1D_Based"),
                    "MOTIONFEAT_BHATT_DIST" : ([], "BHATT_DIST_Hist_2D_Based,BHATT_DIST_Hist_1D_Based"),
                    "MOTIONFEAT_BHATT_COEF" : ([], "BHATT_COEF_Hist_2D_Based,BHATT_COEF_Hist_1D_Based")
                    }
    return metrics_dict

def generate_metrics(cfg, filenames, chunkRepdPastSeq, metric, batches_to_use):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    _, _, batched_test_data = getDataset(cfg, filenames, test_data_only=True)
    # Instanciate the UNet for the reverse diffusion
    denoiser = MacropropsDenoiser(num_res_blocks = cfg.MODEL.NUM_RES_BLOCKS,
                                base_channels           = cfg.MODEL.BASE_CH,
                                base_channels_multiples = cfg.MODEL.BASE_CH_MULT,
                                apply_attention         = cfg.MODEL.APPLY_ATTENTION,
                                dropout_rate            = cfg.MODEL.DROPOUT_RATE,
                                time_multiple           = cfg.MODEL.TIME_EMB_MULT,
                                condition               = cfg.MODEL.CONDITION)
    lr_str = "{:.0e}".format(cfg.TRAIN.SOLVER.LR)
    model_fullname = cfg.MODEL.SAVE_DIR+(cfg.MODEL.MODEL_NAME.format(cfg.TRAIN.EPOCHS, lr_str, cfg.DATASET.TRAIN_FILE_COUNT, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN))
    print(f'model full name:{model_fullname}')
    denoiser.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'))['model'])
    denoiser.to(device)
    match = re.search(r'E\d+_LR\de-\d+_TFC\d+_PL\d+_FL\d', model_fullname)

    # Instantiate the diffusion model
    timesteps=cfg.DIFFUSION.TIMESTEPS
    diffusionmodel = DDPM(timesteps=cfg.DIFFUSION.TIMESTEPS)
    diffusionmodel.to(device)
    pred_seq_list, gt_seq_list = [], []
    taus = 1
    count_batch = 0
    metrics_dict = get_metrics_dict()
    # cicle over batched test data
    for batch in batched_test_data:
        past_test, future_test, stats = batch
        past_test, future_test = past_test.float(), future_test.float()
        past_test, future_test = past_test.to(device=device), future_test.to(device=device)
        # Compute the idx of the past sequences to work on
        if past_test.shape[0] < cfg.DIFFUSION.NSAMPLES:
            random_past_idx = torch.randperm(past_test.shape[0])
        else:
            random_past_idx = torch.randperm(past_test.shape[0])[:cfg.DIFFUSION.NSAMPLES]
        expanded_random_past_idx = torch.repeat_interleave(random_past_idx, chunkRepdPastSeq)
        random_past_idx = expanded_random_past_idx[:cfg.DIFFUSION.NSAMPLES]
        random_past_samples = past_test[random_past_idx]
        random_future_samples = future_test[random_past_idx]

        if cfg.DIFFUSION.SAMPLER == "DDPM":
            x, xnoisy_over_time  = generate_ddpm(denoiser, random_past_samples, diffusionmodel, cfg, device) # AR review .cpu() call here
            if cfg.DIFFUSION.GUIDANCE == "sparsity" or cfg.DIFFUSION.GUIDANCE == "none":
                l1 = torch.mean(torch.abs(x[:,0,:,:,:])).cpu().detach().numpy()
                print('L1 norm {:.2f}'.format(l1))
        elif cfg.DIFFUSION.SAMPLER == "DDIM":
            taus = np.arange(0,timesteps,cfg.DIFFUSION.DDIM_DIVIDER)
            print(f'taus:{taus}')
            x, xnoisy_over_time = generate_ddim(denoiser, random_past_samples,taus,diffusionmodel,cfg,device) # AR review .cpu() call here
        else:
            print(f"{cfg.DIFFUSION.SAMPLER} sampler not supported")

        future_samples_pred = xnoisy_over_time[cfg.DIFFUSION.TIMESTEPS]
        for i in range(len(random_past_idx)):
            pred_seq_list.append(future_samples_pred[i])
            gt_seq_list.append(random_future_samples[i])

        if metric in ['PSNR', 'ALL']:
            mprops_psnr, mprops_max_psnr = psnr_mprops_seq(gt_seq_list, pred_seq_list, cfg.DIFFUSION.PRED_MPROPS_FACTOR, chunkRepdPastSeq, cfg.MACROPROPS.EPS)
            metrics_dict['PSNR'][0].append(mprops_psnr)
            metrics_dict['MAX-PSNR'][0].append(mprops_max_psnr)
        if metric in ['SSIM', 'ALL']:
            mprops_ssim, mprops_max_ssim = ssim_mprops_seq(gt_seq_list, pred_seq_list, cfg.DIFFUSION.PRED_MPROPS_FACTOR, chunkRepdPastSeq)
            metrics_dict['SSIM'][0].append(mprops_ssim)
            metrics_dict['MAX-SSIM'][0].append(mprops_max_ssim)
        if metric in ['MOTION_FEAT_MSE', 'ALL']:
            motion_feat_mse = motion_feature_by_mse(gt_seq_list, pred_seq_list, cfg.METRICS.MOTION_FEATURE.f, cfg.METRICS.MOTION_FEATURE.k, cfg.METRICS.MOTION_FEATURE.GAMMA)
            metrics_dict["MOTIONFEAT_MSE"][0].append(motion_feat_mse)
        if metric in ['MOTION_FEAT_BHATT', 'ALL']:
            motion_feat_bhatt_dist, motion_feat_bhatt_coef = motion_feature_by_bhattacharyya(gt_seq_list, pred_seq_list, cfg.METRICS.MOTION_FEATURE.f, cfg.METRICS.MOTION_FEATURE.k, cfg.METRICS.MOTION_FEATURE.GAMMA)
            metrics_dict["MOTIONFEAT_BHATT_DIST"][0].append(motion_feat_bhatt_dist)
            metrics_dict["MOTIONFEAT_BHATT_COEF"][0].append(motion_feat_bhatt_coef)

        count_batch += 1
        if count_batch == batches_to_use:
            break

    save_all_metrics(metrics_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to sample crowd macroprops from trained model.")
    parser.add_argument('--chunk-repd-past-seq', type=int, default=5, help='Chunk of repeteaded past sequences to use when predict.')
    parser.add_argument('--metric', type=str, default='PSNR', help='Name of the metric to compute')
    parser.add_argument('--batches-to-use', type=int, default=1, help='Total of batches to use to compute metrics.')
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm_4test.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/ATC_ddpm_DSlist4test.yml',help='Configuration YML macroprops list for specific dataset.')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    filenames = cfg.SUNDAY_DATA_LIST
    filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    generate_metrics(cfg, filenames, chunkRepdPastSeq=args.chunk_repd_past_seq, metric=args.metric, batches_to_use=args.batches_to_use)
