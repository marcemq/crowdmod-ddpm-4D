import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os, re
from models.generate import generate_ddpm, generate_ddim

from utils.myparser import getYamlConfig
from utils.dataset import getDataset
from utils.utils import create_directory
from utils.plot_metrics import createBoxPlot, createBoxPlot_bhatt, merge_and_plot_boxplot
from utils.computeMetrics import psnr_mprops_seq, ssim_mprops_seq, motion_feature_metrics
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM

def save_metric_data(cfg, match, data, metric, header):
    file_name = f"{cfg.MODEL.OUTPUT_DIR}/mpSampling_{metric}_NS{cfg.DIFFUSION.NSAMPLES}_{match.group()}.csv"
    np.savetxt(file_name, data, delimiter=",", header=header, comments="")
    return file_name

def save_all_metrics(match, metrics_data_dict, metrics_header_dict, title):
    metrics_filenames_dict = {"title": title}
    # Stack metrics by epoch into an array
    for metric_name, metric_data_list in metrics_data_dict.items():
        metrics_data_dict[metric_name] = np.vstack(metric_data_list)

    # Save each non-empty metric with its required data
    for metric_name, metric_header in metrics_header_dict.items():
        if len(metrics_data_dict[metric_name]) != 0:
            file_name = save_metric_data(cfg, match, metrics_data_dict[metric_name], metric_name, metric_header)
            metrics_filenames_dict[metric_name] = file_name

    with open(f"{cfg.MODEL.OUTPUT_DIR}/metrics_files.json", "w") as json_file:
        json.dump(metrics_filenames_dict, json_file)
    print(f"Dictionary of metrics filenames saved to '{cfg.MODEL.OUTPUT_DIR}/metrics_files.json'")

def save_all_boxplots_metrics(metrics_data_dict, metrics_header_dict, title):
    # Convert the dictionary of arrays into a dictionary of DataFrames
    metrics_df_dict = {key: pd.DataFrame(value) for key, value in metrics_data_dict.items()}

    merge_and_plot_boxplot(df_max=metrics_df_dict['MAX-PSNR'], df=metrics_df_dict['PSNR'], title=f"PSNR and MAX-PSNR of {title}", save_path=f"{cfg.MODEL.OUTPUT_DIR}/BP_PSNR.png")
    merge_and_plot_boxplot(df_max=metrics_df_dict['MAX-SSIM'], df=metrics_df_dict['SSIM'], title=f"SSIM and MAX-SSIM of {title}", save_path=f"{cfg.MODEL.OUTPUT_DIR}/BP_SSIM.png")
    createBoxPlot(metrics_df_dict['MOTION_FEAT_MSE'], title=f"MSE of Motion feature of {title}", columns_to_plot=metrics_header_dict["MOTION_FEAT_MSE"], save_path=f"{cfg.MODEL.OUTPUT_DIR}/BP_MF_MSE.png")
    createBoxPlot_bhatt(metrics_df_dict['MOTION_FEAT_BHATT_COEF'], metrics_df_dict['MOTION_FEAT_BHATT_DIST'], title=f"BHATT of Motion feature of {title}", save_path=f"{cfg.MODEL.OUTPUT_DIR}/BP_BHATT.png")

def get_metrics_dicts():
    metrics_data_dict = {"PSNR" : [],
                    "MAX-PSNR" : [],
                    "SSIM" : [],
                    "MAX-SSIM" : [],
                    "MOTION_FEAT_MSE" : [],
                    "MOTION_FEAT_BHATT_DIST" : [],
                    "MOTION_FEAT_BHATT_COEF" : []
                    }
    metrics_header_dict = {"PSNR" : "rho,vx,vy",
                    "MAX-PSNR" : "rho,vx,vy",
                    "SSIM" : "rho,vx,vy",
                    "MAX-SSIM" : "rho,vx,vy",
                    "MOTION_FEAT_MSE" : "MSE_Hist_2D_Based,MSE_Hist_1D_Based",
                    "MOTION_FEAT_BHATT_DIST" : "BHATT_DIST_Hist_2D_Based,BHATT_DIST_Hist_1D_Based",
                    "MOTION_FEAT_BHATT_COEF" : "BHATT_COEF_Hist_2D_Based,BHATT_COEF_Hist_1D_Based"
                    }
    return metrics_data_dict, metrics_header_dict

def generate_metrics(cfg, filenames, chunkRepdPastSeq, metric, batches_to_use):
    create_directory(cfg.MODEL.OUTPUT_DIR)
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
    taus = 1
    count_batch = 0
    metrics_data_dict, metrics_header_dict = get_metrics_dicts()
    # cicle over batched test data
    for batch in batched_test_data:
        print("===" * 20)
        print(f'Computing metrics on batch:{count_batch+1}')
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
            x, xnoisy_over_time  = generate_ddpm(denoiser, random_past_samples, diffusionmodel, cfg, device, cfg.DIFFUSION.NSAMPLES) # AR review .cpu() call here
            if cfg.DIFFUSION.GUIDANCE == "sparsity" or cfg.DIFFUSION.GUIDANCE == "none":
                l1 = torch.mean(torch.abs(x[:,0,:,:,:])).cpu().detach().numpy()
                print('L1 norm {:.2f}'.format(l1))
        elif cfg.DIFFUSION.SAMPLER == "DDIM":
            taus = np.arange(0,timesteps,cfg.DIFFUSION.DDIM_DIVIDER)
            print(f'taus:{taus}')
            x, xnoisy_over_time = generate_ddim(denoiser, random_past_samples, taus, diffusionmodel, cfg, device, cfg.DIFFUSION.NSAMPLES) # AR review .cpu() call here
        else:
            print(f"{cfg.DIFFUSION.SAMPLER} sampler not supported")

        future_samples_pred = x
        pred_seq_list, gt_seq_list = [], []
        for i in range(len(random_past_idx)):
            pred_seq_list.append(future_samples_pred[i])
            gt_seq_list.append(random_future_samples[i])

        if metric in ['PSNR', 'ALL']:
            mprops_psnr, mprops_max_psnr = psnr_mprops_seq(gt_seq_list, pred_seq_list, cfg.DIFFUSION.PRED_MPROPS_FACTOR, chunkRepdPastSeq, cfg.MACROPROPS.EPS)
            metrics_data_dict['PSNR'].append(mprops_psnr)
            metrics_data_dict['MAX-PSNR'].append(mprops_max_psnr)
        if metric in ['SSIM', 'ALL']:
            mprops_ssim, mprops_max_ssim = ssim_mprops_seq(gt_seq_list, pred_seq_list, cfg.DIFFUSION.PRED_MPROPS_FACTOR, chunkRepdPastSeq)
            metrics_data_dict['SSIM'].append(mprops_ssim)
            metrics_data_dict['MAX-SSIM'].append(mprops_max_ssim)
        if metric in ['MOTION_FEAT_MSE', 'MOTION_FEAT_BHATT', 'ALL']:
            mse_flag = metric == 'MOTION_FEAT_MSE' or metric == 'ALL'
            bhatt_flag = metric == 'MOTION_FEAT_BHATT' or metric == 'ALL'

            mfeat_mse, mfeat_bhatt_dist, mfeat_bhatt_coef = motion_feature_metrics(gt_seq_list, pred_seq_list, cfg.METRICS.MOTION_FEATURE.f, cfg.METRICS.MOTION_FEATURE.k, cfg.METRICS.MOTION_FEATURE.GAMMA, mse_flag, bhatt_flag)

            if mse_flag:
                metrics_data_dict["MOTION_FEAT_MSE"].append(mfeat_mse)
            if bhatt_flag:
                metrics_data_dict["MOTION_FEAT_BHATT_DIST"].append(mfeat_bhatt_dist)
                metrics_data_dict["MOTION_FEAT_BHATT_COEF"].append(mfeat_bhatt_coef)
        count_batch += 1
        if count_batch == batches_to_use:
            break

    title = f"{cfg.DATASET.BATCH_SIZE * chunkRepdPastSeq * batches_to_use} samples in total (BS:{cfg.DATASET.BATCH_SIZE}, Rep:{chunkRepdPastSeq}, TB:{batches_to_use})"
    save_all_metrics(match, metrics_data_dict, metrics_header_dict, title)
    save_all_boxplots_metrics(metrics_data_dict, metrics_header_dict, title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to generate metrics from a trained model.")
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
