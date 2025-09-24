import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os, re
import logging

from tqdm import tqdm
from models.generate import generate_ddpm, generate_ddim, generate_convGRU

from utils.myparser import getYamlConfig
from utils.utils import create_directory, get_filenames_paths, get_test_dataset, get_model_fullname
from utils.plot.plot_metrics import createBoxPlot, createBoxPlot_bhatt, merge_and_plot_boxplot
from utils.metrics.metricsGenerator import MetricsGenerator
from utils.metrics.computeMetrics import psnr_mprops_seq, ssim_mprops_seq, motion_feature_metrics, energy_mprops_seq, re_density_mprops_seq
from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM
from models.convGRU.forecaster import Forecaster

def save_metric_data(match, data, metric, header, samples_per_batch, output_dir):
    file_name = f"{output_dir}/mpSampling_{metric}_NS{samples_per_batch}_{match.group()}.csv"
    np.savetxt(file_name, data, delimiter=",", header=header, comments="")
    return file_name

def save_all_metrics(match, metrics_data_dict, metrics_header_dict, title, samples_per_batch, output_dir):
    metrics_filenames_dict = {"title": title}
    # Stack metrics by epoch into an array
    for metric_name, metric_data_list in metrics_data_dict.items():
        logging.info(f"Metric: {metric_name}, Number of Entries: {len(metric_data_list)}")
        if len(metric_data_list) != 0:
            metrics_data_dict[metric_name] = np.vstack(metric_data_list)

    # Save each non-empty metric with its required data
    for metric_name, metric_header in metrics_header_dict.items():
        if len(metrics_data_dict[metric_name]) != 0:
            file_name = save_metric_data(match, metrics_data_dict[metric_name], metric_name, metric_header, samples_per_batch, output_dir)
            metrics_filenames_dict[metric_name] = file_name

    with open(f"{output_dir}/metrics_files.json", "w") as json_file:
        json.dump(metrics_filenames_dict, json_file)
    logging.info(f"Dictionary of metrics filenames saved to '{output_dir}/metrics_files.json'")

def save_all_boxplots_metrics(metrics_data_dict, metrics_header_dict, title, output_dir):
    # Convert the dictionary of arrays into a dictionary of DataFrames
    metrics_df_dict = {key: pd.DataFrame(value, columns=metrics_header_dict[key].split(",")) for key, value in metrics_data_dict.items()}
    if len(metrics_df_dict['MAX-PSNR']) != 0:
        merge_and_plot_boxplot(df_max=metrics_df_dict['MAX-PSNR'], df=metrics_df_dict['PSNR'], title=f"PSNR and MAX-PSNR of {title}", save_path=f"{output_dir}/BP_PSNR.png", ytick_step=5)
    if len(metrics_df_dict['MAX-SSIM']) != 0:
        merge_and_plot_boxplot(df_max=metrics_df_dict['MAX-SSIM'], df=metrics_df_dict['SSIM'], title=f"SSIM and MAX-SSIM of {title}", save_path=f"{output_dir}/BP_SSIM.png", ytick_step=0.2)
    if len(metrics_df_dict['MOTION_FEAT_MSE']) != 0:
        createBoxPlot(metrics_df_dict['MOTION_FEAT_MSE'], title=f"MSE of Motion feature of {title}", columns_to_plot=metrics_header_dict["MOTION_FEAT_MSE"].split(","), save_path=f"{output_dir}/BP_MF_MSE.png", ytick_step=0.0002)
    if len(metrics_df_dict['MOTION_FEAT_BHATT_COEF']) != 0:
        createBoxPlot_bhatt(metrics_df_dict['MOTION_FEAT_BHATT_COEF'], metrics_df_dict['MOTION_FEAT_BHATT_DIST'], title=f"BHATT of Motion feature of {title}", save_path=f"{output_dir}/BP_BHATT.png")
    if len(metrics_df_dict['MIN-ENERGY']) != 0:
        merge_and_plot_boxplot(df_max=metrics_df_dict['MIN-ENERGY'], df=metrics_df_dict['ENERGY'], title=f"ENERGY and MIN-ENERGY of {title}", save_path=f"{output_dir}/BP_ENERGY.png", ytick_step=None, prefix='min-')
    if len(metrics_df_dict['MIN_RE_DENSITY']) != 0:
        merge_and_plot_boxplot(df_max=metrics_df_dict['MIN_RE_DENSITY'], df=metrics_df_dict['RE_DENSITY'], title=f"Relative DENSITY and MIN_RE_DENSITY of {title}", save_path=f"{output_dir}/BP_RE_DENSITY.png", ytick_step=2, prefix='min-', outliersFlag=True)

def get_metrics_dicts():
    metrics_data_dict = {"PSNR" : [],
                    "MAX-PSNR" : [],
                    "SSIM" : [],
                    "MAX-SSIM" : [],
                    "MOTION_FEAT_MSE" : [],
                    "MOTION_FEAT_BHATT_DIST" : [],
                    "MOTION_FEAT_BHATT_COEF" : [],
                    "ENERGY" : [],
                    "MIN-ENERGY" : [],
                    "RE_DENSITY" : [],
                    "MIN_RE_DENSITY" : []
                    }
    metrics_header_dict = {"PSNR" : "rho,vx,vy",
                    "MAX-PSNR" : "rho,vx,vy",
                    "SSIM" : "rho,vx,vy",
                    "MAX-SSIM" : "rho,vx,vy",
                    "MOTION_FEAT_MSE" : "MSE_Hist_2D_Based,MSE_Hist_1D_Based",
                    "MOTION_FEAT_BHATT_DIST" : "BHATT_DIST_Hist_2D_Based,BHATT_DIST_Hist_1D_Based",
                    "MOTION_FEAT_BHATT_COEF" : "BHATT_COEF_Hist_2D_Based,BHATT_COEF_Hist_1D_Based",
                    "ENERGY" : "GT,PRED",
                    "MIN-ENERGY" : "GT,PRED",
                    "RE_DENSITY" : "re_f6,re_f7,re_f8",
                    "MIN_RE_DENSITY" : "re_f6,re_f7,re_f8"
                    }
    return metrics_data_dict, metrics_header_dict

def compute_metrics(cfg, metricsGenerator, metric, chunkRepdPastSeq, match, batches_to_use, samples_per_batch) :
    if metric in ['PSNR', 'ALL']:
        metricsGenerator.compute_psnr_metric(chunkRepdPastSeq, cfg.MACROPROPS.EPS)
    if metric in ['SSIM', 'ALL']:
        metricsGenerator.compute_ssim_metric(chunkRepdPastSeq)
    if metric in ['MF_MSE', 'MF_BHATT', 'ALL']:
        mse_flag = metric == 'MF_MSE' or metric == 'ALL'
        bhatt_flag = metric == 'MF_BHATT' or metric == 'ALL'
        metricsGenerator.compute_motion_feature_metrics(mse_flag, bhatt_flag)
    if metric in ['ENERGY', 'ALLA']:
         metricsGenerator.compute_energy_metric()

    title = f"{cfg.DATASET.BATCH_SIZE * chunkRepdPastSeq * batches_to_use} samples in total (BS:{cfg.DATASET.BATCH_SIZE}, Rep:{chunkRepdPastSeq}, TB:{batches_to_use})-(DDPM-UNet)"
    metricsGenerator.save_data_metrics(match, title, samples_per_batch)
    metricsGenerator.save_metrics_boxplots(title)

def compute_metrics_base(metric, gt_seq_list, pred_seq_list, metrics_data_dict, chunkRepdPastSeq, output_dir):
    if metric in ['PSNR', 'ALL']:
        mprops_psnr, mprops_max_psnr = psnr_mprops_seq(gt_seq_list, pred_seq_list, cfg.METRICS.PRED_MPROPS_FACTOR, chunkRepdPastSeq, cfg.MACROPROPS.EPS, cfg.METRICS.MPROPS_COUNT)
        metrics_data_dict['PSNR'].append(mprops_psnr)
        metrics_data_dict['MAX-PSNR'].append(mprops_max_psnr)
    if metric in ['SSIM', 'ALL']:
        mprops_ssim, mprops_max_ssim = ssim_mprops_seq(gt_seq_list, pred_seq_list, cfg.METRICS.PRED_MPROPS_FACTOR, chunkRepdPastSeq, cfg.METRICS.MPROPS_COUNT)
        metrics_data_dict['SSIM'].append(mprops_ssim)
        metrics_data_dict['MAX-SSIM'].append(mprops_max_ssim)
    if metric in ['MOTION_FEAT_MSE', 'MOTION_FEAT_BHATT', 'ALL']:
        mse_flag = metric == 'MOTION_FEAT_MSE' or metric == 'ALL'
        bhatt_flag = metric == 'MOTION_FEAT_BHATT' or metric == 'ALL'

        mfeat_mse, mfeat_bhatt_dist, mfeat_bhatt_coef = motion_feature_metrics(gt_seq_list, pred_seq_list, cfg.METRICS.MOTION_FEATURE.f, cfg.METRICS.MOTION_FEATURE.k, cfg.METRICS.MOTION_FEATURE.GAMMA, mse_flag, bhatt_flag, output_dir)

        if mse_flag:
            metrics_data_dict["MOTION_FEAT_MSE"].append(mfeat_mse)
        if bhatt_flag:
            metrics_data_dict["MOTION_FEAT_BHATT_DIST"].append(mfeat_bhatt_dist)
            metrics_data_dict["MOTION_FEAT_BHATT_COEF"].append(mfeat_bhatt_coef)
    #if metric in ['ENERGY', 'ALL']:
    #    mprops_energy, mprops_min_energy = energy_mprops_seq(gt_seq_list, pred_seq_list, cfg.METRICS.PRED_MPROPS_FACTOR, chunkRepdPastSeq, cfg.METRICS.MPROPS_COUNT)
    #    metrics_data_dict['ENERGY'].append(mprops_energy)
    #    metrics_data_dict['MIN-ENERGY'].append(mprops_min_energy)
    if metric in ['RE_DENSITY', 'ALL']:
        mprops_re_density, mprops_min_re_density = re_density_mprops_seq(gt_seq_list, pred_seq_list, chunkRepdPastSeq, cfg.MACROPROPS.EPS)
        metrics_data_dict['RE_DENSITY'].append(mprops_re_density)
        metrics_data_dict['MIN_RE_DENSITY'].append(mprops_min_re_density)
    #AR not sure if I have to return metrics_data_dict

def generate_metrics_ddpm(cfg, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir, mprops_count):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instanciate the UNet for the reverse diffusion
    denoiser = MacropropsDenoiser(input_channels  = mprops_count,
                                  output_channels = mprops_count,
                                  num_res_blocks  = cfg.MODEL.DDPM.UNET.NUM_RES_BLOCKS,
                                  base_channels           = cfg.MODEL.DDPM.UNET.BASE_CH,
                                  base_channels_multiples = cfg.MODEL.DDPM.UNET.BASE_CH_MULT,
                                  apply_attention         = cfg.MODEL.DDPM.UNET.APPLY_ATTENTION,
                                  dropout_rate            = cfg.MODEL.DDPM.UNET.DROPOUT_RATE,
                                  time_multiple           = cfg.MODEL.DDPM.UNET.TIME_EMB_MULT,
                                  condition               = cfg.MODEL.DDPM.UNET.CONDITION)

    # Load model
    logging.info(f'model full name:{model_fullname}')
    denoiser.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'), weights_only=True)['model'])
    denoiser.to(device)
    match = re.search(r'TE\d+_PL\d+_FL\d+_CE\d+_VN[FT]', model_fullname)

    # Instantiate the diffusion model
    timesteps = cfg.MODEL.DDPM.TIMESTEPS
    diffusionmodel = DDPM(timesteps=cfg.MODEL.DDPM.TIMESTEPS)
    diffusionmodel.to(device)
    taus = 1
    count_batch = 0
    pred_seq_list, gt_seq_list = [], []
    # cicle over batched test data
    for batch in batched_test_data:
        logging.info("===" * 20)
        logging.info(f'Computing sampling on batch:{count_batch+1}')
        past_test, future_test = batch
        past_test, future_test = past_test.float(), future_test.float()
        past_test, future_test = past_test.to(device=device), future_test.to(device=device)
        # Compute the idx of the past sequences to work on
        if past_test.shape[0] < samples_per_batch:
            random_past_idx = torch.randperm(past_test.shape[0])
        else:
            random_past_idx = torch.randperm(past_test.shape[0])[:samples_per_batch]
        expanded_random_past_idx = torch.repeat_interleave(random_past_idx, chunkRepdPastSeq)
        random_past_idx = expanded_random_past_idx[:samples_per_batch]
        random_past_samples = past_test[random_past_idx]
        random_future_samples = future_test[random_past_idx]

        if cfg.MODEL.DDPM.SAMPLER == "DDPM":
            x, _  = generate_ddpm(denoiser, random_past_samples, diffusionmodel, cfg, device, samples_per_batch, mprops_count=mprops_count) # AR review .cpu() call here
            if cfg.MODEL.DDPM.GUIDANCE == "sparsity" or cfg.MODEL.DDPM.GUIDANCE=="mass_preservation" or cfg.MODEL.DDPM.GUIDANCE == "None":
                l1 = torch.mean(torch.abs(x[:,0,:,:,:])).cpu().detach().numpy()
                logging.info(f'L1 norm {l1:.2f} using {cfg.MODEL.DDPM.GUIDANCE} guidance')
        elif cfg.MODEL.DDPM.SAMPLER == "DDIM":
            taus = np.arange(0,timesteps,cfg.MODEL.DDPM.DDIM_DIVIDER)
            logging.info(f'taus:{taus}')
            x, _ = generate_ddim(denoiser, random_past_samples, taus, diffusionmodel, cfg, device, samples_per_batch, mprops_count=mprops_count) # AR review .cpu() call here
        else:
            logging.info(f"{cfg.MODEL.DDPM.SAMPLER} sampler not supported")

        future_samples_pred = x
        for i in range(len(random_past_idx)):
            pred_seq_list.append(future_samples_pred[i])
            gt_seq_list.append(random_future_samples[i])

        count_batch += 1
        if count_batch == batches_to_use:
            break

    logging.info("===" * 20)
    logging.info(f'Computing metrics on predicted mprops sequences.')
    metricsGenerator = MetricsGenerator(pred_seq_list, gt_seq_list, cfg.METRICS, output_dir)
    compute_metrics(cfg, metricsGenerator, metric, chunkRepdPastSeq, match, batches_to_use, samples_per_batch)

def generate_metrics_convGRU(cfg, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir, mprops_count):
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instanciate the convGRU forcaster model
    convGRU_model = Forecaster(input_size  = (cfg.MACROPROPS.ROWS, cfg.MACROPROPS.COLS),
                               input_channels       = mprops_count,
                               enc_hidden_channels  = cfg.MODEL.CONVGRU.ENC_HIDDEN_CH,
                               forc_hidden_channels = cfg.MODEL.CONVGRU.FORC_HIDDEN_CH,
                               enc_kernels          = cfg.MODEL.CONVGRU.ENC_KERNELS,
                               forc_kernels         = cfg.MODEL.CONVGRU.FORC_KERNELS,
                               device               = device,
                               bias                 = False)

    # Load model
    logging.info(f'model full name:{model_fullname}')
    convGRU_model.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'), weights_only=True)['model'])
    convGRU_model.to(device)

    count_batch = 0
    metrics_data_dict, metrics_header_dict = get_metrics_dicts()
    with tqdm(total=batches_to_use) as tq:
        # scan test batches
        for batch in batched_test_data:
            tq.set_description(f"Compute Metrics :: batch: {count_batch+1}/{batches_to_use}")
            tq.update(1)
            past_test, future_test = batch
            past_test, future_test = past_test.float(), future_test.float()
            past_test, future_test = past_test.to(device=device), future_test.to(device=device)
            # Compute the idx of the past sequences to work on
            if past_test.shape[0] < samples_per_batch:
                random_past_idx = torch.randperm(past_test.shape[0])
            else:
                random_past_idx = torch.randperm(past_test.shape[0])[:samples_per_batch]
            expanded_random_past_idx = torch.repeat_interleave(random_past_idx, chunkRepdPastSeq)
            random_past_idx = expanded_random_past_idx[:samples_per_batch]
            random_past_samples = past_test[random_past_idx]
            random_future_samples = future_test[random_past_idx]
            predictions = generate_convGRU(convGRU_model, random_past_samples, random_future_samples, teacher_forcing=False)

            # mprops setup for metrics compute
            random_future_samples = random_future_samples[:, :cfg.METRICS.MPROPS_COUNT, :, :, :]
            predictions = predictions[:, :cfg.METRICS.MPROPS_COUNT, :, :, :]

            pred_seq_list, gt_seq_list = [], []
            for i in range(len(random_past_idx)):
                pred_seq_list.append(predictions[i])
                gt_seq_list.append(random_future_samples[i])

            compute_metrics(metric, gt_seq_list, pred_seq_list, metrics_data_dict, chunkRepdPastSeq)
            count_batch += 1
            if count_batch == batches_to_use:
                break

    match = re.search(r'TE\d+_PL\d+_FL\d+_CE\d+_VN[FT]', model_fullname)
    title = f"{cfg.DATASET.BATCH_SIZE * chunkRepdPastSeq * batches_to_use} samples in total (BS:{cfg.DATASET.BATCH_SIZE}, Rep:{chunkRepdPastSeq}, TB:{batches_to_use})-(ConvGRU)"
    save_all_metrics(match, metrics_data_dict, metrics_header_dict, title, samples_per_batch, output_dir)
    save_all_boxplots_metrics(metrics_data_dict, metrics_header_dict, title, output_dir)

def metrics_mgmt(args, cfg):
    """
    Metrics compute management function.
    """
    # === Prepare file paths ===
    filenames_and_numSamples = get_filenames_paths(cfg)
    model_fullname = get_model_fullname(cfg, args.arch, args.model_sample_to_load)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_VN{cfg.DATASET.VELOCITY_NORM}_modelE{args.model_sample_to_load}"
    create_directory(output_dir)

    # === Load test dataset ===
    mprops_count = 4 if args.arch == "ConvGRU" else 3
    batched_test_data = get_test_dataset(cfg, filenames_and_numSamples, mprops_count)

    # === Set samples_per_batch ===
    if args.chunk_repd_past_seq == None:
        samples_per_batch = cfg.MODEL.NSAMPLES
        chunkRepdPastSeq = 20
    else:
        samples_per_batch = cfg.DATASET.BATCH_SIZE*args.chunk_repd_past_seq
        chunkRepdPastSeq = args.chunk_repd_past_seq

    # === Generate metrics per architecture ===
    logging.info(f"=======>>>> Init metrics compute for {cfg.DATASET.NAME} dataset with {args.arch} architecture.")
    if args.arch == "DDPM-UNet":
        generate_metrics_ddpm(cfg, batched_test_data, chunkRepdPastSeq, args.metric, args.batches_to_use, samples_per_batch, model_fullname, output_dir, mprops_count=mprops_count)
    elif args.arch == "ConvGRU":
        generate_metrics_convGRU(cfg, batched_test_data, chunkRepdPastSeq, args.metric, args.batches_to_use, samples_per_batch, model_fullname, output_dir, mprops_count=mprops_count)
    else:
        logging.error("Architecture not supported.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to generate metrics from a trained model.")
    parser.add_argument('--chunk-repd-past-seq', type=int, default=None, help='Chunk of repeteaded past sequences to use when predict.')
    parser.add_argument('--metric', type=str, default='ALL', help='Name of the metric to compute, options: PSNR|SSIM|MOTION_FEAT_BHATT|ENERGY|RE_DENSITY|ALL')
    parser.add_argument('--batches-to-use', type=int, default=1, help='Total of batches to use to compute metrics.')
    parser.add_argument('--config-yml-file', type=str, default='config/4test/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/4test/ATC_ddpm_datafiles.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--model-sample-to-load', type=str, default="000", help='Model sample to be used for generate mprops samples.')
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|ConvGRU')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    metrics_mgmt(args, cfg)