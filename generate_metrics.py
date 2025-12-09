import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os, re
import logging

from tqdm import tqdm
from models.generate import generate_convGRU

from utils.myparser import getYamlConfig
from utils.utils import create_directory, get_filenames_paths, get_test_dataset, get_model_fullname
from utils.plot.plot_metrics import createBoxPlot, createBoxPlot_bhatt, merge_and_plot_boxplot
from utils.metrics.metricsGenerator import MetricsGenerator
from models.diffusion.ddpm import DDPM_model
from models.convGRU.forecaster import Forecaster
from models.flow_matching.flow_matching import FM_model

def compute_metrics(cfg, metricsGenerator, metric, chunkRepdPastSeq, match, batches_to_use, samples_per_batch, arch="DDPM-UNet") :
    if metric in ['PSNR', 'ALL']:
        metricsGenerator.compute_psnr_metric(chunkRepdPastSeq, cfg.MACROPROPS.EPS)
    if metric in ['SSIM', 'ALL']:
        metricsGenerator.compute_ssim_metric(chunkRepdPastSeq)
    if metric in ['MF_MSE', 'MF_BHATT', 'ALL']:
        mse_flag = metric == 'MF_MSE' or metric == 'ALL'
        bhatt_flag = metric == 'MF_BHATT' or metric == 'ALL'
        metricsGenerator.compute_motion_feature_metrics(mse_flag, bhatt_flag)
    if metric in ['ENERGY', 'ALLA']:
         metricsGenerator.compute_energy_metric(chunkRepdPastSeq)
    if metric in ['RE_DENSITY', 'ALL']:
        metricsGenerator.compute_re_density_metric(chunkRepdPastSeq, cfg.MACROPROPS.EPS)

    title = f"{cfg.DATASET.BATCH_SIZE * chunkRepdPastSeq * batches_to_use} samples in total (BS:{cfg.DATASET.BATCH_SIZE}, Rep:{chunkRepdPastSeq}, TB:{batches_to_use})-({arch})"
    metricsGenerator.save_data_metrics(match, title, samples_per_batch)
    metricsGenerator.save_metrics_boxplots(title)

def generate_metrics_ddpm(cfg, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir, mprops_count):
    torch.manual_seed(42)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_modelE{args.model_sample_to_load}_samp{cfg.MODEL.DDPM.SAMPLER}"

    ddpm_model = DDPM_model(cfg, args.arch, mprops_count, output_dir)
    ddpm_model.generate_metrics(batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir)

def generate_metrics_fm(cfg, arch, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir, mprops_count):
    torch.manual_seed(42)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_modelE{args.model_sample_to_load}_{cfg.MODEL.FLOW_MATCHING.W_TYPE}_intg{cfg.MODEL.FLOW_MATCHING.INTEGRATOR}"

    fm_model = FM_model(cfg, arch, mprops_count, output_dir)
    fm_model.generate_metrics(batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir)

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
    match = re.search(r'TE\d+_PL\d+_FL\d+_CE\d+_VN[FT]', model_fullname)

    count_batch = 0
    pred_seq_list, gt_seq_list = [], []

    with tqdm(total=batches_to_use) as tq:
        # scan test batches
        for batch in batched_test_data:
            tq.set_description(f"Computing sampling on batch: {count_batch+1}/{batches_to_use}")
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

            for i in range(len(random_past_idx)):
                pred_seq_list.append(predictions[i])
                gt_seq_list.append(random_future_samples[i])

            count_batch += 1
            if count_batch == batches_to_use:
                break

    logging.info("===" * 20)
    logging.info(f'Computing metrics on predicted mprops sequences with ConvGRU model.')
    metricsGenerator = MetricsGenerator(pred_seq_list, gt_seq_list, cfg.METRICS, output_dir)
    compute_metrics(cfg, metricsGenerator, metric, chunkRepdPastSeq, match, batches_to_use, samples_per_batch)

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
    elif args.arch == "FM-UNet":
        generate_metrics_fm(cfg, args.arch, batched_test_data, chunkRepdPastSeq, args.metric, args.batches_to_use, samples_per_batch, model_fullname, output_dir, mprops_count=mprops_count)
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
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|FM-UNet|ConvGRU')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    metrics_mgmt(args, cfg)