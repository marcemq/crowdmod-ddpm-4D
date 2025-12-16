import argparse
import torch
import logging

from utils.myparser import getYamlConfig
from utils.utils import get_filenames_paths, get_test_dataset, get_model_fullname
from models.diffusion.ddpm import DDPM_model
from models.convGRU.convGRU import ConvGRU_model
from models.flow_matching.flow_matching import FM_model

def generate_metrics_ddpm(cfg, args, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, mprops_count):
    torch.manual_seed(42)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_modelE{args.model_sample_to_load}_samp{cfg.MODEL.DDPM.SAMPLER}"

    ddpm_model = DDPM_model(cfg, args.arch, mprops_count, output_dir)
    ddpm_model.generate_metrics(batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir)

def generate_metrics_fm(cfg, args, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, mprops_count):
    torch.manual_seed(42)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_modelE{args.model_sample_to_load}_{cfg.MODEL.FLOW_MATCHING.W_TYPE}_intg{cfg.MODEL.FLOW_MATCHING.INTEGRATOR}"

    fm_model = FM_model(cfg, args.arch, mprops_count, output_dir)
    fm_model.generate_metrics(batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir)

def generate_metrics_convGRU(cfg, args, batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, mprops_count):
    torch.manual_seed(42)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_modelE{args.model_sample_to_load}"

    convGRU_model = ConvGRU_model(cfg, args.arch, mprops_count, output_dir)
    convGRU_model.generate_metrics(batched_test_data, chunkRepdPastSeq, metric, batches_to_use, samples_per_batch, model_fullname, output_dir)

def metrics_mgmt(args, cfg):
    """
    Metrics compute management function.
    """
    # === Prepare file paths ===
    filenames_and_numSamples = get_filenames_paths(cfg)
    model_fullname = get_model_fullname(cfg, args.arch, args.model_sample_to_load)

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
        generate_metrics_ddpm(cfg, args, batched_test_data, chunkRepdPastSeq, args.metric, args.batches_to_use, samples_per_batch, model_fullname, mprops_count=mprops_count)
    elif args.arch == "FM-UNet":
        generate_metrics_fm(cfg, args, batched_test_data, chunkRepdPastSeq, args.metric, args.batches_to_use, samples_per_batch, model_fullname, mprops_count=mprops_count)
    elif args.arch == "ConvGRU":
        generate_metrics_convGRU(cfg, args, batched_test_data, chunkRepdPastSeq, args.metric, args.batches_to_use, samples_per_batch, model_fullname, mprops_count=mprops_count)
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