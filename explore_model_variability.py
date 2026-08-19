import argparse
import torch
import matplotlib.pyplot as plt
import sys, logging

from models.diffusion.ddpm import DDPM_model
from models.convRNN.convRNN import ConvRNN_model
from models.flow_matching.flow_matching import FM_model
from utils.utils import get_filenames_paths, get_test_dataset, get_model_fullname, get_output_dir
from utils.plot.plot_sampled_mprops import MacropropPlotter
from utils.myparser import getYamlConfig

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler("logs/modVariability.log"),
                        logging.StreamHandler(sys.stdout)]
                    )

def variability_ddpm(cfg, args, batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, mprops_count, total_samples, chunkRepdPastSeq):
    torch.manual_seed(42)
    output_dir = get_output_dir(cfg, args)
    macropropPlotter = MacropropPlotter(cfg, output_dir, arch=args.arch, velScale=args.vel_scale, velUncScale=args.vel_unc_scale, headwidth=args.headwidth)
    
    ddpm_model = DDPM_model(cfg, args.arch, mprops_count, output_dir, args.from_fixed_past)
    ddpm_model.explore_variability(batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, macropropPlotter, total_samples, chunkRepdPastSeq)

def variability_fm(cfg, args, batched_test_data):
    torch.manual_seed(42)
    output_dir = get_output_dir(cfg, args)

def variability_convRNN(cfg, args, batched_test_data):
    torch.manual_seed(42)
    output_dir = get_output_dir(cfg, args)

def model_variability_mgmt(args, cfg):
    """
    Model variability management function.
    """
    torch.manual_seed(42)
    # === Prepare file paths ===
    filenames_and_numSamples = get_filenames_paths(cfg)
    model_fullname = get_model_fullname(cfg, args.arch, args.model_sample_to_load)

    # === Load test dataset ===
    mprops_count = 4 if args.arch == "ConvRNN" else 3
    batched_test_data = get_test_dataset(cfg, filenames_and_numSamples, mprops_count, from_fixed_past=args.from_fixed_past)
    
    # === Set total_samples ===
    if args.chunk_repd_past_seq == None:
        total_samples = cfg.MODEL.NSAMPLES
        chunkRepdPastSeq = 20
    else:
        total_samples = batched_test_data.batch_size*args.chunk_repd_past_seq
        chunkRepdPastSeq = args.chunk_repd_past_seq

    logging.info(f"Total samples to predict:{total_samples}")

    # === Generate samples per architecture ===
    logging.info(f"=======>>>> Init sampling for {cfg.DATASET.NAME} dataset with {args.arch} architecture.")
    if args.arch in ["DDPM-UNet", "DDPM-DiT"]:
        variability_ddpm(cfg, args, batched_test_data, args.plot_type, model_fullname, args.plot_mprop, args.plot_past, args.same_past_seq, mprops_count)
    elif args.arch in ["FM-UNet", "FM-DiT"]:
        variability_fm(cfg, args, batched_test_data, args.plot_type, model_fullname, args.plot_mprop, args.plot_past, args.same_past_seq, mprops_count)
    elif args.arch == "ConvRNN":
        variability_convRNN(cfg, args, batched_test_data, args.plot_type, model_fullname, args.plot_mprop, args.plot_past, args.same_past_seq, mprops_count)
    else:
        logging.error("Architecture not supported.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to sample crowd macroprops from trained model.")
    parser.add_argument('--plot-mprop', type=str, default="Density&Vel", help='Macroprops to be plotted. Options: Density|Uncertainty|Density&Vel')
    parser.add_argument('--plot-past', type=str, default='Last2', help='Past macroprops to be plotted')
    parser.add_argument('--vel-scale', type=float, default=0.5, help='Scale to be applied to velocity mprops vectors')
    parser.add_argument('--headwidth', type=int, default=5, help='Headwidth to be applied to velocity mprops vectors')
    parser.add_argument('--vel-unc-scale', type=int, default=1, help='Scale to be applied to velocity uncertainty mprops vectors')
    parser.add_argument('--plot-type', type=str, default='Static', help='Macroprops plot type can be static (.svg) or dinamic (.gif)')
    parser.add_argument('--config-yml-file', type=str, default='config/4test/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/4test/ATC_ddpm_datafiles.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--model-sample-to-load', type=str, default="000", help='Model sample to be used for generate mprops samples. Default value is for best model.')
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|DDPM-DiT|FM-UNet|FM-DiT|ConvRNN')
    parser.add_argument('--chunk-repd-past-seq', type=int, default=None, help='Chunk of repeteaded past sequences to use when predict.')
    parser.add_argument('--from-fixed-past', type=bool, default=False, help='Compute model variability from fixed past seqs for comparison.')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    model_variability_mgmt(args, cfg)