import argparse
import torch
import matplotlib.pyplot as plt
import logging

from models.diffusion.ddpm import DDPM_model
from models.convGRU.convGRU import ConvGRU_model
from models.flow_matching.flow_matching import FM_model
from utils.utils import get_filenames_paths, get_test_dataset, get_model_fullname
from utils.plot.plot_sampled_mprops import MacropropPlotter
from utils.myparser import getYamlConfig
from torchvision.utils import make_grid

def inverseTransform(y, stats):
    stats = stats[0,:,:]
    yt = torch.randn_like(y)
    for i in range(1, len(stats)-1):
        yt[:,i,:,:] = (y[:,i,:,:] + 1)/2 * (stats[i,3] - stats[i,2]) + stats[i,2]
    return yt

def getGrid(x, cols, mode="RGB", showGrid=False):
    "x shape: NsamplesxCHxRxC"
    # Set as grid and show
    #mp_as_img = x[0,:,:,:].permute(3,0,1,2)
    if mode == "RGB":
        mp_as_img = x[:,0:3,:,:]
    elif mode == "GRAY":
        mp_as_img = x[:,0:1,:,:]

    grid_img = make_grid(mp_as_img, nrow=cols, padding=True, pad_value=1, normalize=True)
    if showGrid:
        plt.imshow(grid_img, cmap='gray')
        plt.axis("off")
        plt.show()
    return grid_img

def generate_samples_ddpm(cfg, batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, mprops_count):
    torch.manual_seed(42)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_modelE{args.model_sample_to_load}_samp{cfg.MODEL.DDPM.SAMPLER}"
    macropropPlotter = MacropropPlotter(cfg, output_dir, arch=args.arch, velScale=args.vel_scale, velUncScale=args.vel_unc_scale, headwidth=args.headwidth)

    ddpm_model = DDPM_model(cfg, args.arch, mprops_count, output_dir)
    ddpm_model.sampling(batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, macropropPlotter)

def generate_samples_fm(cfg, args, batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, mprops_count):
    torch.manual_seed(42)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_modelE{args.model_sample_to_load}_{cfg.MODEL.FLOW_MATCHING.W_TYPE}_intg{cfg.MODEL.FLOW_MATCHING.INTEGRATOR}"
    macropropPlotter = MacropropPlotter(cfg, output_dir, arch=args.arch, velScale=args.vel_scale, velUncScale=args.vel_unc_scale, headwidth=args.headwidth)

    fm_model = FM_model(cfg, args.arch, mprops_count, output_dir)
    fm_model.sampling(batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, macropropPlotter)

def generate_samples_convGRU(cfg, args, batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, mprops_count):
    torch.manual_seed(42)
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_modelE{args.model_sample_to_load}"
    macropropPlotter = MacropropPlotter(cfg, output_dir, arch=args.arch, velScale=args.vel_scale, velUncScale=args.vel_unc_scale, headwidth=args.headwidth)

    convGRU_model = ConvGRU_model(cfg, args.arch, mprops_count, output_dir)
    convGRU_model.sampling( batched_test_data, plotType, model_fullname, plotMprop, plotPast, samePastSeq, macropropPlotter)

def sampling_mgmt(args, cfg):
    """
    Sampling management function.
    """
    # === Prepare file paths ===
    filenames_and_numSamples = get_filenames_paths(cfg)
    model_fullname = get_model_fullname(cfg, args.arch, args.model_sample_to_load)

    # === Load test dataset ===
    mprops_count = 4 if args.arch == "ConvGRU" else 3
    batched_test_data = get_test_dataset(cfg, filenames_and_numSamples, mprops_count)

    # === Generate samples per architecture ===
    logging.info(f"=======>>>> Init sampling for {cfg.DATASET.NAME} dataset with {args.arch} architecture.")
    if args.arch == "DDPM-UNet":
        generate_samples_ddpm(cfg, batched_test_data, args.plot_type, model_fullname, args.plot_mprop, args.plot_past, args.same_past_seq, mprops_count)
    elif args.arch == "FM-UNet":
        generate_samples_fm(cfg, args, batched_test_data, args.plot_type, model_fullname, args.plot_mprop, args.plot_past, args.same_past_seq, mprops_count)
    elif args.arch == "ConvGRU":
        generate_samples_convGRU(cfg, args, batched_test_data, args.plot_type, model_fullname, args.plot_mprop, args.plot_past, args.same_past_seq, mprops_count)
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
    parser.add_argument('--same-past-seq', type=bool, default=False, help='Use the same past sequence to predict different mprops from it.')
    parser.add_argument('--config-yml-file', type=str, default='config/4test/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/4test/ATC_ddpm_datafiles.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--model-sample-to-load', type=str, default="000", help='Model sample to be used for generate mprops samples. Default value is for best model.')
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|FM-UNet|ConvGRU')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    sampling_mgmt(args, cfg)

# execution example:
# python3 generate_samples.py --plot-mprop="Density" --plot-past="Last2"
# python3 generate_samples.py --plot-mprop="Density&Vel" --plot-past="Last2" --vel-scale=0.25
# python3 generate_samples.py --plot-mprop="Uncertainty" --plot-past="Last2" --vel-unc-scale=3
# python3 generate_samples.py --plot-type="Dynamic" --vel-scale=0.25 --vel-unc-scale=3
# python3 generate_samples.py --plot-mprop="Density" --plot-past="Last2" --same-past-seq=True