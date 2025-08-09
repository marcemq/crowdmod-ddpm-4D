import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os, re
import logging
from models.generate import generate_ddpm, generate_ddim, generate_convGRU

from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM
from models.convGRU.forecaster import Forecaster
from utils.dataset import getDataset, getClassicDataset, getDataset4Test
from utils.utils import create_directory
from utils.plot.plot_sampled_mprops import plotStaticMacroprops, plotDynamicMacroprops, plotDensityOverTime
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

def set_predictions_plot(predictions, random_past_idx, random_past_samples, random_future_samples, model_fullname, plotType, plotMprop, plotPast, velScale, velUncScale, headwidth, output_dir):
    seq_frames = []
    for i in range(len(random_past_idx)):
        future_sample_pred = predictions[i]
        future_sample_gt = random_future_samples[i]
        past_sample = random_past_samples[i]
        seq_pred = torch.cat([past_sample, future_sample_pred], dim=3)
        seq_gt = torch.cat([past_sample, future_sample_gt], dim=3)
        seq_frames.append(seq_pred)
        seq_frames.append(seq_gt)

    match = re.search(r'TE\d+_PL\d+_FL\d+_CE\d+_VN[FT]', model_fullname)
    if plotType == "Static":
        plotStaticMacroprops(seq_frames, cfg, match, plotMprop, plotPast, velScale, velUncScale, output_dir)
    elif plotType == "Dynamic":
        plotDynamicMacroprops(seq_frames, cfg, velScale, headwidth, output_dir)

    plotDensityOverTime(seq_frames, cfg, output_dir)

def generate_samples_ddpm(cfg, batched_test_data, plotType, output_dir, model_fullname, plotMprop, plotPast, velScale, velUncScale, samePastSeq, headwidth, mprops_count):
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

    logging.info(f'model full name:{model_fullname}')
    denoiser.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'))['model'])
    denoiser.to(device)

    # Instantiate the diffusion model 
    timesteps=cfg.MODEL.DDPM.TIMESTEPS
    diffusionmodel = DDPM(timesteps=cfg.MODEL.DDPM.TIMESTEPS)
    diffusionmodel.to(device)
    taus = 1

    for batch in batched_test_data:
        past_test, future_test, stats = batch
        past_test, future_test = past_test.float(), future_test.float()
        past_test, future_test = past_test.to(device=device), future_test.to(device=device)
        #x_train, y_train, stats = batch
        random_past_idx = torch.randperm(past_test.shape[0])[:cfg.MODEL.NSAMPLES4PLOTS]
        # Predict different sequences for the same past sequence
        if samePastSeq:
            fixed_past_idx = random_past_idx[0]
            random_past_idx.fill_(fixed_past_idx)

        random_past_samples = past_test[random_past_idx]
        random_future_samples = future_test[random_past_idx]
        if cfg.MODEL.DDPM.SAMPLER == "DDPM":
            predictions, xnoisy_over_time  = generate_ddpm(denoiser, random_past_samples, diffusionmodel, cfg, device, cfg.MODEL.NSAMPLES4PLOTS, mprops_count=mprops_count) # AR review .cpu() call here
            if cfg.MODEL.DDPM.GUIDANCE == "sparsity" or cfg.MODEL.DDPM.GUIDANCE == "none":
                l1 = torch.mean(torch.abs(predictions[:,0,:,:,:])).cpu().detach().numpy()
                logging.info('L1 norm {:.2f}'.format(l1))
        elif cfg.MODEL.DDPM.SAMPLER == "DDIM":
            taus = np.arange(0,timesteps,cfg.MODEL.DDPM.DDIM_DIVIDER)
            logging.info(f'taus:{taus}')
            predictions, xnoisy_over_time = generate_ddim(denoiser, random_past_samples, taus, diffusionmodel, cfg, device, cfg.MODEL.NSAMPLES4PLOTS, mprops_count=mprops_count) # AR review .cpu() call here
        else:
            logging.info(f"{cfg.MODEL.DDPM.SAMPLER} sampler not supported")

        set_predictions_plot(predictions, random_past_idx, random_past_samples, random_future_samples, model_fullname, plotType, plotMprop, plotPast, velScale, velUncScale, headwidth, output_dir)
        break

def generate_samples_convGRU(cfg, batched_test_data, plotType, output_dir, model_fullname, plotMprop, plotPast, velScale, velUncScale, samePastSeq, headwidth, mprops_count):
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

    logging.info(f'model full name:{model_fullname}')
    convGRU_model.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'))['model'])
    convGRU_model.to(device)

    for batch in batched_test_data:
        past_test, future_test, stats = batch
        past_test, future_test = past_test.float(), future_test.float()
        past_test, future_test = past_test.to(device=device), future_test.to(device=device)
        random_past_idx = torch.randperm(past_test.shape[0])[:cfg.MODEL.NSAMPLES4PLOTS]
        # Predict different sequences for the same past sequence
        if samePastSeq:
            fixed_past_idx = random_past_idx[0]
            random_past_idx.fill_(fixed_past_idx)

        random_past_samples = past_test[random_past_idx]
        random_future_samples = future_test[random_past_idx]
        predictions = generate_convGRU(convGRU_model, random_past_samples, random_future_samples, cfg.MODEL.CONVGRU.TEACHER_FORCING)
        set_predictions_plot(predictions, random_past_idx, random_past_samples, random_future_samples, model_fullname, plotType, plotMprop, plotPast, velScale, velUncScale, headwidth, output_dir)
        break

def sampling_mgmt(args, cfg):
    """
    Sampling management function.
    """
    # === Prepare file paths ===
    filenames = cfg.DATA_LIST
    if cfg.DATASET.NAME in ["ATC", "ATC4TEST"]:
        filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    elif cfg.DATASET.NAME in ["HERMES-BO", "HERMES-CR-120", "HERMES-CR-120-OBST"]:
        filenames = [filename.replace(".txt", ".pkl") for filename in filenames]
    else:
        logging.error("Dataset not supported")

    filenames = [ os.path.join(cfg.DATA_FS.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    model_fullname = cfg.DATA_FS.SAVE_DIR+(cfg.MODEL.NAME.format(args.arch, cfg.TRAIN.EPOCHS, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN, args.model_sample_to_load, cfg.DATASET.VELOCITY_NORM))
    output_dir = f"{cfg.DATA_FS.OUTPUT_DIR}/{args.arch}_VN{cfg.DATASET.VELOCITY_NORM}_modelE{args.model_sample_to_load}"
    create_directory(output_dir)

    # === Load test dataset ===
    mprops_count = 4 if args.arch == "DDPM-UNet" else 3
    if cfg.DATASET.DATASET_TYPE == "BySplitRatio":
        _, batched_test_data = getClassicDataset(cfg, filenames, mprops_count=mprops_count)
    elif cfg.DATASET.DATASET_TYPE == "ByFilenames":
        _, _, batched_test_data = getDataset(cfg, filenames, test_data_only=True, mprops_count=mprops_count)
    else:
        logging.error(f"Dataset type not supported.")
    logging.info(f"Batched Test dataset loaded.")

    # === Generate samples per architecture ===
    if args.arch == "DDPM-UNet":
        generate_samples_ddpm(cfg, batched_test_data, args.plot_type, output_dir, model_fullname, args.plot_mprop, args.plot_past, args.vel_scale, args.vel_unc_scale, args.same_past_seq, args.headwidth, mprops_count=mprops_count)
    elif args.arch == "ConvGRU":
        generate_samples_convGRU(cfg, batched_test_data, args.plot_type, output_dir, model_fullname, args.plot_mprop, args.plot_past, args.vel_scale, args.vel_unc_scale, args.same_past_seq, args.headwidth, mprops_count=mprops_count)
    else:
        logging.error("Architecture not supported.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to sample crowd macroprops from trained model.")
    parser.add_argument('--plot-mprop', type=str, default='Density', help='Macroprops to be plotted')
    parser.add_argument('--plot-past', type=str, default='Last2', help='Past macroprops to be plotted')
    parser.add_argument('--vel-scale', type=float, default=0.5, help='Scale to be applied to velocity mprops vectors')
    parser.add_argument('--headwidth', type=int, default=5, help='Headwidth to be applied to velocity mprops vectors')
    parser.add_argument('--vel-unc-scale', type=int, default=1, help='Scale to be applied to velocity uncertainty mprops vectors')
    parser.add_argument('--plot-type', type=str, default='Static', help='Macroprops plot type can be static (.svg) or dinamic (.gif)')
    parser.add_argument('--same-past-seq', type=bool, default=False, help='Use the same past sequence to predict different mprops from it.')
    parser.add_argument('--config-yml-file', type=str, default='config/4test/ATC_ddpm.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/4test/ATC_ddpm_datafiles.yml',help='Configuration YML macroprops list for specific dataset.')
    parser.add_argument('--model-sample-to-load', type=str, default="000", help='Model sample to be used for generate mprops samples. Default value is for best model.')
    parser.add_argument('--arch', type=str, default='DDPM-UNet', help='Architecture to be used, options: DDPM-UNet|ConvGRU')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    sampling_mgmt(args, cfg)

# execution example:
# python3 generate_samples.py --plot-mprop="Density" --plot-past="Last2"
# python3 generate_samples.py --plot-mprop="Density&Vel" --plot-past="Last2" --vel-scale=0.25
# python3 generate_samples.py --plot-mprop="Uncertainty" --plot-past="Last2" --vel-unc-scale=3
# python3 generate_samples.py --plot-type="Dynamic" --vel-scale=0.25 --vel-unc-scale=3
# python3 generate_samples.py --plot-mprop="Density" --plot-past="Last2" --same-past-seq=True