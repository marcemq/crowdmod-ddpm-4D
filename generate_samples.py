import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os, re
import logging
from models.generate import generate_ddpm, generate_ddim

from models.unet import MacropropsDenoiser
from models.diffusion.ddpm import DDPM
from utils.dataset import getDataset, getClassicDataset, getDataset4Test
from utils.utils import create_directory
from utils.plot_sampled_mprops import plotStaticMacroprops, plotDynamicMacroprops, plotDensityOverTime
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

def generate_samples(cfg, filenames, plotType, plotMprop="Density", plotPast="Last2", velScale=0.5, velUncScale=1, samePastSeq=False, headwidth=5):
    create_directory(cfg.MODEL.OUTPUT_DIR)
    torch.manual_seed(42)
    # Setting the device to work with
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get batched datasets ready to iterate
    if cfg.DATASET.DATASET_TYPE == "BySplitRatio":
        _, batched_test_data = getClassicDataset(cfg, filenames)
    elif cfg.DATASET.DATASET_TYPE == "ByFilenames":
        _, _, batched_test_data = getDataset(cfg, filenames, test_data_only=True)
    elif cfg.DATASET.DATASET_TYPE == "ByFilenames4Test":
        batched_test_data = getDataset4Test(cfg, filenames)

    logging.info(f"Batched Test dataset loaded.")
    # Instanciate the UNet for the reverse diffusion
    denoiser = MacropropsDenoiser(input_channels  = cfg.MACROPROPS.MPROPS_COUNT,
                                  output_channels = cfg.MACROPROPS.MPROPS_COUNT,
                                  num_res_blocks  = cfg.MODEL.NUM_RES_BLOCKS,
                                  base_channels           = cfg.MODEL.BASE_CH,
                                  base_channels_multiples = cfg.MODEL.BASE_CH_MULT,
                                  apply_attention         = cfg.MODEL.APPLY_ATTENTION,
                                  dropout_rate            = cfg.MODEL.DROPOUT_RATE,
                                  time_multiple           = cfg.MODEL.TIME_EMB_MULT,
                                  condition               = cfg.MODEL.CONDITION)
    lr_str = "{:.0e}".format(cfg.TRAIN.SOLVER.LR)
    if re.search(r"[{}]", cfg.MODEL.MODEL_NAME):
        model_fullname = cfg.MODEL.SAVE_DIR+(cfg.MODEL.MODEL_NAME.format(cfg.TRAIN.EPOCHS, lr_str, cfg.DATASET.TRAIN_FILE_COUNT, cfg.DATASET.PAST_LEN, cfg.DATASET.FUTURE_LEN))
    else:
        model_fullname = cfg.MODEL.SAVE_DIR+cfg.MODEL.MODEL_NAME
    logging.info(f'model full name:{model_fullname}')

    denoiser.load_state_dict(torch.load(model_fullname, map_location=torch.device('cpu'))['model'])
    denoiser.to(device)

    # Instantiate the diffusion model 
    timesteps=cfg.DIFFUSION.TIMESTEPS
    diffusionmodel = DDPM(timesteps=cfg.DIFFUSION.TIMESTEPS)
    diffusionmodel.to(device)
    seq_frames = []
    taus = 1
    for batch in batched_test_data:
        past_test, future_test, stats = batch
        past_test, future_test = past_test.float(), future_test.float()
        past_test, future_test = past_test.to(device=device), future_test.to(device=device)
        #x_train, y_train, stats = batch
        random_past_idx = torch.randperm(past_test.shape[0])[:cfg.DIFFUSION.NSAMPLES4PLOTS]
        # Predict different sequences for the same past sequence
        if samePastSeq:
            fixed_past_idx = random_past_idx[0]
            random_past_idx.fill_(fixed_past_idx)

        random_past_samples = past_test[random_past_idx]
        random_future_samples = future_test[random_past_idx]
        if cfg.DIFFUSION.SAMPLER == "DDPM":
            x, xnoisy_over_time  = generate_ddpm(denoiser, random_past_samples, diffusionmodel, cfg, device, cfg.DIFFUSION.NSAMPLES4PLOTS) # AR review .cpu() call here
            if cfg.DIFFUSION.GUIDANCE == "sparsity" or cfg.DIFFUSION.GUIDANCE == "none":
                l1 = torch.mean(torch.abs(x[:,0,:,:,:])).cpu().detach().numpy()
                print('L1 norm {:.2f}'.format(l1))
        elif cfg.DIFFUSION.SAMPLER == "DDIM":
            taus = np.arange(0,timesteps,cfg.DIFFUSION.DDIM_DIVIDER)
            print(f'taus:{taus}')
            x, xnoisy_over_time = generate_ddim(denoiser, random_past_samples, taus, diffusionmodel, cfg, device, cfg.DIFFUSION.NSAMPLES4PLOTS) # AR review .cpu() call here
        else:
            print(f"{cfg.DIFFUSION.SAMPLER} sampler not supported")

        future_sample_pred = x
        for i in range(len(random_past_idx)):
            future_sample_pred_iv = future_sample_pred[i]
            future_sample_gt_iv = random_future_samples[i]
            past_sample_iv = random_past_samples[i]
            seq_pred = torch.cat([past_sample_iv, future_sample_pred_iv], dim=3)
            seq_gt = torch.cat([past_sample_iv, future_sample_gt_iv], dim=3)
            seq_frames.append(seq_pred)
            seq_frames.append(seq_gt)

        match = re.search(r'E\d+_LR\de-\d+_TFC\d+_PL\d+_FL\d', model_fullname)
        if plotType == "Static":
            plotStaticMacroprops(seq_frames, cfg, match, plotMprop, plotPast, velScale, velUncScale)
        elif plotType == "Dynamic":
            plotDynamicMacroprops(seq_frames, cfg, match, velScale, velUncScale, headwidth)

        plotDensityOverTime(seq_frames, cfg)
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to sample crowd macroprops from trained model.")
    parser.add_argument('--plot-mprop', type=str, default='Density', help='Macroprops to be plotted')
    parser.add_argument('--plot-past', type=str, default='Last2', help='Past macroprops to be plotted')
    parser.add_argument('--vel-scale', type=float, default=0.5, help='Scale to be applied to velocity mprops vectors')
    parser.add_argument('--headwidth', type=int, default=5, help='Headwidth to be applied to velocity mprops vectors')
    parser.add_argument('--vel-unc-scale', type=int, default=1, help='Scale to be applied to velocity uncertainty mprops vectors')
    parser.add_argument('--plot-type', type=str, default='Static', help='Macroprops plot type can be static (.svg) or dinamic (.gif)')
    parser.add_argument('--same-past-seq', type=bool, default=False, help='Use the same past sequence to predict different mprops from it.')
    parser.add_argument('--config-yml-file', type=str, default='config/ATC_ddpm_4test.yml', help='Configuration YML file for specific dataset.')
    parser.add_argument('--configList-yml-file', type=str, default='config/ATC_ddpm_DSlist4test.yml',help='Configuration YML macroprops list for specific dataset.')
    args = parser.parse_args()

    cfg = getYamlConfig(args.config_yml_file, args.configList_yml_file)
    filenames = cfg.DATA_LIST

    if cfg.DATASET.NAME in ["ATC", "ATC4TEST"]:
        filenames = [filename.replace(".csv", ".pkl") for filename in filenames]
    elif cfg.DATASET.NAME in ["HERMES-BO", "HERMES-CR-120", "HERMES-CR-120-OBS"]:
        filenames = [filename.replace(".txt", ".pkl") for filename in filenames]
    else:
        logging.info("Dataset not supported")

    filenames = [ os.path.join(cfg.PICKLE.PICKLE_DIR, filename) for filename in filenames if filename.endswith('.pkl')]
    generate_samples(cfg, filenames, plotType=args.plot_type, plotMprop=args.plot_mprop, plotPast=args.plot_past, velScale=args.vel_scale, velUncScale=args.vel_unc_scale, samePastSeq=args.same_past_seq, headwidth=args.headwidth)

# execution example:
# python3 generate_samples.py --plot-mprop="Density" --plot-past="Last2"
# python3 generate_samples.py --plot-mprop="Density&Vel" --plot-past="Last2" --vel-scale=0.25
# python3 generate_samples.py --plot-mprop="Uncertainty" --plot-past="Last2" --vel-unc-scale=3
# python3 generate_samples.py --plot-type="Dynamic" --vel-scale=0.25 --vel-unc-scale=3
# python3 generate_samples.py --plot-mprop="Density" --plot-past="Last2" --same-past-seq=True