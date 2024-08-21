import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def psnr_mprops_seq(gt_seq_list, pred_seq_list):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_psnr = np.zeros((nsamples, 4))

    for i in range(nsamples):
        one_pred_seq =  pred_seq_list[i].cpu().numpy()
        one_gt_seq =  gt_seq_list[i].cpu().numpy()
        psnr_rho, psnr_vx, psnr_vy, psnr_unc = 0, 0, 0, 0
        for j in range(pred_len):
            psnr_rho += psnr(one_gt_seq[0, :, :, j], one_pred_seq[0, :, :, j], data_range=one_gt_seq[0, :, :, j].max() - one_gt_seq[0, :, :, j].min())
            psnr_vx  += psnr(one_gt_seq[1, :, :, j], one_pred_seq[1, :, :, j], data_range=one_gt_seq[1, :, :, j].max() - one_gt_seq[1, :, :, j].min())
            psnr_vy  += psnr(one_gt_seq[2, :, :, j], one_pred_seq[2, :, :, j], data_range=one_gt_seq[2, :, :, j].max() - one_gt_seq[2, :, :, j].min())
            psnr_unc += psnr(one_gt_seq[3, :, :, j], one_pred_seq[3, :, :, j], data_range=one_gt_seq[3, :, :, j].max() - one_gt_seq[3, :, :, j].min())
        # Average PSNR across frames
        mprops_nsamples_psnr[i] = (psnr_rho/pred_len, psnr_vx/pred_len, psnr_vy/pred_len, psnr_unc/pred_len)

    return mprops_nsamples_psnr

def ssim_mprops_seq(gt_seq_list, pred_seq_list):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_ssim = np.zeros((nsamples, 4))

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()
        ssim_rho, ssim_vx, ssim_vy, ssim_unc = 0, 0, 0, 0

        for j in range(pred_len):
            ssim_rho += ssim(one_gt_seq[0, :, :, j], one_pred_seq[0, :, :, j], data_range=one_gt_seq[0, :, :, j].max() - one_gt_seq[0, :, :, j].min())
            ssim_vx  += ssim(one_gt_seq[1, :, :, j], one_pred_seq[1, :, :, j], data_range=one_gt_seq[1, :, :, j].max() - one_gt_seq[1, :, :, j].min())
            ssim_vy  += ssim(one_gt_seq[2, :, :, j], one_pred_seq[2, :, :, j], data_range=one_gt_seq[2, :, :, j].max() - one_gt_seq[2, :, :, j].min())
            ssim_unc += ssim(one_gt_seq[3, :, :, j], one_pred_seq[3, :, :, j], data_range=one_gt_seq[3, :, :, j].max() - one_gt_seq[3, :, :, j].min())

        # Average SSIM across frames
        mprops_nsamples_ssim[i] = (ssim_rho/pred_len, ssim_vx/pred_len, ssim_vy/pred_len, ssim_unc/pred_len)
    return mprops_nsamples_ssim

def lpips_mprops_seq(gt_seq_list, pred_seq_list):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_lpips = np.zeros((nsamples, 4))
    return mprops_nsamples_lpips