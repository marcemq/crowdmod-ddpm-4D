import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

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

        mprops_nsamples_psnr[i] = (psnr_rho/pred_len, psnr_vx/pred_len, psnr_vy/pred_len, psnr_unc/pred_len)

    return mprops_nsamples_psnr