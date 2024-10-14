import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from utils.motionFeatureExtractor import MotionFeatureExtractor, get_bhattacharyya_dist_coef

def my_psnr(y_gt, y_hat, data_range, eps):
    # Compute mean squared error
    err = np.mean((y_gt - y_hat) ** 2, dtype=np.float64)
    # Prevent overflow and division by zero
    err = max(err, 0.001)
    # Calculate PSNR
    data_range = float(data_range)
    psnr = 10 * np.log10((data_range ** 2) / err)  
    return psnr

def psnr_mprops_seq(gt_seq_list, pred_seq_list, mprops_factor, chunkRepdPastSeq, eps):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_psnr = np.zeros((nsamples, 3))
    mprops_max_psnr = np.zeros((nsamples//chunkRepdPastSeq, 3))

    for i in range(nsamples):
        one_pred_seq =  pred_seq_list[i].cpu().numpy()
        one_gt_seq =  gt_seq_list[i].cpu().numpy()

        mprops_factor = np.array(mprops_factor)
        one_pred_seq = one_pred_seq * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis]
        one_gt_seq = one_gt_seq * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis]
        # Calculate data ranges for each macroprop
        rho_range = int(one_gt_seq[0].max() - one_gt_seq[0].min())
        vx_range  = int(one_gt_seq[1].max() - one_gt_seq[1].min())
        vy_range  = int(one_gt_seq[2].max() - one_gt_seq[2].min())

        psnr_rho, psnr_vx, psnr_vy = 0, 0, 0
        for j in range(pred_len):
            psnr_rho += my_psnr(one_gt_seq[0, :, :, j], one_pred_seq[0, :, :, j], data_range=rho_range, eps=eps)
            psnr_vx  += my_psnr(one_gt_seq[1, :, :, j], one_pred_seq[1, :, :, j], data_range=vx_range, eps=eps)
            psnr_vy  += my_psnr(one_gt_seq[2, :, :, j], one_pred_seq[2, :, :, j], data_range=vy_range, eps=eps)

        # Average PSNR across frames, except for unc channel
        mprops_nsamples_psnr[i] = (psnr_rho/pred_len, psnr_vx/pred_len, psnr_vy/pred_len)

    # Compute the MAX PSNR by repeteaded seqs on each macroprops
    for i in range(0, nsamples, chunkRepdPastSeq):
        psnr_chunk = mprops_nsamples_psnr[i:i+chunkRepdPastSeq]
        max_rho = psnr_chunk[:,0].max()
        max_vx  = psnr_chunk[:,1].max()
        max_vy  = psnr_chunk[:,2].max()
        mprops_max_psnr[i // chunkRepdPastSeq] = (max_rho, max_vx, max_vy)

    return mprops_nsamples_psnr, mprops_max_psnr

def ssim_mprops_seq(gt_seq_list, pred_seq_list, mprops_factor, chunkRepdPastSeq):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_ssim = np.zeros((nsamples, 3))
    mprops_max_ssim = np.zeros((nsamples//chunkRepdPastSeq, 3))

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()

        mprops_factor = np.array(mprops_factor)
        one_pred_seq = one_pred_seq * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis]
        one_gt_seq = one_gt_seq * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis]
         # Calculate data ranges for each macroprop
        rho_range = int(one_gt_seq[0].max() - one_gt_seq[0].min())
        vx_range  = int(one_gt_seq[1].max() - one_gt_seq[1].min())
        vy_range  = int(one_gt_seq[2].max() - one_gt_seq[2].min())

        ssim_rho, ssim_vx, ssim_vy = 0, 0, 0
        for j in range(pred_len):
            ssim_rho += ssim(one_gt_seq[0, :, :, j], one_pred_seq[0, :, :, j], data_range=rho_range)
            ssim_vx  += ssim(one_gt_seq[1, :, :, j], one_pred_seq[1, :, :, j], data_range=vx_range)
            ssim_vy  += ssim(one_gt_seq[2, :, :, j], one_pred_seq[2, :, :, j], data_range=vy_range)

        # Average SSIM across frames, except for unc channel
        mprops_nsamples_ssim[i] = (ssim_rho/pred_len, ssim_vx/pred_len, ssim_vy/pred_len)

    # Compute the MAX SSIM by repeteaded seqs on each macroprops
    for i in range(0, nsamples, chunkRepdPastSeq):
        ssim_chunk = mprops_nsamples_ssim[i:i+chunkRepdPastSeq]
        max_rho = ssim_chunk[:, 0].max()
        max_vx  = ssim_chunk[:, 1].max()
        max_vy  = ssim_chunk[:, 2].max()
        mprops_max_ssim[i // chunkRepdPastSeq] = (max_rho, max_vx, max_vy)

    return mprops_nsamples_ssim, mprops_max_ssim

def _save_mag_rho_data(all_mag_rho_vol, nameToUse):
    file_name = f"metrics/all_mag_rho_{nameToUse}.csv"
    np.savetxt(file_name, all_mag_rho_vol, delimiter=",", comments="")

def motion_feature_by_mse(gt_seq_list, pred_seq_list, f, k, gamma, mag_rho_flag=False):
    mf_extractor_pred = MotionFeatureExtractor(pred_seq_list, f=f, k=k, gamma=gamma)
    mf_extractor_gt = MotionFeatureExtractor(gt_seq_list, f=f, k=k, gamma=gamma)

    mf_2D_pred = mf_extractor_pred.motion_feature_2D_hist()
    mf_2D_gt = mf_extractor_gt.motion_feature_2D_hist()
    mf_1D_pred, all_mag_rho_vol_pred = mf_extractor_pred.motion_feature_1D_hist()
    mf_1D_gt, all_mag_rho_vol_gt = mf_extractor_gt.motion_feature_1D_hist()
    if mag_rho_flag:
        _save_mag_rho_data(all_mag_rho_vol_pred, "PRED")
        _save_mag_rho_data(all_mag_rho_vol_gt, "GT")

    motion_feat_mse = np.zeros((len(pred_seq_list), 2))

    for sample in range(len(pred_seq_list)):
        mse_2D = mean_squared_error(mf_2D_gt[sample], mf_2D_pred[sample])
        mse_1D = mean_squared_error(mf_1D_gt[sample], mf_1D_pred[sample])
        motion_feat_mse[sample] = (mse_2D, mse_1D)

    return motion_feat_mse

def motion_feature_by_bhattacharyya(gt_seq_list, pred_seq_list, f, k, gamma):
    num_angle_bins = 8
    num_magnitude_bins=9

    mf_extractor_pred = MotionFeatureExtractor(pred_seq_list, f=f, k=k, gamma=gamma, num_magnitude_bins=num_magnitude_bins, num_angle_bins=num_angle_bins)
    mf_extractor_gt = MotionFeatureExtractor(gt_seq_list, f=f, k=k, gamma=gamma, num_magnitude_bins=num_magnitude_bins, num_angle_bins=num_angle_bins)

    mf_2D_pred = mf_extractor_pred.motion_feature_2D_hist()
    mf_2D_gt = mf_extractor_gt.motion_feature_2D_hist()
    mf_1D_pred, _ = mf_extractor_pred.motion_feature_1D_hist()
    mf_1D_gt, _ = mf_extractor_gt.motion_feature_1D_hist()

    motion_feat_bhatt_dist = np.zeros((len(pred_seq_list), 2))
    motion_feat_bhatt_coef = np.zeros((len(pred_seq_list), 2))
    for sample in range(len(pred_seq_list)):
        bhat_dist_2D, bhat_coef_2D = get_bhattacharyya_dist_coef(mf_2D_gt[sample], mf_2D_pred[sample])
        bhat_dist_1D, bhat_coef_1D  = get_bhattacharyya_dist_coef(mf_1D_gt[sample], mf_1D_pred[sample])
        motion_feat_bhatt_dist[sample] = (bhat_dist_2D, bhat_dist_1D)
        motion_feat_bhatt_coef[sample] = (bhat_coef_2D, bhat_coef_1D)

    return motion_feat_bhatt_dist, motion_feat_bhatt_coef