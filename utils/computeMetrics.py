import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from utils.motionFeatureExtractor import MotionFeatureExtractor, get_bhattacharyya_dist_coef

def my_psnr(y_gt, y_hat, data_range, eps):
    # Compute mean squared error
    err = np.mean((y_gt - y_hat) ** 2, dtype=np.float64)
    # Prevent overflow and division by zero
    err = max(err, eps)
    # Calculate PSNR
    tmp_num = 20 * np.log10(data_range)
    tmp_den = 10 * np.log10(err)
    psnr = tmp_num - tmp_den
    return psnr

def get_mprops_ranges(gt_seq_list, mprops_factor):
    nsamples = len(gt_seq_list)
     # Initialize arrays to store max and min values for each sample and each property
    max_vals = np.zeros((nsamples, 4))
    min_vals = np.zeros((nsamples, 4))

    for i, one_gt_seq in enumerate(gt_seq_list):
        # Convert the tensor to a numpy array and scale it
        one_gt_seq = one_gt_seq.cpu().numpy() * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis]

        # Calculate max and min values for rho, vx, and vy, storing them in columns
        max_vals[i, 0], min_vals[i, 0] = one_gt_seq[0].max(), one_gt_seq[0].min()  # rho
        max_vals[i, 1], min_vals[i, 1] = one_gt_seq[1].max(), one_gt_seq[1].min()  # vx
        max_vals[i, 2], min_vals[i, 2] = one_gt_seq[2].max(), one_gt_seq[2].min()  # vy
        max_vals[i, 3], min_vals[i, 3] = one_gt_seq[3].max(), one_gt_seq[3].min()  # unc

    # Compute the overall max and min values for each macro-property across all samples
    global_max_rho, global_max_vx, global_max_vy, global_max_unc = max_vals.max(axis=0)
    global_min_rho, global_min_vx, global_min_vy, global_min_unc = min_vals.min(axis=0)

    # Compute the range for each macro-property
    rho_range = float(global_max_rho - global_min_rho)
    vx_range  = float(global_max_vx - global_min_vx)
    vy_range  = float(global_max_vy - global_min_vy)
    unc_range = float(global_max_unc - global_min_unc)

    return rho_range, vx_range, vy_range, unc_range

def psnr_mprops_seq(gt_seq_list, pred_seq_list, mprops_factor, chunkRepdPastSeq, eps):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_psnr = np.zeros((nsamples, 4))
    mprops_max_psnr = np.zeros((nsamples//chunkRepdPastSeq, 4))
    mprops_factor = np.array(mprops_factor)

    rho_range, vx_range, vy_range, unc_range = get_mprops_ranges(gt_seq_list, mprops_factor)
    print(f'Range of macroprops \n rho:{rho_range:.4f}, vx:{vx_range:.4f}, vy:{vy_range:.4f} and unc:{unc_range:.4f}')

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()

        one_pred_seq = one_pred_seq * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        one_gt_seq = one_gt_seq * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

        psnr_rho, psnr_vx, psnr_vy, psnr_unc = 0, 0, 0, 0
        for j in range(pred_len):
            psnr_rho += my_psnr(one_gt_seq[0, :, :, j], one_pred_seq[0, :, :, j], data_range=rho_range, eps=eps)
            psnr_vx  += my_psnr(one_gt_seq[1, :, :, j], one_pred_seq[1, :, :, j], data_range=vx_range, eps=eps)
            psnr_vy  += my_psnr(one_gt_seq[2, :, :, j], one_pred_seq[2, :, :, j], data_range=vy_range, eps=eps)
            psnr_unc += my_psnr(one_gt_seq[3, :, :, j], one_pred_seq[3, :, :, j], data_range=unc_range, eps=eps)

        # Average PSNR across frames, except for unc channel
        mprops_nsamples_psnr[i] = (psnr_rho/pred_len, psnr_vx/pred_len, psnr_vy/pred_len, psnr_unc/pred_len)

    # Compute the MAX PSNR by repeteaded seqs on each macroprops
    for i in range(0, nsamples, chunkRepdPastSeq):
        psnr_chunk = mprops_nsamples_psnr[i:i+chunkRepdPastSeq]
        max_rho = psnr_chunk[:,0].max()
        max_vx  = psnr_chunk[:,1].max()
        max_vy  = psnr_chunk[:,2].max()
        max_unc = psnr_chunk[:,3].max()
        mprops_max_psnr[i // chunkRepdPastSeq] = (max_rho, max_vx, max_vy, max_unc)

    return mprops_nsamples_psnr, mprops_max_psnr

def ssim_mprops_seq(gt_seq_list, pred_seq_list, mprops_factor, chunkRepdPastSeq):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_ssim = np.zeros((nsamples, 4))
    mprops_max_ssim = np.zeros((nsamples//chunkRepdPastSeq, 4))

    rho_range, vx_range, vy_range, unc_range = get_mprops_ranges(gt_seq_list, mprops_factor)

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()

        mprops_factor = np.array(mprops_factor)
        one_pred_seq = one_pred_seq * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        one_gt_seq = one_gt_seq * mprops_factor[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

        ssim_rho, ssim_vx, ssim_vy, ssim_unc = 0, 0, 0, 0
        for j in range(pred_len):
            ssim_rho += ssim(one_gt_seq[0, :, :, j], one_pred_seq[0, :, :, j], data_range=rho_range)
            ssim_vx  += ssim(one_gt_seq[1, :, :, j], one_pred_seq[1, :, :, j], data_range=vx_range)
            ssim_vy  += ssim(one_gt_seq[2, :, :, j], one_pred_seq[2, :, :, j], data_range=vy_range)
            ssim_unc += ssim(one_gt_seq[3, :, :, j], one_pred_seq[3, :, :, j], data_range=unc_range)

        # Average SSIM across frames, except for unc channel
        mprops_nsamples_ssim[i] = (ssim_rho/pred_len, ssim_vx/pred_len, ssim_vy/pred_len, ssim_unc/pred_len)

    # Compute the MAX SSIM by repeteaded seqs on each macroprops
    for i in range(0, nsamples, chunkRepdPastSeq):
        ssim_chunk = mprops_nsamples_ssim[i:i+chunkRepdPastSeq]
        max_rho = ssim_chunk[:, 0].max()
        max_vx  = ssim_chunk[:, 1].max()
        max_vy  = ssim_chunk[:, 2].max()
        max_unc = ssim_chunk[:, 3].max()
        mprops_max_ssim[i // chunkRepdPastSeq] = (max_rho, max_vx, max_vy, max_unc)

    return mprops_nsamples_ssim, mprops_max_ssim

def _save_mag_rho_data(all_mag_rho_vol, nameToUse):
    file_name = f"metrics/all_mag_rho_{nameToUse}.csv"
    np.savetxt(file_name, all_mag_rho_vol, delimiter=",", comments="")

def motion_feature_metrics(gt_seq_list, pred_seq_list, f, k, gamma, mse_metric=False, bhatt_metrics=False):
    mf_extractor_pred = MotionFeatureExtractor(pred_seq_list, f=f, k=k, gamma=gamma)
    mf_extractor_gt = MotionFeatureExtractor(gt_seq_list, f=f, k=k, gamma=gamma)

    mf_2D_pred = mf_extractor_pred.motion_feature_2D_hist()
    mf_2D_gt = mf_extractor_gt.motion_feature_2D_hist()
    mf_1D_pred, all_mag_rho_vol_pred = mf_extractor_pred.motion_feature_1D_hist()
    mf_1D_gt, all_mag_rho_vol_gt = mf_extractor_gt.motion_feature_1D_hist()
    mfeat_mse, mfeat_bhatt_dist, mfeat_bhatt_coef = None, None, None

    if mse_metric:
        mfeat_mse = motion_feature_by_mse(mf_2D_pred, mf_2D_gt, mf_1D_pred, mf_1D_gt)
    if bhatt_metrics:
        mfeat_bhatt_dist, mfeat_bhatt_coef = motion_feature_by_bhattacharyya(mf_2D_pred, mf_2D_gt, mf_1D_pred, mf_1D_gt)

    return mfeat_mse, mfeat_bhatt_dist, mfeat_bhatt_coef

def motion_feature_by_mse(mf_2D_pred, mf_2D_gt, mf_1D_pred, mf_1D_gt):
    motion_feat_mse = np.zeros((len(mf_2D_pred), 2))
    for sample in range(len(mf_2D_pred)):
        mse_2D = mean_squared_error(mf_2D_gt[sample], mf_2D_pred[sample])
        mse_1D = mean_squared_error(mf_1D_gt[sample], mf_1D_pred[sample])
        motion_feat_mse[sample] = (mse_2D, mse_1D)

    return motion_feat_mse

def motion_feature_by_bhattacharyya(mf_2D_pred, mf_2D_gt, mf_1D_pred, mf_1D_gt):
    motion_feat_bhatt_dist = np.zeros((len(mf_2D_pred), 2))
    motion_feat_bhatt_coef = np.zeros((len(mf_2D_pred), 2))
    for sample in range(len(mf_2D_pred)):
        bhat_dist_2D, bhat_coef_2D = get_bhattacharyya_dist_coef(mf_2D_gt[sample], mf_2D_pred[sample])
        bhat_dist_1D, bhat_coef_1D  = get_bhattacharyya_dist_coef(mf_1D_gt[sample], mf_1D_pred[sample])
        motion_feat_bhatt_dist[sample] = (bhat_dist_2D, bhat_dist_1D)
        motion_feat_bhatt_coef[sample] = (bhat_coef_2D, bhat_coef_1D)

    return motion_feat_bhatt_dist, motion_feat_bhatt_coef