import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from utils.motionFeatureExtractor import MotionFeatureExtractor, get_bhattacharyya_dist_coef
from models.guidance import compute_energy

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

def get_mprops_ranges(gt_seq_list, mprops_factor, mprops_count):
    nsamples = len(gt_seq_list)
     # Initialize arrays to store max and min values for each sample and each property
    max_vals = np.zeros((nsamples, mprops_count))
    min_vals = np.zeros((nsamples, mprops_count))

    for i, one_gt_seq in enumerate(gt_seq_list):
        # Convert the tensor to a numpy array and scale it
        one_gt_seq = one_gt_seq.cpu().numpy() * mprops_factor

        # Calculate max and min values for rho, vx, and vy, storing them in columns
        max_vals[i, 0], min_vals[i, 0] = one_gt_seq[0].max(), one_gt_seq[0].min()  # rho
        max_vals[i, 1], min_vals[i, 1] = one_gt_seq[1].max(), one_gt_seq[1].min()  # vx
        max_vals[i, 2], min_vals[i, 2] = one_gt_seq[2].max(), one_gt_seq[2].min()  # vy

    # Compute the overall max and min values for each macro-property across all samples
    global_max_rho, global_max_vx, global_max_vy= max_vals.max(axis=0)
    global_min_rho, global_min_vx, global_min_vy= min_vals.min(axis=0)

    # Compute the range for each macro-property
    rho_range = float(global_max_rho - global_min_rho)
    vx_range  = float(global_max_vx - global_min_vx)
    vy_range  = float(global_max_vy - global_min_vy)

    return rho_range, vx_range, vy_range

def psnr_mprops_seq(gt_seq_list, pred_seq_list, mprops_factor, chunkRepdPastSeq, eps, mprops_count):
    mprops_factor = np.array(mprops_factor)[:mprops_count, np.newaxis, np.newaxis, np.newaxis]
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_psnr = np.zeros((nsamples, mprops_count))
    mprops_max_psnr = np.zeros((nsamples//chunkRepdPastSeq, mprops_count))
    mprops_factor = np.array(mprops_factor)

    rho_range, vx_range, vy_range = get_mprops_ranges(gt_seq_list, mprops_factor, mprops_count)
    print(f'Range of macroprops \n rho:{rho_range:.4f}, vx:{vx_range:.4f} and vy:{vy_range:.4f}')

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()

        one_pred_seq = one_pred_seq * mprops_factor
        one_gt_seq = one_gt_seq * mprops_factor

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

def ssim_mprops_seq(gt_seq_list, pred_seq_list, mprops_factor, chunkRepdPastSeq, mprops_count):
    mprops_factor = np.array(mprops_factor)[:mprops_count, np.newaxis, np.newaxis, np.newaxis]
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_ssim = np.zeros((nsamples, mprops_count))
    mprops_max_ssim = np.zeros((nsamples//chunkRepdPastSeq, mprops_count))

    rho_range, vx_range, vy_range = get_mprops_ranges(gt_seq_list, mprops_factor, mprops_count)

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()

        mprops_factor = np.array(mprops_factor)
        one_pred_seq = one_pred_seq * mprops_factor
        one_gt_seq = one_gt_seq * mprops_factor

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

def energy_mprops_seq(gt_seq_list, pred_seq_list, mprops_factor, chunkRepdPastSeq, mprops_count):
    """
    Not sure if for energy we need to apply the factor to use an scaled version of sequences
    """
    mprops_factor = np.array(mprops_factor)[:mprops_count, np.newaxis, np.newaxis, np.newaxis]
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_energy = np.zeros((nsamples, 2))
    mprops_min_energy = np.zeros((nsamples//chunkRepdPastSeq, 2))
    mprops_factor = np.array(mprops_factor)

    pred_seq_tensor = torch.stack(pred_seq_list).cpu()
    gt_seq_tensor = torch.stack(gt_seq_list).cpu()

    pred_seq_tensor = pred_seq_tensor * mprops_factor[np.newaxis, ...]
    gt_seq_tensor = gt_seq_tensor * mprops_factor[np.newaxis, ...]

    pred_seq_energy = compute_energy(pred_seq_tensor, delta_t=1, delta_l=1)
    gt_seq_energy = compute_energy(gt_seq_tensor, delta_t=1, delta_l=1)

    mprops_nsamples_energy[:, 0] = gt_seq_energy
    mprops_nsamples_energy[:, 1] = pred_seq_energy

    # Compute the MIN energy by repeteaded seqs
    for i in range(0, nsamples, chunkRepdPastSeq):
        energy_chunk = mprops_nsamples_energy[i:i+chunkRepdPastSeq]
        min_gt_energy = energy_chunk[:, 0].min()
        min_pred_energy = energy_chunk[:, 1].min()
        mprops_min_energy[i // chunkRepdPastSeq] = (min_gt_energy, min_pred_energy)

    return mprops_nsamples_energy, mprops_min_energy

def re_density_mprops_seq(gt_seq_list, pred_seq_list, chunkRepdPastSeq, eps):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_re_density = np.zeros((nsamples, pred_len))
    mprops_min_re_density = np.zeros((nsamples//chunkRepdPastSeq, pred_len))

    for i in range(nsamples):
        one_pred_seq = pred_seq_list[i].cpu().numpy()
        one_gt_seq = gt_seq_list[i].cpu().numpy()

        density_pred = one_pred_seq[0]  # shape: (R, C, L)
        density_gt   = one_gt_seq[0]    # shape: (R, C, L)

        pred_total_density = density_pred.sum(axis=(0, 1))
        gt_total_density = density_gt.sum(axis=(0, 1))

        rel_error = np.abs(pred_total_density - gt_total_density) / (gt_total_density + eps)
        mprops_nsamples_re_density[i] = rel_error

    # Compute the MAX DENSITY by repeteaded seqs on each macroprops
    for i in range(0, nsamples, chunkRepdPastSeq):
        density_chunk = mprops_nsamples_re_density[i:i+chunkRepdPastSeq]
        min_rel_errors = density_chunk.min(axis=0)
        mprops_min_re_density[i // chunkRepdPastSeq] = min_rel_errors

    return mprops_nsamples_re_density, mprops_min_re_density

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