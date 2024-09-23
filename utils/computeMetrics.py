import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from utils.motionFeatureExtractor import MotionFeatureExtractor

def my_psnr(y_gt, y_hat, data_range, eps):
    err = np.mean((y_gt - y_hat) ** 2, dtype=np.float64)
    data_range = float(data_range)  # prevent overflow for small integer types
    if err < eps:
        psnr = 10 * np.log10((data_range**2) / eps)
    else:
        psnr = 10 * np.log10((data_range**2) / err)
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

def lpips_mprops_seq(gt_seq_list, pred_seq_list):
    nsamples = len(pred_seq_list)
    _, _, _, pred_len = pred_seq_list[0].shape
    mprops_nsamples_lpips = np.zeros((nsamples, 4))
    return mprops_nsamples_lpips

def motion_feature_metric(gt_seq_list, pred_seq_list, f, k):
    mf_extractor_pred = MotionFeatureExtractor(pred_seq_list, f=f, k=k)
    mf_extractor_gt = MotionFeatureExtractor(gt_seq_list, f=f, k=k)

    mf_2D_pred = mf_extractor_pred.motion_feature_2D_hist()
    mf_1D_pred = mf_extractor_pred.motion_feature_1D_hist()
    mf_2D_gt = mf_extractor_gt.motion_feature_2D_hist()
    mf_1D_gt = mf_extractor_gt.motion_feature_1D_hist()

    print("2D Motion Feature Shape for predicted seqs:", mf_2D_pred.shape)
    print("1D Motion Feature Shape for predicted seqs:", mf_1D_pred.shape)

    motion_feat_mse = np.zeros((len(pred_seq_list), 2))

    for sample in range(len(pred_seq_list)):
        mse_2D = mean_squared_error(mf_2D_gt[sample], mf_2D_pred[sample])
        mse_1D = mean_squared_error(mf_1D_gt[sample], mf_1D_pred[sample])
        motion_feat_mse[sample] = (mse_2D, mse_1D)

    return motion_feat_mse