import numpy as np
from utils.plot.plot_metrics import plot_motion_feat_hist2D, plot_motion_feat_hist1D
from sklearn.preprocessing import MinMaxScaler

class MotionFeatureExtractor:
    def __init__(self, seq_list, f, k, gamma=0.5, num_magnitude_bins=18, num_angle_bins=16, output_dir=None):
        self.f = f
        self.k = k
        self.gamma = gamma
        self.nsamples = len(seq_list)
        self.seq_list = seq_list
        self.output_dir = output_dir

        # r,c: spatial dimensions, F: temporal dimension
        self._, self.r, self.c, self.F = seq_list[0].shape
        self.N = self.r * self.c
        self.num_magnitude_bins = num_magnitude_bins
        self.num_angle_bins = num_angle_bins
        self.scaler = MinMaxScaler(feature_range=(0, 255))
        self.mag_rho, self.angle_phi = self.compute_norm_angle_4samples()
        self.mag_rho_transf =  self.mag_rho_transform()

    def get_vel_vector_field(self, one_seq):
        v_x = one_seq[1, :, :, :]  # Shape (r, c, F)
        v_y = one_seq[2, :, :, :]  # Shape (r, c, F)
        # Reshape v_x and v_y to (F, N) where N = r * c
        v_x_flat = v_x.reshape(self.N, self.F).T  # Shape (F, N)
        v_y_flat = v_y.reshape(self.N, self.F).T  # Shape (F, N)
        # Stack v_x and v_y along the last axis to create U of shape (F, N, 2)
        U = np.stack((v_x_flat, v_y_flat), axis=-1)  # Shape (F, N, 2)
        return U

    def compute_norm_angle_4samples(self):
        """
        Computes the magnitude and angle for each key point in the vector field U.
        """
        mag_rho = np.zeros((self.nsamples, self.F, self.N))
        angle_phi = np.zeros((self.nsamples, self.F, self.N))

        for sample in range(self.nsamples):
            one_pred_seq = self.seq_list[sample].cpu().numpy()
            U = self.get_vel_vector_field(one_pred_seq)
            mag_rho[sample]   = np.sqrt(U[..., 0]**2 + U[..., 1]**2)  # Shape (F, N)
            #AR if mag_rho[sample] small then angle_phi[sample]=0
            angle_phi[sample] = np.arctan2(U[..., 1], U[..., 0])
        return mag_rho, angle_phi

    def mag_rho_transform(self):
        mag_rho_transf = np.zeros((self.nsamples, self.F, self.N))
        total_clipped = 0
        for sample in range(self.nsamples):
            mag_rho_sample = self.mag_rho[sample]
            mag_rho_normalized = self.scaler.fit_transform(mag_rho_sample).reshape(self.F, self.N)
            # Range here is [0,8]
            mag_rho_log = np.log2(mag_rho_normalized + 1)
            mag_rho_transf[sample] = mag_rho_log

        return mag_rho_transf

    def motion_feature_2D_hist(self, num_plot_hist2D=10, plot_prob=0.05, active_bins_threshold=5):
        all_motion_feature_vectors = []
        plotted = 0
        for sample in range(self.nsamples):
            motion_feature_vector = []
            # Reshape each frame's data back into a (r, c) grid
            mag_rho_reshaped = self.mag_rho_transf[sample].reshape(self.F, self.r, self.c)
            angle_phi_reshaped = self.angle_phi[sample].reshape(self.F, self.r, self.c)
            for i in range(0, self.F, self.f):  # Temporal volumes of size f
                for row in range(0, self.r, self.k):  # Spatial rows (k x k blocks)
                    for col in range(0, self.c, self.k):  # Spatial columns (k x k blocks)
                        # Extract a sub-volume of size (f, k, k)
                        mag_volume = mag_rho_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        angle_volume = angle_phi_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        # Compute 2D histogram (quantized magnitude vs angle)
                        hist_2D, mag_edges, angle_edges = np.histogram2d(mag_volume, angle_volume, bins=[self.num_magnitude_bins, self.num_angle_bins], range=[[0, 8.0], [-np.pi, np.pi]])

                        active_bins = np.sum(hist_2D >= 2)
                        if plotted < num_plot_hist2D and np.random.rand() < plot_prob and active_bins >= active_bins_threshold:
                            plot_motion_feat_hist2D(hist_2D, mag_edges, angle_edges, sample, i, row, col, plotted, self.output_dir, self.num_angle_bins)
                            plotted += 1
                        # Flatten and add to the motion feature vector
                        hist_2D = hist_2D.flatten()
                        motion_feature_vector.append(hist_2D)
            # Concatenate histograms from all volumes into a single vector
            motion_feature_vector = np.concatenate(motion_feature_vector)
            motion_feature_vector = motion_feature_vector / (motion_feature_vector.sum() + 1)
            all_motion_feature_vectors.append(motion_feature_vector)
        # Return the motion feature vectors for all sequences
        return np.array(all_motion_feature_vectors)

    def motion_feature_1D_hist(self, num_plot_hist1D=10, plot_prob=0.05, active_bins_threshold=5):
        all_motion_feature_vectors = []
        all_mag_rho_volumnes = []
        plotted = 0

        for sample in range(self.nsamples):
            motion_feature_vector = []
            # Reshape each frame's data back into a (r, c) grid
            mag_rho_reshaped = self.mag_rho_transf[sample].reshape(self.F, self.r, self.c)
            angle_phi_reshaped = self.angle_phi[sample].reshape(self.F, self.r, self.c)
            for i in range(0, self.F, self.f):  # Temporal volumes of size f
                for row in range(0, self.r, self.k):  # Spatial rows (k x k blocks)
                    for col in range(0, self.c, self.k):  # Spatial columns (k x k blocks)
                        # Extract a sub-volume of size (f, k, k)
                        mag_volume = mag_rho_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        angle_volume = angle_phi_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        all_mag_rho_volumnes.append(mag_volume)
                        # Quantize only angles
                        angle_bins = np.digitize(angle_volume, np.linspace(-np.pi, np.pi, self.num_angle_bins+1)) - 1
                        # Initialize a 1D histogram (8 bins for angles)
                        hist_1D = np.zeros(self.num_angle_bins)
                        # Sum magnitudes into the corresponding angle bins
                        for bin_idx in range(self.num_angle_bins):
                            hist_1D[bin_idx] = np.sum(np.power(mag_volume[angle_bins == bin_idx], self.gamma))

                        active_bins = np.sum(hist_1D >= 2)
                        if plotted < num_plot_hist1D and np.random.rand() < plot_prob and active_bins >= active_bins_threshold :
                            plot_motion_feat_hist1D(hist_1D, sample, i, row, col, plotted, self.output_dir, self.num_angle_bins)
                            plotted += 1
                        # Append this histogram to the motion feature vector avoing division by cero
                        motion_feature_vector.append(hist_1D)
            # Concatenate histograms from all volumes into a single vector
            motion_feature_vector = np.concatenate(motion_feature_vector)
            motion_feature_vector = motion_feature_vector / (motion_feature_vector.sum() + 1)
            all_motion_feature_vectors.append(motion_feature_vector)
        # Return the motion feature vectors for all sequences
        return np.array(all_motion_feature_vectors), np.array(all_mag_rho_volumnes)

def get_mag_angle_seq(motion_feature, sample):
    mag_rho_rs = motion_feature.mag_rho_transf[sample].reshape(motion_feature.F, motion_feature.r, motion_feature.c)
    angle_phi_rs = motion_feature.angle_phi[sample].reshape(motion_feature.F, motion_feature.r, motion_feature.c)
    return mag_rho_rs, angle_phi_rs

def get_mag_angle_volume(mag_rho_rs, angle_phi_rs, i, row, col, f, k):
    mag_vol = mag_rho_rs[i:i+f, row:row+k, col:col+k].flatten()
    angle_vol = angle_phi_rs[i:i+f, row:row+k, col:col+k].flatten()
    return mag_vol, angle_vol

def set_zero_angle_to_smallMag(hist_2D):
    total_first_mag = np.sum(hist_2D[0, :])
    # Reset first magnitude bin
    hist_2D[0, :] = 0
    # Put everything into angle=0 position
    hist_2D[0, 0] = total_first_mag
    return hist_2D

def get_motion_feature_2D_hist(mf_pred, mf_gt, num_plot_hist2D=10, plot_prob=0.05, active_bins_threshold=5):
    all_mf_pred, all_mf_gt = [], []
    plotted = 0

    for sample in range(mf_pred.nsamples):
        mf_vec_pred, mf_vec_gt = [], []
        # Reshape each frame's data back into a (r, c) grid
        mag_rho_rs_pred, angle_phi_rs_pred = get_mag_angle_seq(mf_pred, sample)
        mag_rho_rs_gt, angle_phi_rs_gt = get_mag_angle_seq(mf_gt, sample)

        for i in range(0, mf_pred.F, mf_pred.f):  # Temporal volumes of size f
            for row in range(0, mf_pred.r, mf_pred.k):  # Spatial rows (k x k blocks)
                for col in range(0, mf_pred.c, mf_pred.k):  # Spatial columns (k x k blocks)
                    # Extract a sub-volume of size (f, k, k) for PRED ang GT
                    mag_vol_pred, angle_vol_pred = get_mag_angle_volume(mag_rho_rs_pred, angle_phi_rs_pred, i, row, col, mf_pred.f, mf_pred.k)
                    mag_vol_gt, angle_vol_gt = get_mag_angle_volume(mag_rho_rs_gt, angle_phi_rs_gt, i, row, col, mf_gt.f, mf_gt.k)
                    
                    # Compute 2D histogram (quantized magnitude vs angle)
                    hist_2D_pred, mag_edges_pred, angle_edges_pred = np.histogram2d(mag_vol_pred, angle_vol_pred, bins=[mf_pred.num_magnitude_bins, mf_pred.num_angle_bins], range=[[0, 8.0], [-np.pi, np.pi]])
                    hist_2D_gt, mag_edges_gt, angle_edges_gt = np.histogram2d(mag_vol_gt, angle_vol_gt, bins=[mf_gt.num_magnitude_bins, mf_gt.num_angle_bins], range=[[0, 8.0], [-np.pi, np.pi]])

                    # Set zero angle to small magnitutes
                    hist_2D_pred = set_zero_angle_to_smallMag(hist_2D_pred)
                    hist_2D_gt = set_zero_angle_to_smallMag(hist_2D_gt)

                    active_bins = np.sum(hist_2D_gt >= 2)
                    if plotted < num_plot_hist2D and np.random.rand() < plot_prob and active_bins >= active_bins_threshold:
                        plot_motion_feat_hist2D(hist_2D_pred, mag_edges_pred, angle_edges_pred, sample, i, row, col, plotted, mf_pred.output_dir, mf_pred.num_angle_bins, "pred")
                        plot_motion_feat_hist2D(hist_2D_gt, mag_edges_gt, angle_edges_gt, sample, i, row, col, plotted, mf_gt.output_dir, mf_gt.num_angle_bins, "gt")
                        plotted += 1
                    # Flatten and add to the motion feature vector
                    hist_2D_pred = hist_2D_pred.flatten()
                    hist_2D_gt = hist_2D_gt.flatten()
                    mf_vec_pred.append(hist_2D_pred)
                    mf_vec_gt.append(hist_2D_gt)
        # Concatenate histograms from all volumes into a single vector
        mf_vec_pred = np.concatenate(mf_vec_pred)
        mf_vec_pred = mf_vec_pred / (mf_vec_pred.sum() + 1)

        mf_vec_gt = np.concatenate(mf_vec_gt)
        mf_vec_gt = mf_vec_gt / (mf_vec_gt.sum() + 1)

        all_mf_pred.append(mf_vec_pred)
        all_mf_gt.append(mf_vec_gt)
    # Return the motion feature vectors for all pred and GT sequences
    return np.array(all_mf_pred), np.array(all_mf_gt)

def get_motion_feature_1D_hist(mf_pred, mf_gt, num_plot_hist1D=10, plot_prob=0.05, active_bins_threshold=5):
    all_mf_pred, all_mf_gt = [], []
    plotted = 0

    for sample in range(mf_pred.nsamples):
        mf_vec_pred, mf_vec_gt = [], []
        # Reshape each frame's data back into a (r, c) grid
        mag_rho_rs_pred, angle_phi_rs_pred = get_mag_angle_seq(mf_pred, sample)
        mag_rho_rs_gt, angle_phi_rs_gt = get_mag_angle_seq(mf_gt, sample)

        for i in range(0, mf_pred.F, mf_pred.f):  # Temporal volumes of size f
            for row in range(0, mf_pred.r, mf_pred.k):  # Spatial rows (k x k blocks)
                for col in range(0, mf_pred.c, mf_pred.k):  # Spatial columns (k x k blocks)
                    # Extract a sub-volume of size (f, k, k) for PRED ang GT
                    mag_vol_pred, angle_vol_pred = get_mag_angle_volume(mag_rho_rs_pred, angle_phi_rs_pred, i, row, col, mf_pred.f, mf_pred.k)
                    mag_vol_gt, angle_vol_gt = get_mag_angle_volume(mag_rho_rs_gt, angle_phi_rs_gt, i, row, col, mf_gt.f, mf_gt.k)
                    # Quantize only angles
                    angle_bins_pred = np.digitize(angle_vol_pred, np.linspace(-np.pi, np.pi, mf_pred.num_angle_bins+1)) - 1
                    angle_bins_gt = np.digitize(angle_vol_gt, np.linspace(-np.pi, np.pi, mf_gt.num_angle_bins+1)) - 1
                    # Initialize a 1D histogram (8 bins for angles)
                    hist_1D_pred = np.zeros(mf_pred.num_angle_bins)
                    hist_1D_gt = np.zeros(mf_gt.num_angle_bins)
                    # Sum magnitudes into the corresponding angle bins
                    for bin_idx in range(mf_pred.num_angle_bins):
                        hist_1D_pred[bin_idx] = np.sum(np.power(mag_vol_pred[angle_bins_pred == bin_idx], mf_pred.gamma))
                        hist_1D_gt[bin_idx] = np.sum(np.power(mag_vol_gt[angle_bins_gt == bin_idx], mf_gt.gamma))

                    active_bins = np.sum(hist_1D_gt >= 2)
                    if plotted < num_plot_hist1D and np.random.rand() < plot_prob and active_bins >= active_bins_threshold :
                        plot_motion_feat_hist1D(hist_1D_pred, sample, i, row, col, plotted, mf_pred.output_dir, mf_pred.num_angle_bins, "pred")
                        plot_motion_feat_hist1D(hist_1D_gt, sample, i, row, col, plotted, mf_gt.output_dir, mf_gt.num_angle_bins, "gt")
                        plotted += 1
                    # Append this histogram to the motion feature vector avoing division by cero
                    mf_vec_pred.append(hist_1D_pred)
                    mf_vec_gt.append(hist_1D_gt)
        # Concatenate histograms from all volumes into a single vector
        mf_vec_pred = np.concatenate(mf_vec_pred)
        mf_vec_pred = mf_vec_pred / (mf_vec_pred.sum() + 1)

        mf_vec_gt = np.concatenate(mf_vec_gt)
        mf_vec_gt = mf_vec_gt / (mf_vec_gt.sum() + 1)

        all_mf_pred.append(mf_vec_pred)
        all_mf_gt.append(mf_vec_gt)
        # Return the motion feature vectors for all sequences
    return np.array(all_mf_pred), np.array(all_mf_gt)

def get_bhattacharyya_dist_coef(P, Q):
    """
    Given two discrete probabilistic distributions P and Q
    Returns:
    - Bhattacharyya distance
    - Bhattacharyya coefficient
    """
    P = np.array(P)
    Q = np.array(Q)
    # Compute Bhattacharyya coefficient
    bhattacharyya_coef = np.sum(np.sqrt(P * Q))
    # Avoid taking the log of zero by adding a small epsilon
    epsilon = 1e-2
    bhattacharyya_coef = np.clip(bhattacharyya_coef, epsilon, 1.0)
    # Compute Bhattacharyya distance
    bhattacharyya_dist = -np.log(bhattacharyya_coef)

    return bhattacharyya_dist, bhattacharyya_coef