import numpy as np
from sklearn.preprocessing import MinMaxScaler

class MotionFeatureExtractor:
    def __init__(self, nsamples, one_seq_example, f, k, num_magnitude_bins=9, num_angle_bins=8):
        self.f = f
        self.k = k
        self.nsamples = nsamples
        self._, self.r, self.c, self.F = one_seq_example.shape
        self.N = self.r * self.c,
        self.num_magnitude_bins = num_magnitude_bins
        self.num_angle_bins = num_angle_bins
        self.scaler = MinMaxScaler(feature_range=(0, 255))

    def get_vel_vector_field(self, one_seq):
        v_x = one_seq[1, :, :, :]  # Shape (r, c, F)
        v_y = one_seq[2, :, :, :]  # Shape (r, c, F)
        # Reshape v_x and v_y to (F, N) where N = r * c
        v_x_flat = v_x.reshape(self.N, self.F).T  # Shape (F, N)
        v_y_flat = v_y.reshape(self.N, self.F).T  # Shape (F, N)
        # Stack v_x and v_y along the last axis to create U of shape (F, N, 2)
        U = np.stack((v_x_flat, v_y_flat), axis=-1)  # Shape (F, N, 2)
        return U

    def compute_norm_angle_4samples(self, seq_list):
        """
        Computes the magnitude and angle for each key point in the vector field U.
        """
        mag_rho = np.zeros((self.nsamples, self.F, self.N))
        angle_phi = np.zeros((self.nsamples, self.F, self.N))

        for sample in range(self.nsamples):
            one_pred_seq = seq_list[sample].cpu().numpy()
            U = self.get_vel_vector_field(one_pred_seq)
            mag_rho[sample] = np.sqrt(U[..., 0]**2 + U[..., 1]**2)  # Shape (F, N)
            angle_phi[sample] = np.abs(np.arctanh(U[..., 0] / U[..., 1])) # Shape (F, N)

        return mag_rho, angle_phi

    def mag_rho_transform(self, mag_rho):
        mag_rho_transf = np.zeros((self.nsamples, self.F, self.N))

        for sample in range(self.nsamples):
            mag_rho_clipped = np.clip(mag_rho[sample], 0, 255)
            mag_rho_normalized = self.scaler.fit_transform(mag_rho_clipped).reshape(self.F, self.N)
            mag_rho_log = np.log2(mag_rho_normalized + 1)
            mag_rho_transf[sample] = mag_rho_log

        return mag_rho_transf

    def motion_feature_2D_hist(self, seq_list):
        mag_rho, angle_phi = self.compute_norm_angle_4samples(seq_list)
        mag_rho_transf =  self.mag_rho_transform(mag_rho)
        all_motion_feature_vectors = []

        for sample in range(self.nsamples):
            motion_feature_vector = []
            # Reshape each frame's data back into a (r, c) grid
            mag_rho_reshaped = mag_rho_transf[sample].reshape(self.F, self.r, self.c)
            angle_phi_reshaped = angle_phi[sample].reshape(self.F, self.r, self.c)
            for i in range(0, self.F, self.f):  # Temporal volumes of size f
                for row in range(0, self.r, self.k):  # Spatial rows (k x k blocks)
                    for col in range(0, self.c, self.k):  # Spatial columns (k x k blocks)
                        # Extract a sub-volume of size (f, k, k)
                        mag_volume = mag_rho_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        angle_volume = angle_phi_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        # Quantize magnitudes and angles
                        mag_bins = np.digitize(mag_volume, np.linspace(0, 255, self.num_magnitude_bins+1)) - 1
                        angle_bins = np.digitize(angle_volume, np.linspace(0, 2*np.pi, self.num_angle_bins+1)) - 1
                        # Compute 2D histogram (quantized magnitude vs angle)
                        hist_2D, _, _ = np.histogram2d(mag_bins, angle_bins, bins=[self.num_magnitude_bins, self.num_angle_bins])
                        # Flatten and add to the motion feature vector
                        motion_feature_vector.append(hist_2D.flatten())
            # Concatenate histograms from all volumes into a single vector
            motion_feature_vector = np.concatenate(motion_feature_vector)
            all_motion_feature_vectors.append(motion_feature_vector)
        # Return the motion feature vectors for all sequences
        return np.array(all_motion_feature_vectors)

    def motion_feature_1D_hist(self, seq_list):
        mag_rho, angle_phi = self.compute_norm_angle_4samples(seq_list)
        mag_rho_transf =  self.mag_rho_transform(mag_rho)
        all_motion_feature_vectors = []

        for sample in range(self.nsamples):
            motion_feature_vector = []
            # Reshape each frame's data back into a (r, c) grid
            mag_rho_reshaped = mag_rho_transf[sample].reshape(self.F, self.r, self.c)
            angle_phi_reshaped = angle_phi[sample].reshape(self.F, self.r, self.c)
            for i in range(0, self.F, self.f):  # Temporal volumes of size f
                for row in range(0, self.r, self.k):  # Spatial rows (k x k blocks)
                    for col in range(0, self.c, self.k):  # Spatial columns (k x k blocks)
                        # Extract a sub-volume of size (f, k, k)
                        mag_volume = mag_rho_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        angle_volume = angle_phi_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        # Quantize only angles
                        angle_bins = np.digitize(angle_volume, np.linspace(0, 2*np.pi, self.num_angle_bins+1)) - 1
                        # Initialize a 1D histogram (8 bins for angles)
                        hist_1D = np.zeros(self.num_angle_bins)
                        # Sum magnitudes into the corresponding angle bins
                        for bin_idx in range(self.num_angle_bins):
                            hist_1D[bin_idx] = np.sum(mag_volume[angle_bins == bin_idx])
                        # Append this histogram to the motion feature vector
                        motion_feature_vector.append(hist_1D)
            # Concatenate histograms from all volumes into a single vector
            motion_feature_vector = np.concatenate(motion_feature_vector)
            all_motion_feature_vectors.append(motion_feature_vector)
        # Return the motion feature vectors for all sequences
        return np.array(all_motion_feature_vectors)