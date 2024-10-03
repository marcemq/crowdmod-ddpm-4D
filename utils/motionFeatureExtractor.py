import numpy as np
from sklearn.preprocessing import MinMaxScaler

class MotionFeatureExtractor:
    def __init__(self, seq_list, f, k, gamma=0.5, num_magnitude_bins=9, num_angle_bins=8):
        self.f = f
        self.k = k
        self.gamma = gamma
        self.nsamples = len(seq_list)
        self.seq_list = seq_list
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
            mag_rho[sample] = np.sqrt(U[..., 0]**2 + U[..., 1]**2)  # Shape (F, N)
            angle_phi[sample] = np.abs(np.arctan2(U[..., 1], U[..., 0]))
        return mag_rho, angle_phi

    def mag_rho_transform(self):
        mag_rho_transf = np.zeros((self.nsamples, self.F, self.N))
        total_clipped = 0
        for sample in range(self.nsamples):
            mag_rho_sample = self.mag_rho[sample]
            mag_rho_clipped = np.clip(mag_rho_sample, 0, 255)
            mag_rho_normalized = self.scaler.fit_transform(mag_rho_clipped).reshape(self.F, self.N)
            mag_rho_log = np.log2(mag_rho_normalized + 1)
            mag_rho_transf[sample] = mag_rho_log
            # Count clipped values to have a sense of information loss
            clipped_values = np.sum((mag_rho_sample < 0) | (mag_rho_sample > 255))
            total_clipped += clipped_values
        print(f'Total clipped values at mag_rho_transform:{total_clipped}')

        return mag_rho_transf

    def motion_feature_2D_hist(self):
        all_motion_feature_vectors = []
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
                        hist_2D, _, _ = np.histogram2d(mag_volume, angle_volume, bins=[self.num_magnitude_bins, self.num_angle_bins], range=[[0, 255], [0, 2*np.pi]])
                        # Flatten and add to the motion feature vector
                        motion_feature_vector.append(hist_2D.flatten())
            # Concatenate histograms from all volumes into a single vector
            motion_feature_vector = np.concatenate(motion_feature_vector)
            all_motion_feature_vectors.append(motion_feature_vector)
        # Return the motion feature vectors for all sequences
        return np.array(all_motion_feature_vectors)

    def _get_mag_angle_counts(self):
        all_mag_rho_volumnes = []
        all_angle_phi_volumnes = []
        # Collect all magnitude values from each sub-volume
        for sample in range(self.nsamples):
            mag_rho_reshaped = self.mag_rho_transf[sample].reshape(self.F, self.r, self.c)
            angle_phi_reshaped = self.angle_phi[sample].reshape(self.F, self.r, self.c)
            
            for i in range(0, self.F, self.f):  # Temporal volumes of size f
                for row in range(0, self.r, self.k):  # Spatial rows (k x k blocks)
                    for col in range(0, self.c, self.k):  # Spatial columns (k x k blocks)
                        # Extract a sub-volume of size (f, k, k)
                        mag_volume = mag_rho_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        angle_volume = angle_phi_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        all_mag_rho_volumnes.append(mag_volume)  # Collect all magnitudes
                        all_angle_phi_volumnes.append(angle_volume)

        # Convert to numpy array and compute histogram
        all_mag_rho_volumnes = np.array(all_mag_rho_volumnes)
        all_angle_phi_volumnes = np.array(all_angle_phi_volumnes)

        unique_magnitudes, counts = np.unique(all_mag_rho_volumnes, return_counts=True)
        mag_rho_count_dict = dict(zip(unique_magnitudes, counts))
        unique_angles, counts = np.unique(all_angle_phi_volumnes, return_counts=True)
        angle_phi_count_dict = dict(zip(unique_angles, counts))

        return mag_rho_count_dict, angle_phi_count_dict

    def motion_feature_1D_hist(self):
        all_motion_feature_vectors = []
        all_mag_rho_volumnes = []

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
                        mag_bin_indices = np.digitize(mag_volume, np.linspace(0, 8, self.num_magnitude_bins + 1)) - 1
                        mag_counts = np.bincount(mag_bin_indices, minlength=self.num_magnitude_bins)
                        mag_volume_norm = np.array([value / mag_counts[bin_idx] for value, bin_idx in zip(mag_volume, mag_bin_indices)])
                        all_mag_rho_volumnes.append(mag_volume_norm)

                        angle_volume = angle_phi_reshaped[i:i+self.f, row:row+self.k, col:col+self.k].flatten()
                        angle_bin_indices = np.digitize(angle_volume, np.linspace(0, 2*np.pi, self.num_angle_bins+1)) - 1
                        angle_counts = np.bincount(angle_bin_indices, minlength=self.num_angle_bins)
                        angle_volume_norm = np.array([value / angle_counts[bin_idx] for value, bin_idx in zip(angle_volume, angle_bin_indices)])

                        # Quantize only angles
                        angle_bins = np.digitize(angle_volume, np.linspace(0, 2*np.pi, self.num_angle_bins+1)) - 1
                        # Initialize a 1D histogram (8 bins for angles)
                        hist_1D = np.zeros(self.num_angle_bins)
                        # Sum magnitudes into the corresponding angle bins
                        for bin_idx in range(self.num_angle_bins):
                            hist_1D[bin_idx] = np.sum(np.power(mag_volume[angle_bins == bin_idx], self.gamma))
                        # Append this histogram to the motion feature vector
                        motion_feature_vector.append(hist_1D)
            # Concatenate histograms from all volumes into a single vector
            motion_feature_vector = np.concatenate(motion_feature_vector)
            all_motion_feature_vectors.append(motion_feature_vector)
        # Return the motion feature vectors for all sequences
        return np.array(all_motion_feature_vectors), np.array(all_mag_rho_volumnes)