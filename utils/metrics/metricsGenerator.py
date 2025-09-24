import numpy as np
import logging, json, os
import pandas as pd

from utils.plot.plot_metrics import createBoxPlot, createBoxPlot_bhatt, merge_and_plot_boxplot

class MetricsGenerator:
    HEADERS = {
        "PSNR": "rho,vx,vy",
        "SSIM": "rho,vx,vy",
        "MAX-PSNR": "rho,vx,vy",
        "MAX-SSIM": "rho,vx,vy",
        "MF_MSE": "MSE_Hist_2D_Based,MSE_Hist_1D_Based",
        "MF_BHATT_DIST": "BHATT_DIST_Hist_2D_Based,BHATT_DIST_Hist_1D_Based",
        "MF_BHATT_COEF": "BHATT_COEF_Hist_2D_Based,BHATT_COEF_Hist_1D_Based",
        "ENERGY": "GT,PRED",
        "MIN-ENERGY": "GT,PRED",
        "RE_DENSITY": "re_f6,re_f7,re_f8",
        "MIN_RE_DENSITY": "re_f6,re_f7,re_f8",
    }
    def __init__(self, pred_seq_list, gt_seq_list, metrics_params, output_dir=None):
        self.pred_seq_list = pred_seq_list
        self.gt_seq_list = gt_seq_list
        self.params = metrics_params
        self.output_dir = output_dir
        self.data_dict = {name: None for name in self.HEADERS}

    def _get_mprops_ranges(self, mprops_factor, mprops_count):
        nsamples = len(self.gt_seq_list)
        # Initialize arrays to store max and min values for each sample and each property
        max_vals = np.zeros((nsamples, mprops_count))
        min_vals = np.zeros((nsamples, mprops_count))

        for i, one_gt_seq in enumerate(self.gt_seq_list):
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

    def _my_psnr(self, y_gt, y_hat, data_range, eps):
        # Compute mean squared error
        err = np.mean((y_gt - y_hat) ** 2, dtype=np.float64)
        # Prevent overflow and division by zero
        err = max(err, eps)
        # Calculate PSNR
        tmp_num = 20 * np.log10(data_range)
        tmp_den = 10 * np.log10(err)
        psnr = tmp_num - tmp_den
        return psnr

    def compute_psnr_metric(self, chunkRepdPastSeq, eps):
        """
        Compute PSNR and MAX-PSNR for predicted and gt sequences.
        """
        mprops_factor = np.array(self.params.PRED_MPROPS_FACTOR)[:self.params.MPROPS_COUNT, np.newaxis, np.newaxis, np.newaxis]
        nsamples = len(self.pred_seq_list)
        _, _, _, pred_len = self.pred_seq_list[0].shape
        nsamples_psnr = np.zeros((nsamples, self.params.MPROPS_COUNT))
        max_psnr = np.zeros((nsamples//chunkRepdPastSeq, self.params.MPROPS_COUNT))
        mprops_factor = np.array(mprops_factor)

        rho_range, vx_range, vy_range = self._get_mprops_ranges(mprops_factor, self.params.MPROPS_COUNT)
        logging.info(f'Range of macroprops \n rho:{rho_range:.4f}, vx:{vx_range:.4f} and vy:{vy_range:.4f}')

        for i in range(nsamples):
            one_pred_seq = self.pred_seq_list[i].cpu().numpy()
            one_gt_seq = self.gt_seq_list[i].cpu().numpy()

            one_pred_seq = one_pred_seq * mprops_factor
            one_gt_seq = one_gt_seq * mprops_factor

            psnr_rho, psnr_vx, psnr_vy = 0, 0, 0
            for j in range(pred_len):
                psnr_rho += self._my_psnr(one_gt_seq[0, :, :, j], one_pred_seq[0, :, :, j], data_range=rho_range, eps=eps)
                psnr_vx  += self._my_psnr(one_gt_seq[1, :, :, j], one_pred_seq[1, :, :, j], data_range=vx_range, eps=eps)
                psnr_vy  += self._my_psnr(one_gt_seq[2, :, :, j], one_pred_seq[2, :, :, j], data_range=vy_range, eps=eps)

            # Average PSNR across frames, except for unc channel
            nsamples_psnr[i] = (psnr_rho/pred_len, psnr_vx/pred_len, psnr_vy/pred_len)

        # Compute the MAX PSNR by repeteaded seqs on each macroprops
        for i in range(0, nsamples, chunkRepdPastSeq):
            psnr_chunk = nsamples_psnr[i:i+chunkRepdPastSeq]
            max_rho = psnr_chunk[:,0].max()
            max_vx  = psnr_chunk[:,1].max()
            max_vy  = psnr_chunk[:,2].max()
            max_psnr[i // chunkRepdPastSeq] = (max_rho, max_vx, max_vy)

        self.data_dict['PSNR']= nsamples_psnr
        self.data_dict['MAX-PSNR'] = max_psnr

    def _save_metric_data(self, match, data, metric, header, samples_per_batch):
        file_name = f"{self.output_dir}/{metric}_NS{samples_per_batch}_{match.group()}.csv"
        np.savetxt(file_name, data, delimiter=",", header=header, comments="")
        return file_name

    def save_data_metrics(self, match, title, samples_per_batch):
        """
        Save all non-empty metrics to CSV and record their filenames in a JSON.
        """
        metrics_filenames_dict = {"title": title}
        # Save each non-empty metric with its required data
        for metric_name, metric_header in self.HEADERS.items():
            data = self.data_dict[metric_name]
            if data is not None:
                logging.info(f"Saving metric {metric_name}, entries: {data.shape[0]}")
                file_name = self._save_metric_data(match, data, metric_name, metric_header, samples_per_batch)
                metrics_filenames_dict[metric_name] = file_name

        json_path = os.path.join(self.output_dir, "metrics_files.json")
        with open(json_path, "w") as json_file:
            json.dump(metrics_filenames_dict, json_file, indent=2)
        logging.info(f"Metrics filenames saved to {json_path}")

    def save_metrics_boxplots(self, title):
        # Convert the dictionary of arrays into a dictionary of DataFrames
        metrics_df_dict = {key: pd.DataFrame(value, columns=self.HEADER[key].split(",")) for key, value in self.data_dict.items()}
        if len(metrics_df_dict['MAX-PSNR']) != 0:
            merge_and_plot_boxplot(df_max=metrics_df_dict['MAX-PSNR'], df=metrics_df_dict['PSNR'], title=f"PSNR and MAX-PSNR of {title}", save_path=f"{self.output_dir}/BP_PSNR.png", ytick_step=5)
        if len(metrics_df_dict['MAX-SSIM']) != 0:
            merge_and_plot_boxplot(df_max=metrics_df_dict['MAX-SSIM'], df=metrics_df_dict['SSIM'], title=f"SSIM and MAX-SSIM of {title}", save_path=f"{self.output_dir}/BP_SSIM.png", ytick_step=0.2)
        if len(metrics_df_dict['MF_MSE']) != 0:
            createBoxPlot(metrics_df_dict['MF_MSE'], title=f"MSE of Motion feature of {title}", columns_to_plot=self.HEADERS["MF_MSE"].split(","), save_path=f"{self.output_dir}/BP_MF_MSE.png", ytick_step=0.0002)
        if len(metrics_df_dict['MF_BHATT_COEF']) != 0:
            createBoxPlot_bhatt(metrics_df_dict['MF_BHATT_COEF'], metrics_df_dict['MF_BHATT_DIST'], title=f"BHATT of Motion feature of {title}", save_path=f"{self.output_dir}/BP_BHATT.png")
        if len(metrics_df_dict['MIN-ENERGY']) != 0:
            merge_and_plot_boxplot(df_max=metrics_df_dict['MIN-ENERGY'], df=metrics_df_dict['ENERGY'], title=f"ENERGY and MIN-ENERGY of {title}", save_path=f"{self.output_dir}/BP_ENERGY.png", ytick_step=None, prefix='min-')
        if len(metrics_df_dict['MIN_RE_DENSITY']) != 0:
            merge_and_plot_boxplot(df_max=metrics_df_dict['MIN_RE_DENSITY'], df=metrics_df_dict['RE_DENSITY'], title=f"Relative DENSITY and MIN_RE_DENSITY of {title}", save_path=f"{self.output_dir}/BP_RE_DENSITY.png", ytick_step=2, prefix='min-', outliersFlag=True)
