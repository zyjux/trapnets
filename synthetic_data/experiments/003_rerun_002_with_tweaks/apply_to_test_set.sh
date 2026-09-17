#!/usr/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate glp

apply_script="/home/lverhoef/glp/synthetic_data/apply_model.py"

# data_file="/mnt/mlnas01/lverhoef/synthetic_ellipses/train_valid_ds.nc"
data_file="/mnt/data2/lverhoef/synthetic_ellipses/train_valid_ds.nc"

save_file_base="/home/lverhoef/glp/synthetic_data/experiments/002_test_dataset_size/models/cnn"
glp_save_file_base="/home/lverhoef/glp/synthetic_data/experiments/002_test_dataset_size/models/glp_cnn"
aug_save_file_base="/home/lverhoef/glp/synthetic_data/experiments/002_test_dataset_size/models/aug_cnn"

experiment_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

dataset_ratios=(1.0 0.5 0.25 0.1 0.05 0.01 0.005 0.002)

start_idx=8000
end_idx=9000

for ratio in "${dataset_ratios[@]}"; do
  safe_ratio=${ratio//\./_}
  printf "\nRatio: %s" "${ratio}"
  printf "\n------------"

  # Next apply the CNN
  printf "\nApplying CNN"
  cfg_file="${experiment_dir}/test_config_file.yml"
  sed -i "s/\(ds_size_ratio:\).*/\1 ${ratio}/" "${cfg_file}"
  full_save_file="${save_file_base}_${safe_ratio}.pt"
  sed -i "s/\(model_save_file:\).*/\1 \"${full_save_file//\//\\\/}\"/" "${cfg_file}"
  out_file="${experiment_dir}/outputs/cnn_test_${safe_ratio}.nc"
  log_file="${experiment_dir}/logs/cnn_test_apply_${safe_ratio}.log"
  err_file="${experiment_dir}/logs/cnn_test_apply_${safe_ratio}.err"
  python "${apply_script}" "${cfg_file}" "${out_file}" --data_file="${data_file}" \
    --start_idx="${start_idx}" --end_idx="${end_idx}" > "${log_file}" 2>"${err_file}"

  # Then apply the GLP CNN
  printf "\nApplying GLP CNN"
  cfg_file="${experiment_dir}/test_glp_config_file.yml"
  sed -i "s/\(ds_size_ratio:\).*/\1 ${ratio}/" "${cfg_file}"
  full_save_file="${glp_save_file_base}_${safe_ratio}.pt"
  sed -i "s/\(model_save_file:\).*/\1 \"${full_save_file//\//\\\/}\"/" "${cfg_file}"
  out_file="${experiment_dir}/outputs/glp_test_${safe_ratio}.nc"
  log_file="${experiment_dir}/logs/glp_test_apply_${safe_ratio}.log"
  err_file="${experiment_dir}/logs/glp_test_apply_${safe_ratio}.err"
  python "${apply_script}" "${cfg_file}" "${out_file}" --data_file="${data_file}" \
    --start_idx="${start_idx}" --end_idx="${end_idx}" > "${log_file}" 2>"${err_file}"

  # Finally apply the Augmented CNN
  printf "\nApplying Augmented CNN"
  cfg_file="${experiment_dir}/aug_test_config_file.yml"
  sed -i "s/\(ds_size_ratio:\).*/\1 ${ratio}/" "${cfg_file}"
  full_save_file="${aug_save_file_base}_${safe_ratio}.pt"
  sed -i "s/\(model_save_file:\).*/\1 \"${full_save_file//\//\\\/}\"/" "${cfg_file}"
  out_file="${experiment_dir}/outputs/aug_test_${safe_ratio}.nc"
  log_file="${experiment_dir}/logs/aug_test_apply_${safe_ratio}.log"
  err_file="${experiment_dir}/logs/aug_test_apply_${safe_ratio}.err"
  python "${apply_script}" "${cfg_file}" "${out_file}" --data_file="${data_file}" \
    --start_idx="${start_idx}" --end_idx="${end_idx}" > "${log_file}" 2>"${err_file}"

  printf "\n"
done
