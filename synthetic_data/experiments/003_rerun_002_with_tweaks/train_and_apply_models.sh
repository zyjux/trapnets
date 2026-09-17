#!/usr/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate glp

train_script="/home/lverhoef/glp/synthetic_data/train_model.py"
apply_script="/home/lverhoef/glp/synthetic_data/apply_model.py"

# data_file="/mnt/mlnas01/lverhoef/synthetic_ellipses/train_valid_ds.nc"
data_file="/mnt/data2/lverhoef/synthetic_ellipses/train_valid_ds.nc"

save_file_base="/home/lverhoef/glp/synthetic_data/experiments/003_rerun_002_with_tweaks/models/cnn"
glp_save_file_base="/home/lverhoef/glp/synthetic_data/experiments/003_rerun_002_with_tweaks//models/glp_cnn"
aug_save_file_base="/home/lverhoef/glp/synthetic_data/experiments/003_rerun_002_with_tweaks//models/aug_cnn"

experiment_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

dataset_ratios=(1.0 0.5 0.25 0.1 0.05 0.01 0.005 0.002)
# dataset_ratios=(0.005)

for ratio in "${dataset_ratios[@]}"; do
  safe_ratio=${ratio//\./_}
  printf "\nRatio: %s" "${ratio}"
  printf "\n------------"

  # Train CNN while creating the appropriate logs
  printf "\nTraining CNN"
  cfg_file="$experiment_dir/test_config_file.yml"
  sed -i "s/\(ds_size_ratio:\).*/\1 ${ratio}/" "${cfg_file}"
  full_save_file="${save_file_base}_${safe_ratio}.pt"
  sed -i "s/\(model_save_file:\).*/\1 \"${full_save_file//\//\\\/}\"/" "${cfg_file}"
  log_file="$experiment_dir/logs/cnn_${safe_ratio}.log"
  err_file="$experiment_dir/logs/cnn_${safe_ratio}.err"
  python "${train_script}" "${cfg_file}" > "${log_file}" 2> "${err_file}"

  # Then train GLP CNN
  printf "\nTraining GLP CNN"
  cfg_file="$experiment_dir/test_glp_config_file.yml"
  sed -i "s/\(ds_size_ratio:\).*/\1 ${ratio}/" "${cfg_file}"
  full_save_file="${glp_save_file_base}_${safe_ratio}.pt"
  sed -i "s/\(model_save_file:\).*/\1 \"${full_save_file//\//\\\/}\"/" "${cfg_file}"
  log_file="$experiment_dir/logs/glp_${safe_ratio}.log"
  err_file="$experiment_dir/logs/glp_${safe_ratio}.err"
  python "${train_script}" "${cfg_file}" > "${log_file}" 2> "${err_file}"

  # Finally train augmented CNN
  printf "\nTraining Augmented CNN"
  cfg_file="$experiment_dir/aug_test_config_file.yml"
  sed -i "s/\(ds_size_ratio:\).*/\1 ${ratio}/" "${cfg_file}"
  full_save_file="${aug_save_file_base}_${safe_ratio}.pt"
  sed -i "s/\(model_save_file:\).*/\1 \"${full_save_file//\//\\\/}\"/" "${cfg_file}"
  log_file="$experiment_dir/logs/aug_${safe_ratio}.log"
  err_file="$experiment_dir/logs/aug_${safe_ratio}.err"
  python "${train_script}" "${cfg_file}" > "${log_file}" 2> "${err_file}"

  # Next apply the CNN
  printf "\nApplying CNN"
  cfg_file="${experiment_dir}/test_config_file.yml"
  out_file="${experiment_dir}/outputs/cnn_validation_${safe_ratio}.nc"
  log_file="${experiment_dir}/logs/cnn_apply_${safe_ratio}.log"
  err_file="${experiment_dir}/logs/cnn_apply_${safe_ratio}.err"
  python "${apply_script}" "${cfg_file}" "${out_file}" --data_file="${data_file}" \
    > "${log_file}" 2>"${err_file}"

  # Then apply the GLP CNN
  printf "\nApplying GLP CNN"
  cfg_file="${experiment_dir}/test_glp_config_file.yml"
  out_file="${experiment_dir}/outputs/glp_validation_${safe_ratio}.nc"
  log_file="${experiment_dir}/logs/glp_apply_${safe_ratio}.log"
  err_file="${experiment_dir}/logs/glp_apply_${safe_ratio}.err"
  python "${apply_script}" "${cfg_file}" "${out_file}" --data_file="${data_file}" \
    > "${log_file}" 2>"${err_file}"

  # Finally apply the Augmented CNN
  printf "\nApplying Augmented CNN"
  cfg_file="${experiment_dir}/aug_test_config_file.yml"
  out_file="${experiment_dir}/outputs/aug_validation_${safe_ratio}.nc"
  log_file="${experiment_dir}/logs/aug_apply_${safe_ratio}.log"
  err_file="${experiment_dir}/logs/aug_apply_${safe_ratio}.err"
  python "${apply_script}" "${cfg_file}" "${out_file}" --data_file="${data_file}" \
    > "${log_file}" 2>"${err_file}"

done
