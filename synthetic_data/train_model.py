import argparse
from time import perf_counter

import numpy as np
import torch
import torchvision.transforms.v2 as tvtf
import xarray as xr
import yaml
from data_utils import DATA_FILE, AddGaussianNoise, Ellipse_Dataset, angle_categorizer
from network_def import CNN, EarlyStopper, crps_loss, model_config, train, validate
from torch.utils.data import DataLoader
from torchinfo import summary

parser = argparse.ArgumentParser(prog="TrainModel", description="Trains GLP models")
parser.add_argument("config_file")
args = parser.parse_args()


def main():
    # Load config file
    with open(args.config_file, "r") as f:
        cfg = model_config(**yaml.load(f, Loader=yaml.SafeLoader))

    loss_function_translation = {"crps": crps_loss}

    hyperparam_config = cfg.training_hyperparameters

    # Load dataset
    full_ds = xr.load_dataset(DATA_FILE)
    # Categorize angles
    full_ds["target"] = (
        "sample",
        angle_categorizer(
            full_ds.angle.values,
            hyperparam_config["num_classes"],
            diagonal=hyperparam_config["diagonal_categorization"],
        ),
    )

    ds = full_ds.isel(sample=slice(8000))
    valid_ds = full_ds.isel(sample=slice(9000, None))

    total_samples = ds.sizes["sample"]
    try:
        desired_samples = int(total_samples * hyperparam_config["ds_size_ratio"])
    except KeyError:
        pass
    else:
        rng = np.random.default_rng(hyperparam_config["random_seed"])
        sample_indices = rng.choice(
            total_samples, desired_samples, replace=False, shuffle=False
        )
        ds = ds.isel(sample=sample_indices)
        print(f"Using reduced dataset of {desired_samples} examples")

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    aug_config = cfg.data_augmentation
    transform_list = []
    if aug_config is not None:
        if "rotations" in aug_config.keys():
            rotate_transform = tvtf.RandomRotation(aug_config["rotations"]["max_degrees"])  # type: ignore
            transform_list += [rotate_transform] * aug_config["rotations"][
                "num_rotations"
            ]
        if "noisings" in aug_config.keys():
            noise_transform = AddGaussianNoise(
                std=aug_config["noisings"]["standard_deviation"]
            )
            transform_list += [noise_transform] * aug_config["noisings"]["num_noisings"]
        if "translations" in aug_config.keys():
            max_shift = aug_config["translations"]["max_shift"]
            translate_transform = tvtf.RandomAffine(0, translate=(max_shift, max_shift))  # type: ignore
            transform_list += [translate_transform] * aug_config["translations"][
                "num_translations"
            ]
    transform_list = tuple(transform_list)

    cnn_train_ds = Ellipse_Dataset(ds, transforms=transform_list)
    cnn_valid_ds = Ellipse_Dataset(valid_ds)

    batch_size = hyperparam_config["batch_size"]
    cnn_train_dataloader = DataLoader(
        cnn_train_ds, num_workers=4, batch_size=batch_size, drop_last=True, shuffle=True
    )
    cnn_valid_dataloader = DataLoader(
        cnn_valid_ds, num_workers=4, batch_size=batch_size
    )

    cnn_model = CNN(cfg).to(device)
    cnn_loss_fn = loss_function_translation[hyperparam_config["loss_function"]]()
    cnn_optimizer = torch.optim.Adam(
        cnn_model.parameters(), lr=hyperparam_config["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, "min")

    if hyperparam_config["early_stopping"]["enabled"]:
        early_stopper = EarlyStopper(
            patience=hyperparam_config["early_stopping"]["patience"],
            min_delta=hyperparam_config["early_stopping"]["min_delta"],
        )

    summary(cnn_model, input_size=(batch_size, 1, 128, 128))

    epochs = hyperparam_config["epochs"]
    print("Training CNN \n")
    t_time_start = perf_counter()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n----------------------------")
        start_time = perf_counter()
        train(
            cnn_train_dataloader,
            cnn_model,
            cnn_loss_fn,
            cnn_optimizer,
            accumulation_batches=hyperparam_config["accumulation_batches"],
            device=device,
        )
        val_loss, val_accuracy = validate(
            cnn_valid_dataloader, cnn_model, cnn_loss_fn, device=device
        )
        scheduler.step(val_loss)
        if hyperparam_config["early_stopping"]["enabled"]:
            if early_stopper.early_stop(val_loss, val_accuracy):
                print(
                    f"Early stopping with minimum validation loss of {early_stopper.min_validation_loss}"
                    f", max accuracy of {early_stopper.max_accuracy}"
                )
                break
            if early_stopper.counter == 0:
                print(f"Validation loss improved, saving model")
                torch.save(cnn_model.state_dict(), cfg.model_save_file)
        else:
            torch.save(cnn_model.state_dict(), cfg.model_save_file)

        print(f"Epoch time: {perf_counter() - start_time:.2f} seconds \n")
    t_time = perf_counter() - t_time_start
    print(
        f"Done! Total training time: {t_time // 60:.0f} minutes, {t_time % 60:.2f} seconds, average epoch time: {t_time/epochs:.2f} seconds"
    )


if __name__ == "__main__":
    main()
