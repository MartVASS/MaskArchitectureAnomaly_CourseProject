import glob
import hashlib
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
EOMT_ROOT = PROJECT_ROOT / "eomt"
sys.path.insert(0, str(EOMT_ROOT))


DATASETS = {
    "RoadAnomaly": "data/Validation_Dataset/RoadAnomaly/images/*",
    "RoadAnomaly21": "data/Validation_Dataset/RoadAnomaly21/images/*",
    "fs_static": "data/Validation_Dataset/fs_static/images/*",
    "LostFound": "data/Validation_Dataset/FS_LostFound_full/images/*",
    "RoadObsticle21": "data/Validation_Dataset/RoadObsticle21/images/*",
}


def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    if np.all(tpr < 0.95):
        return 1.0
    return float(fpr[np.argmax(tpr >= 0.95)])


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_checkpoint(config):
    from huggingface_hub import hf_hub_download

    name = config["trainer"]["logger"]["init_args"]["name"]
    return hf_hub_download(
        repo_id=f"S362484/{name}",
        filename="eomt_cityscapes.bin",
    )


def build_eomt_model(config, checkpoint_path, img_size, num_classes, device):
    import importlib

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {
        k: v for k, v in network_cfg["init_args"].items() if k != "encoder"
    }
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {
        k: v for k, v in config["model"]["init_args"].items() if k != "network"
    }
    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded EoMT checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return model


def gt_path_from_image_path(path):
    path_gt = path.replace("images", "labels_masks")
    if "RoadObsticle21" in path_gt:
        path_gt = path_gt.replace("webp", "png")
    if "fs_static" in path_gt:
        path_gt = path_gt.replace("jpg", "png")
    if "RoadAnomaly" in path_gt:
        path_gt = path_gt.replace("jpg", "png")
    return path_gt


def load_anomaly_gt(path_gt, img_size):
    mask = Image.open(path_gt)
    mask = Resize(img_size, interpolation=Image.NEAREST)(mask)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in path_gt:
        ood_gts = np.where(ood_gts == 2, 1, ood_gts)
    if "LostAndFound" in path_gt:
        ood_gts = np.where(ood_gts == 0, 255, ood_gts)
        ood_gts = np.where(ood_gts == 1, 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in path_gt:
        ood_gts = np.where(ood_gts == 14, 255, ood_gts)
        ood_gts = np.where(ood_gts < 20, 0, ood_gts)
        ood_gts = np.where(ood_gts == 255, 1, ood_gts)

    return ood_gts


def forward_eomt(model, image_path, img_size, device):
    transform = Compose([Resize(img_size), ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) * 255.0

    with torch.no_grad():
        mask_logits_per_layer, class_logits_per_layer = model(image_tensor)

    mask_logits = mask_logits_per_layer[-1]
    class_logits = class_logits_per_layer[-1]
    mask_logits = F.interpolate(
        mask_logits,
        size=img_size,
        mode="bilinear",
        align_corners=False,
    )

    mask_probs = mask_logits.sigmoid()
    class_probs = class_logits.softmax(dim=-1)[..., :-1]
    pixel_probs = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
    pixel_logits = torch.einsum("bqhw,bqc->bchw", mask_probs, class_logits[..., :-1])

    return pixel_probs[0], pixel_logits[0], mask_probs[0], class_logits[0]


def temperature_pixel_probs(mask_probs, class_logits, temperature):
    class_probs = (class_logits / temperature).softmax(dim=-1)[..., :-1]
    return torch.einsum("qhw,qc->chw", mask_probs, class_probs)


def anomaly_scores(pixel_probs, pixel_logits, mask_probs, class_logits, methods, temperature):
    scores = {}
    if "msp" in methods:
        scores["msp"] = 1.0 - pixel_probs.max(dim=0).values
    if "entropy" in methods:
        prob_dist = pixel_probs / pixel_probs.sum(dim=0, keepdim=True).clamp_min(1e-8)
        entropy = -(prob_dist * prob_dist.clamp_min(1e-8).log()).sum(dim=0)
        scores["entropy"] = entropy / np.log(pixel_probs.shape[0])
    if "maxlogit" in methods:
        scores["maxlogit"] = -pixel_logits.max(dim=0).values
    if "rba" in methods:
        scores["rba"] = -pixel_probs.tanh().sum(dim=0)
    if "temperature" in methods:
        temperature_probs = temperature_pixel_probs(mask_probs, class_logits, temperature)
        scores["temperature"] = 1.0 - temperature_probs.max(dim=0).values
    return {k: v.detach().cpu().numpy().astype("float32") for k, v in scores.items()}


def cache_path_for_image(predictions_dir, checkpoint_name, dataset_name, image_path, img_size):
    rel_path = os.path.relpath(image_path, PROJECT_ROOT)
    cache_key = hashlib.sha1(f"{rel_path}|{img_size}".encode("utf-8")).hexdigest()[:16]
    stem = Path(image_path).stem
    return predictions_dir / checkpoint_name / dataset_name / f"{stem}_{cache_key}.pt"


def load_or_compute_logits(model, path, img_size, device, predictions_dir, checkpoint_name, dataset_name, use_cache):
    cache_path = None
    if predictions_dir is not None:
        cache_path = cache_path_for_image(predictions_dir, checkpoint_name, dataset_name, path, img_size)
        if use_cache and cache_path.exists():
            cached = torch.load(cache_path, map_location=device, weights_only=False)
            if "mask_probs" in cached and "class_logits" in cached:
                return (
                    cached["pixel_probs"],
                    cached["pixel_logits"],
                    cached["mask_probs"],
                    cached["class_logits"],
                )

    pixel_probs, pixel_logits, mask_probs, class_logits = forward_eomt(model, path, img_size, device)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pixel_probs": pixel_probs.detach().cpu().half(),
                "pixel_logits": pixel_logits.detach().cpu().half(),
                "mask_probs": mask_probs.detach().cpu().half(),
                "class_logits": class_logits.detach().cpu().half(),
            },
            cache_path,
        )

    return pixel_probs, pixel_logits, mask_probs, class_logits


def evaluate_dataset(
    model,
    dataset_name,
    dataset_path,
    methods,
    temperatures,
    img_size,
    device,
    predictions_dir,
    checkpoint_name,
    use_cache,
):
    score_values = {
        (method, temperature): []
        for method in methods
        for temperature in method_temperatures(method, temperatures)
    }
    label_values = []
    paths = sorted(glob.glob(os.path.expanduser(dataset_path)))

    for path in paths:
        print(path)
        path_gt = gt_path_from_image_path(path)
        ood_gts = load_anomaly_gt(path_gt, img_size)
        if 1 not in np.unique(ood_gts):
            continue

        pixel_probs, pixel_logits, mask_probs, class_logits = load_or_compute_logits(
            model,
            path,
            img_size,
            device,
            predictions_dir,
            checkpoint_name,
            dataset_name,
            use_cache,
        )
        pixel_probs = pixel_probs.float()
        pixel_logits = pixel_logits.float()
        mask_probs = mask_probs.float()
        class_logits = class_logits.float()

        valid_mask = (ood_gts == 0) | (ood_gts == 1)
        labels = (ood_gts[valid_mask] == 1).astype(np.uint8)
        label_values.append(labels)

        for method in methods:
            for temperature in method_temperatures(method, temperatures):
                scores = anomaly_scores(
                    pixel_probs,
                    pixel_logits,
                    mask_probs,
                    class_logits,
                    [method],
                    temperature,
                )
                score_values[(method, temperature)].append(scores[method][valid_mask])

        del pixel_probs, pixel_logits, mask_probs, class_logits, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not label_values:
        raise RuntimeError(f"No valid anomaly labels found for dataset path: {dataset_path}")

    labels = np.concatenate(label_values)
    results = {}
    for method in methods:
        for temperature in method_temperatures(method, temperatures):
            values = np.concatenate(score_values[(method, temperature)])
            auprc = average_precision_score(labels, values)
            fpr95 = fpr_at_95_tpr(values, labels)
            results[(method, temperature)] = {"auprc": auprc, "fpr95": fpr95}
    return results


def best_temperature(results, method):
    method_results = {
        temperature: metrics
        for (result_method, temperature), metrics in results.items()
        if result_method == method
    }
    if not method_results:
        return None
    return max(method_results.items(), key=lambda item: item[1]["auprc"])


def parse_temperatures(values):
    temperatures = [float(value) for value in values]
    if any(temperature <= 0 for temperature in temperatures):
        raise ValueError("All temperatures must be positive.")
    return temperatures


def method_temperatures(method, temperatures):
    if method == "temperature":
        return temperatures
    return [1.0]


def default_checkpoint_name(config, checkpoint_path):
    if checkpoint_path is not None:
        return Path(checkpoint_path).stem
    return config["trainer"]["logger"]["init_args"]["name"]


def main():
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["msp"],
        choices=["msp", "entropy", "maxlogit", "rba", "temperature"],
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        default=["0.5", "0.75", "1.1"],
        help="Positive temperature values. Use several values to sweep calibration.",
    )
    parser.add_argument(
        "--config",
        default=str(EOMT_ROOT / "configs" / "dinov2" / "cityscapes" / "semantic" / "eomt_base_640.yaml"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=(1024, 1024), metavar=("H", "W"))
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--results-file", default="results_eomt.txt")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    img_size = tuple(args.img_size)

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or download_checkpoint(config)
    checkpoint_name = args.checkpoint_name or default_checkpoint_name(config, args.checkpoint)
    predictions_dir = Path(args.predictions_dir) if args.predictions_dir is not None else None
    temperatures = parse_temperatures(args.temperatures)
    model = build_eomt_model(config, checkpoint_path, img_size, args.num_classes, device)

    dataset_list = []
    if args.datasets is not None:
        for dataset_name in args.datasets:
            dataset_list.append((dataset_name, DATASETS[dataset_name]))
    else:
        dataset_list = [("RoadAnomaly21", DATASETS["RoadAnomaly21"])]

    with open(args.results_file, "a", encoding="utf-8") as file:
        file.write(f"\nEoMT anomaly evaluation | checkpoint={checkpoint_name}\n")
        for dataset_name, dataset_path in dataset_list:
            print("\nDataset:", dataset_name)
            results = evaluate_dataset(
                model,
                dataset_name,
                dataset_path,
                args.methods,
                temperatures,
                img_size,
                device,
                predictions_dir,
                checkpoint_name,
                use_cache=not args.no_cache,
            )
            for (method, temperature), metrics in results.items():
                temp_text = f"T={temperature:<7g}" if method == "temperature" else " " * 9
                line = (
                    f"{checkpoint_name:>36}  {dataset_name:>16}  "
                    f"{method:>11}  {temp_text}  "
                    f"AUPRC: {metrics['auprc'] * 100.0:.4f}  "
                    f"FPR@TPR95: {metrics['fpr95'] * 100.0:.4f}"
                )
                print(line)
                file.write(line + "\n")
            for method in args.methods:
                if method != "temperature" or len(temperatures) <= 1:
                    continue
                best = best_temperature(results, method)
                if best is None:
                    continue
                temperature, metrics = best
                line = (
                    f"{checkpoint_name:>36}  {dataset_name:>16}  "
                    f"{method:>11}  BEST_T={temperature:<7g}  "
                    f"AUPRC: {metrics['auprc'] * 100.0:.4f}  "
                    f"FPR@TPR95: {metrics['fpr95'] * 100.0:.4f}"
                )
                print(line)
                file.write(line + "\n")


if __name__ == "__main__":
    main()
