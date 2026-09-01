import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_loaders.data_loader import get_dataset_loaders
from data_loaders.configs import dataset_config

from data_loaders.cub_loader import (
    CLASS_NAMES,
    CONCEPT_SEMANTICS,
    SELECTED_CONCEPTS,
)

import textwrap

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 27,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


PANEL_LABEL_SIZE = 27
CLASS_TITLE_SIZE = 26
ANNOTATION_TYPE_SIZE = 20
CONCEPT_LABEL_SIZE = 18
CONCEPT_TEXT_SIZE = 17
TEXT_X = 0.06
TOP_LABEL_Y = 0.98
TOP_TEXT_Y = 0.82
BOTTOM_LABEL_Y = 0.36
BOTTOM_TEXT_Y = 0.20

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pretty_name(name):
    name = (
        str(name)
        .replace("+", " ")
        .replace("_", " ")
        .strip()
    )

    replacements = {
        "oldworld": "old world",
        "Black footed Albatross": "Black-footed Albatross",
    }

    return replacements.get(name, name)

def wrap_text(text, width=36):
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def format_concept_lines(concepts):
    if not concepts:
        return "none"
    return "\n".join(concepts)

def tensor_to_display_image(x, channel_order="rgb"):
    """
    Convert a tensor or array into an RGB image for matplotlib.

    channel_order:
        "rgb": channels are already RGB
        "bgr": channels are BGR and must be reversed
    """
    image = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    image = np.squeeze(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = image.astype(np.float32)

    # Reverse ImageNet normalization.
    if image.min() < 0:
        image = image * IMAGENET_STD + IMAGENET_MEAN
    elif image.max() > 1.5:
        image = image / 255.0

    # aPY appears to be stored as BGR.
    if channel_order.lower() == "bgr":
        image = image[..., [2, 1, 0]]

    return np.clip(image, 0.0, 1.0)


def find_sample(data_loader, target_class=None, occurrence=0):
    """
    Return one sample from a DataLoader yielding (image, class, concepts).

    occurrence determines which matching image is selected:
      0 -> first matching sample
      1 -> second matching sample
      ...
    """
    matched = 0

    for batch in data_loader:
        if len(batch) != 3:
            raise ValueError(
                f"Expected batches of (image, class, concepts), got {len(batch)} items."
            )

        images, labels, concepts = batch

        for index in range(len(labels)):
            label = int(labels[index].detach().cpu().item())

            if target_class is not None and label != target_class:
                continue

            if matched == occurrence:
                return (
                    images[index],
                    label,
                    concepts[index].detach().cpu().numpy(),
                )

            matched += 1

    raise ValueError(
        f"No sample found for target_class={target_class}, occurrence={occurrence}."
    )


def select_continuous_concepts(values, concept_names, n_top=5, n_bottom=5):
    """Select the highest- and lowest-valued continuous concepts."""
    values = np.asarray(values).reshape(-1)

    sorted_indices = np.argsort(values)
    bottom_indices = sorted_indices[:n_bottom]
    top_indices = sorted_indices[::-1][:n_top]

    def get_name(idx):
        if concept_names is not None and idx < len(concept_names):
            return pretty_name(concept_names[idx])
        return f"Concept {idx}"

    highest = [f"{get_name(idx)} ({values[idx]:.2f})" for idx in top_indices]
    lowest = [f"{get_name(idx)} ({values[idx]:.2f})" for idx in bottom_indices]

    return highest, lowest


def select_binary_concepts(
    values,
    concept_names,
    n_present=4,
    n_absent=2,
):
    """Select a compact set of present and absent binary concepts."""
    values = np.asarray(values).reshape(-1)

    present_indices = np.flatnonzero(values >= 0.5)
    absent_indices = np.flatnonzero(values < 0.5)

    present_indices = present_indices[:n_present]
    absent_indices = absent_indices[:n_absent]

    def get_name(idx):
        if concept_names is not None and idx < len(concept_names):
            return pretty_name(concept_names[idx])
        return f"Concept {idx}"

    present = [f"{get_name(idx)} (1)" for idx in present_indices]
    absent = [f"{get_name(idx)} (0)" for idx in absent_indices]

    return present, absent

def wrap_concept_line(prefix, concepts, width=58):
    text = prefix + ", ".join(concepts)
    return textwrap.fill(
        text,
        width=width,
        subsequent_indent=" " * len(prefix),
    )

def raw_image_to_display(image):
    """Convert an unprocessed image array to RGB for matplotlib."""
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image = image.astype(np.float32)

    if image.max() <= 1.5:
        image = image * 255.0

    return np.clip(image, 0, 255).astype(np.uint8)

def save_dataset_panel(
    image,
    class_name,
    annotation_description,
    output_path,
    highest_concepts=None,
    lowest_concepts=None,
    present_concepts=None,
    absent_concepts=None,
):
    """Save one consistently sized dataset panel."""

    fig = plt.figure(figsize=(5.8, 7.4))

    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[4.5, 2.0],
        left=0.04,
        right=0.98,
        top=0.925,
        bottom=0.055,
        hspace=0.12,
    )

    # Image block
    image_axis = fig.add_subplot(grid[0])
    image_axis.imshow(image)
    image_axis.set_title(
        f"Class: {pretty_name(class_name)}",
        fontsize=CLASS_TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )
    image_axis.axis("off")
    image_axis.set_anchor("N")

    # Text block
    text_axis = fig.add_subplot(grid[1])
    text_axis.axis("off")

    # Only reserve space for the annotation description when one is shown.
    if annotation_description:
        text_axis.text(
            TEXT_X,
            TOP_LABEL_Y,
            annotation_description,
            transform=text_axis.transAxes,
            va="top",
            fontsize=ANNOTATION_TYPE_SIZE,
            fontstyle="italic",
        )

    if highest_concepts is not None:
        # Top block: Highest-valued concepts
        text_axis.text(
            TEXT_X,
            TOP_LABEL_Y,
            "Highest-valued concepts:",
            transform=text_axis.transAxes,
            va="top",
            fontsize=CONCEPT_LABEL_SIZE,
            fontweight="bold",
        )

        text_axis.text(
            TEXT_X,
            TOP_TEXT_Y,
            format_concept_lines(highest_concepts),
            transform=text_axis.transAxes,
            va="top",
            fontsize=CONCEPT_TEXT_SIZE,
            linespacing=1.25,
        )

        # Bottom block: Lowest-valued concepts
        text_axis.text(
            TEXT_X,
            BOTTOM_LABEL_Y,
            "Lowest-valued concepts:",
            transform=text_axis.transAxes,
            va="top",
            fontsize=CONCEPT_LABEL_SIZE,
            fontweight="bold",
        )

        text_axis.text(
            TEXT_X,
            BOTTOM_TEXT_Y,
            format_concept_lines(lowest_concepts),
            transform=text_axis.transAxes,
            va="top",
            fontsize=CONCEPT_TEXT_SIZE,
            linespacing=1.25,
        )

    else:
        # Top block: Present
        text_axis.text(
            TEXT_X,
            TOP_LABEL_Y,
            "Present:",
            transform=text_axis.transAxes,
            va="top",
            fontsize=CONCEPT_LABEL_SIZE,
            fontweight="bold",
        )

        text_axis.text(
            TEXT_X,
            TOP_TEXT_Y,
            format_concept_lines(present_concepts),
            transform=text_axis.transAxes,
            va="top",
            fontsize=CONCEPT_TEXT_SIZE,
            linespacing=1.25,
        )

        # Bottom block: Absent
        text_axis.text(
            TEXT_X,
            BOTTOM_LABEL_Y,
            "Absent:",
            transform=text_axis.transAxes,
            va="top",
            fontsize=CONCEPT_LABEL_SIZE,
            fontweight="bold",
        )

        text_axis.text(
            TEXT_X,
            BOTTOM_TEXT_Y,
            format_concept_lines(absent_concepts),
            transform=text_axis.transAxes,
            va="top",
            fontsize=CONCEPT_TEXT_SIZE,
            linespacing=1.25,
        )

    # Do not use bbox_inches="tight": all panels must retain the same canvas.
    fig.savefig(
        output_path,
        dpi=350,
        pad_inches=0,
    )
    plt.close(fig)


def load_cub_class_names(root_dir):
    """Load CUB class names from classes.txt."""
    path = Path(root_dir) / "classes.txt"

    if not path.exists():
        return None

    names = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                names.append(pretty_name(parts[1]))

    return names


def load_concept_names(path, expected_count):
    """
    Load concept names from a text file.

    Supports lines such as:
        1 has_bill_shape::dagger
        has_bill_shape::dagger
    """
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Concept-name file not found: {path}")

    names = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                name = parts[1]
            else:
                name = line

            names.append(pretty_name(name))

    if len(names) != expected_count:
        print(
            f"Warning: found {len(names)} concept names, but the model uses "
            f"{expected_count}. Generic CUB concept names will be used."
        )
        return None

    return names


def create_combined_figure(panel_paths, output_path):
    """Combine equally sized panels with consistent alignment."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19.5, 8.5),
    )

    panel_labels = ["","",""]

    for axis, path, label in zip(axes, panel_paths, panel_labels):
        panel = plt.imread(path)

        axis.imshow(panel)
        axis.set_title(
            label,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            pad=8,
        )
        axis.axis("off")
        axis.set_anchor("N")

    fig.subplots_adjust(
        left=0.012,
        right=0.988,
        top=0.94,
        bottom=0.015,
        wspace=0.035,
    )

    # Again, avoid tight cropping so all axes remain aligned.
    fig.savefig(
        output_path,
        dpi=350,
        pad_inches=0,
    )
    plt.close(fig)


def resolve_named_class(class_name, label_to_idx, dataset_name):
    if class_name not in label_to_idx:
        available = ", ".join(sorted(label_to_idx.keys()))
        raise ValueError(
            f"Unknown {dataset_name} class '{class_name}'. "
            f"Available classes: {available}"
        )

    return int(label_to_idx[class_name])


def generate_awa2(args):
    config = dataset_config["AwA2"].copy()
    config["data_dir"] = args.data_dir
    config["seed"] = args.seed
    config["batch_size"] = args.batch_size

    result = get_dataset_loaders(
        "awa2",
        config,
        config["seed"],
        args.save_dir,
        config["batch_size"],
        args.data_dir,
    )

    _, _, test_loader, _, concept_names, _, _, label_to_idx = result

    target_idx = resolve_named_class(
        args.awa2_class,
        label_to_idx,
        "AwA2",
    )

    image, label, concepts = find_sample(
        test_loader,
        target_class=target_idx,
        occurrence=args.awa2_occurrence,
    )

    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    class_name = idx_to_label[label]

    highest, lowest = select_continuous_concepts(
        concepts,
        concept_names,
        n_top=args.n_highest,
        n_bottom=args.n_lowest,
    )

    output_path = Path(args.output_dir) / "awa2_example.png"

    save_dataset_panel(
        image=tensor_to_display_image(image),
        class_name=class_name,
        annotation_description="",#"Continuous class-level concept profile",
        highest_concepts=highest,
        lowest_concepts=lowest,
        output_path=output_path,
    )

    return output_path


def generate_apy(args):
    """
    Generate the aPY panel from the raw images returned by the CV loader.

    This avoids displaying normalized model-input tensors, which can produce
    visible colour and edge artifacts after inverse normalization.
    """
    config = dataset_config["aPY"].copy()
    config["data_dir"] = args.data_dir
    config["seed"] = args.seed
    config["batch_size"] = args.batch_size
    config["to_crop"] = False

    result = get_dataset_loaders(
        "aPY_cv",
        config,
        config["seed"],
        args.save_dir,
        config["batch_size"],
        args.data_dir,
    )

    (
        images,
        _,
        labels,
        _,
        annotations,
        _,
        _,
        classes,
        concept_names,
        label_to_idx,
        _,
        _,
        _,
    ) = result

    images = np.asarray(images)
    labels = np.asarray(labels).reshape(-1)
    annotations = np.asarray(annotations)

    target_idx = resolve_named_class(
        args.apy_class,
        label_to_idx,
        "aPY",
    )

    matching_indices = np.flatnonzero(labels == target_idx)

    if args.apy_occurrence >= len(matching_indices):
        raise ValueError(
            f"aPY class '{args.apy_class}' has only "
            f"{len(matching_indices)} available images, but occurrence "
            f"{args.apy_occurrence} was requested."
        )

    sample_idx = int(matching_indices[args.apy_occurrence])

    image = images[sample_idx]
    label = int(labels[sample_idx])
    concepts = annotations[sample_idx]

    idx_to_label = {
        int(idx): name
        for name, idx in label_to_idx.items()
    }
    class_name = idx_to_label[label]

    # The loader may return concept names as a pandas object.
    if hasattr(concept_names, "values"):
        concept_names = concept_names.values.squeeze()

    concept_names = list(np.asarray(concept_names).reshape(-1))

    present, absent = select_binary_concepts(
        concepts,
        concept_names,
        n_present=args.n_present,
        n_absent=args.n_absent,
    )

    output_path = Path(args.output_dir) / "apy_example.png"

    save_dataset_panel(
        image=raw_image_to_display(image),
        class_name=class_name,
        annotation_description="",#"Binary instance-level concept annotations",
        present_concepts=present,
        absent_concepts=absent,
        output_path=output_path,
    )

    return output_path


def generate_cub(args):
    config = dataset_config["CUB"].copy()
    config["root_dir"] = args.cub_root
    config["seed"] = args.seed
    config["batch_size"] = args.batch_size

    result = get_dataset_loaders(
        "cub",
        config,
        config["seed"],
        args.save_dir,
        config["batch_size"],
        args.data_dir,
    )

    _, _, test_loader = result[:3]

    image, label, concepts = find_sample(
        test_loader,
        target_class=args.cub_class,
        occurrence=args.cub_occurrence,
    )

    if label < len(CLASS_NAMES):
        class_name = CLASS_NAMES[label].replace("_", " ")
    else:
        class_name = f"CUB class {label}"

    concept_names = [
        format_cub_concept_name(CONCEPT_SEMANTICS[index])
        for index in SELECTED_CONCEPTS
    ]

    if len(concept_names) != len(concepts):
        raise ValueError(
            f"CUB loader returned {len(concepts)} concepts, but "
            f"{len(concept_names)} selected concept names were constructed."
        )

    present, absent = select_binary_concepts(
        concepts,
        concept_names,
        n_present=args.n_present,
        n_absent=args.n_absent,
    )

    output_path = Path(args.output_dir) / "cub_example.png"

    save_dataset_panel(
        image=tensor_to_display_image(image),
        class_name=class_name,
        annotation_description="",#"Binary instance-level concept annotations",
        present_concepts=present,
        absent_concepts=absent,
        output_path=output_path,
    )

    return output_path


def format_cub_concept_name(name):
    """
    Convert:
        has_wing_color::black
    into:
        wing: black
    """
    name = str(name).strip()

    if name.startswith("has_"):
        name = name[4:]

    if "::" in name:
        group, value = name.split("::", 1)

        group = group.replace("_", " ").strip()
        value = value.replace("_", " ").strip()

        # Make group names shorter and more readable
        group = group.replace(" color", "")
        group = group.replace(" pattern", " pattern")
        group = group.replace(" shape", " shape")
        group = group.replace(" length", " length")

        # Small cleanup
        value = value.replace("(up or down)", "up/down")

        return f"{group}: {value}"

    return name.replace("_", " ")


def main():
    parser = argparse.ArgumentParser(
        description="Generate representative dataset examples for the thesis."
    )

    parser.add_argument("--data_dir", default="../data/")
    parser.add_argument(
        "--cub_root",
        default="../data/CUB_200_2011/",
    )
    parser.add_argument(
        "--cub_concepts_file",
        default="../data/CUB_200_2011/attributes/attributes.txt",
        help=(
            "Optional text file containing the 112 CUB concept names. "
            "Generic names are used when omitted."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="manuscript-latex-source/images/datasets",
    )
    parser.add_argument(
        "--save_dir",
        default="./tmp_dataset_examples",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)

    # Representative classes.
    parser.add_argument("--awa2_class", default="zebra")
    parser.add_argument("--apy_class", default="bicycle")
    parser.add_argument(
        "--cub_class",
        type=int,
        default=0,
        help="Zero-based CUB class index.",
    )

    # Select another image from the same class by changing occurrence.
    parser.add_argument("--awa2_occurrence", type=int, default=0)
    parser.add_argument("--apy_occurrence", type=int, default=0)
    parser.add_argument("--cub_occurrence", type=int, default=17)

    parser.add_argument(
        "--n_highest",
        type=int,
        default=3,
        help="Number of highest-valued AwA2 concepts to display.",
    )
    parser.add_argument(
        "--n_lowest",
        type=int,
        default=3,
        help="Number of lowest-valued AwA2 concepts to display.",
    )
    parser.add_argument("--n_present", type=int, default=3)
    parser.add_argument("--n_absent", type=int, default=3)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    awa2_path = generate_awa2(args)
    apy_path = generate_apy(args)
    cub_path = generate_cub(args)

    combined_path = Path(args.output_dir) / "dataset_examples_combined.png"

    create_combined_figure(
        [awa2_path, apy_path, cub_path],
        combined_path,
    )

    print("\nGenerated files:")
    print(f"  {awa2_path}")
    print(f"  {apy_path}")
    print(f"  {cub_path}")
    print(f"  {combined_path}")


if __name__ == "__main__":
    main()