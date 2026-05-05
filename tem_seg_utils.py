# ------------------------------------------------------------
# Utility File
# AUTHOR: NITISH SARKER
# ------------------------------------------------------------

# ------------------------------------------------------------
# Image reading and preprocessing
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage import io, color, exposure, filters, morphology, measure, segmentation
from sklearn.cluster import KMeans

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cv2

# ------------------------------------------------------------
# Image reading and preprocessing
# ------------------------------------------------------------
def read_grayscale_image(path):
    """
    Read grayscale, RGB, or RGBA image and return normalized grayscale image in [0, 1].
    """
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)
    elif img.ndim == 2:
        img = img.astype(float)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img

def contrast_preprocess(img, use_clahe=True, use_smoothing=False, smoothing_method="gaussian", gaussian_sigma=0.5, median_radius=1, clip_limit=0.03):
    """
    Recommended order:
        normalized image -> light smoothing (optional) -> CLAHE -> segmentation
    """
    img = img.astype(float)
    if use_smoothing:
        if smoothing_method == "gaussian":
            img = filters.gaussian(
                img,
                sigma=gaussian_sigma,
                preserve_range=True
            )
        elif smoothing_method == "median":
            img = filters.median(
                img,
                morphology.disk(median_radius)
            )
        else:
            raise ValueError("smoothing_method must be 'gaussian' or 'median'")
    if use_clahe:
        img = exposure.equalize_adapthist(
            img,
            clip_limit=clip_limit
        )
    else:
        img = exposure.rescale_intensity(
            img,
            in_range="image",
            out_range=(0, 1)
        )
    return img

# ------------------------------------------------------------
# KMeans segmentation
# ------------------------------------------------------------
def kmeans_segment_classes(img, n_classes=4, random_state=42):
    """
    Segment image intensity using KMeans and sort classes by mean intensity.
    Returns:
        labels, class_means, sorted_classes
    """
    pixels = img.reshape(-1, 1)
    km = KMeans(
        n_clusters=n_classes,
        random_state=random_state,
        n_init=20
    )
    labels_flat = km.fit_predict(pixels)
    labels = labels_flat.reshape(img.shape)
    class_means = np.array([
        img[labels == k].mean() for k in range(n_classes)
    ])
    sorted_classes = np.argsort(class_means)
    return labels, class_means, sorted_classes


def get_selected_populations(sorted_classes, population_mode="darkest_plus_second"):
    """
    Select which contrast populations to analyze.
    population_mode:
        "darkest_only", "darkest_plus_second"
    """
    if population_mode == "darkest_only":
        return [
            {
                "population_name": "darkest",
                "rank": 1,
                "class_id": int(sorted_classes[0]),
                "cmap": "turbo",
                "mask_color": "red"
            }
        ]

    elif population_mode == "darkest_plus_second":
        return [
            {
                "population_name": "darkest",
                "rank": 1,
                "class_id": int(sorted_classes[0]),
                "cmap": "turbo",
                "mask_color": "red"
            },
            {
                "population_name": "second_darkest",
                "rank": 2,
                "class_id": int(sorted_classes[1]),
                "cmap": "plasma",
                "mask_color": "yellow"
            }
        ]
    else:
        raise ValueError(
            "population_mode must be 'darkest_only' or 'darkest_plus_second'"
        )

# ------------------------------------------------------------
# Mask cleanup and measurement
# ------------------------------------------------------------
def clean_mask(mask, pixel_size_nm, min_feature_diameter_nm=2.0):
    """
    Remove small objects and smooth mask.
    The area cutoff is based on the equivalent circular area corresponding to min_feature_diameter_nm.
    """
    min_radius_nm = min_feature_diameter_nm / 2
    min_area_nm2 = np.pi * min_radius_nm**2
    min_area_pixels = max(
        1,
        int(np.ceil(min_area_nm2 / (pixel_size_nm**2)))
    )
    cleaned = morphology.remove_small_objects(
        mask.astype(bool),
        min_size=min_area_pixels
    )
    cleaned = morphology.remove_small_holes(
        cleaned,
        area_threshold=min_area_pixels
    )
    cleaned = morphology.binary_opening(cleaned, morphology.disk(1))
    cleaned = morphology.binary_closing(cleaned, morphology.disk(1))

    return cleaned, min_area_pixels

def measure_features(
    mask,
    img_original,
    pixel_size_nm,
    image_name,
    feature_population,
    feature_class_rank,
    kmeans_class_id,
    size_filter_column="feret_diameter_max_nm",
    min_feature_size_nm=2.0,
    max_feature_size_nm=None
):
    """
    Measure connected features in one population mask.
    Recommended size_filter_column:
        "feret_diameter_max_nm" for irregular domains
        "equivalent_diameter_nm" for near-circular domains
    """
    labeled = measure.label(mask)
    props = measure.regionprops(labeled, intensity_image=img_original)
    records = []
    for p in props:
        area_px = p.area
        area_nm2 = area_px * pixel_size_nm**2
        equivalent_diameter_nm = p.equivalent_diameter_area * pixel_size_nm
        feret_diameter_max_nm = p.feret_diameter_max * pixel_size_nm
        major_axis_length_nm = p.major_axis_length * pixel_size_nm
        minor_axis_length_nm = p.minor_axis_length * pixel_size_nm
        metric_dict = {
            "equivalent_diameter_nm": equivalent_diameter_nm,
            "feret_diameter_max_nm": feret_diameter_max_nm,
            "major_axis_length_nm": major_axis_length_nm,
            "minor_axis_length_nm": minor_axis_length_nm
        }
        if size_filter_column not in metric_dict:
            raise ValueError(
                f"size_filter_column must be one of {list(metric_dict.keys())}"
            )
        size_for_filter = metric_dict[size_filter_column]
        if size_for_filter < min_feature_size_nm:
            continue
        if max_feature_size_nm is not None:
            if size_for_filter > max_feature_size_nm:
                continue
        circularity = np.nan
        if p.perimeter > 0:
            circularity = 4 * np.pi * p.area / (p.perimeter**2)
        records.append({
            "image": image_name,
            "feature_population": feature_population,
            "feature_class_rank": feature_class_rank,
            "kmeans_class_id": kmeans_class_id,
            "feature_id": int(p.label),
            "area_px": area_px,
            "area_nm2": area_nm2,
            "equivalent_diameter_nm": equivalent_diameter_nm,
            "feret_diameter_max_nm": feret_diameter_max_nm,
            "major_axis_length_nm": major_axis_length_nm,
            "minor_axis_length_nm": minor_axis_length_nm,
            "aspect_ratio_major_minor":
                major_axis_length_nm / (minor_axis_length_nm + 1e-12),
            "circularity": circularity,
            "solidity": p.solidity,
            "extent": p.extent,
            "centroid_y_px": p.centroid[0],
            "centroid_x_px": p.centroid[1],
            "mean_intensity": p.mean_intensity
        })
    df = pd.DataFrame(records)

    return df, labeled

def estimate_kmeans_dark_cluster_threshold(
    image_paths,
    read_func,
    preprocess_func,
    segment_func,
    n_classes=4,
    random_state=42,
    selected_rank=1,
    metric="mean",
    threshold_percentile=75,
    **preprocess_kwargs
):
    """
    Run KMeans on each image, identify the selected dark-ranked cluster,
    compute its darkness metric, and set a dataset-level threshold.
    selected_rank:
        1 = darkest cluster
        2 = second-darkest cluster
    metric:
        "mean"   = mean intensity of selected cluster
        "median" = median intensity of selected cluster
        "p90"    = 90th percentile intensity of selected cluster
    threshold_percentile:
        75 means allow clusters darker than or equal to the
        75th percentile of observed selected-cluster darkness metrics.
    """
    records = []
    for path in image_paths:
        image_name = os.path.splitext(os.path.basename(path))[0]
        img = read_func(path)
        img_proc = preprocess_func(
            img,
            **preprocess_kwargs
        )
        labels, class_means, sorted_classes = segment_func(
            img_proc,
            n_classes=n_classes,
            random_state=random_state
        )
        class_id = int(sorted_classes[selected_rank - 1])
        cluster_pixels = img_proc[labels == class_id]
        if metric == "mean":
            darkness_metric = float(np.mean(cluster_pixels))
        elif metric == "median":
            darkness_metric = float(np.median(cluster_pixels))
        elif metric == "p90":
            darkness_metric = float(np.percentile(cluster_pixels, 90))
        else:
            raise ValueError("metric must be 'mean', 'median', or 'p90'")
        records.append({
            "image": image_name,
            "selected_rank": selected_rank,
            "kmeans_class_id": class_id,
            "darkness_metric": darkness_metric,
            "cluster_mean_intensity": float(class_means[class_id]),
            "cluster_pixel_fraction": float(cluster_pixels.size / img_proc.size)
        })
    df_dark_cluster_metrics = pd.DataFrame(records)
    darkness_threshold = np.percentile(
        df_dark_cluster_metrics["darkness_metric"],
        threshold_percentile
    )

    return darkness_threshold, df_dark_cluster_metrics

# ------------------------------------------------------------
# Overlays
# ------------------------------------------------------------
def make_single_mask_overlay(
    img,
    mask,
    color_name="magenta",
    alpha=0.45
):
    """
    Simple transparent mask overlay.
    """
    color_dict = {
        "red": np.array([1, 0, 0]),
        "yellow": np.array([1, 1, 0]),
        "green": np.array([0, 1, 0]),
        "blue": np.array([0, 0, 1]),
        "cyan": np.array([0, 1, 1]),
        "magenta": np.array([1, 0, 1])
    }
    color = color_dict.get(color_name, np.array([1, 0, 0]))
    base = np.dstack([img, img, img])
    overlay = base.copy()
    overlay[mask] = (
        (1 - alpha) * overlay[mask]
        + alpha * color
    )
    return np.clip(overlay, 0, 1)


def make_multi_population_overlay(
    img,
    population_masks,
    alpha=0.45
):
    """
    Overlay selected populations with distinct fixed colors.
    population_masks should be a list of dicts:
        {
            "mask": mask,
            "mask_color": "red",
            "population_name": "darkest"
        }
    """
    overlay = np.dstack([img, img, img]).copy()
    color_dict = {
        "red": np.array([1, 0, 0]),
        "yellow": np.array([1, 1, 0]),
        "green": np.array([0, 1, 0]),
        "blue": np.array([0, 0, 1]),
        "cyan": np.array([0, 1, 1]),
        "magenta": np.array([1, 0, 1])
    }
    for item in population_masks:
        mask = item["mask"]
        color = color_dict.get(
            item.get("mask_color", "red"),
            np.array([1, 0, 0])
        )
        overlay[mask] = (
            (1 - alpha) * overlay[mask]
            + alpha * color
        )

    return np.clip(overlay, 0, 1)


def make_domain_colormap_overlay(
    img,
    labeled_mask,
    df_features,
    size_column="feret_diameter_max_nm",
    cmap_name="cividis",
    alpha=0.70
):
    """
    Color each segmented domain by a feature-size metric while preserving
    the actual segmented domain shape.
    Recommended:
        size_column = "feret_diameter_max_nm"
    """
    base = np.dstack([img, img, img])
    overlay = base.copy()
    if df_features is None or df_features.empty:
        return np.clip(overlay, 0, 1)
    if size_column not in df_features.columns:
        return np.clip(overlay, 0, 1)
    df_valid = df_features.dropna(subset=[size_column]).copy()
    if df_valid.empty:
        return np.clip(overlay, 0, 1)

    values = df_valid[size_column].values
    if np.isclose(values.min(), values.max()):
        norm = mcolors.Normalize(
            vmin=values.min() - 1e-6,
            vmax=values.max() + 1e-6
        )
    else:
        norm = mcolors.Normalize(
            vmin=values.min(),
            vmax=values.max()
        )
    cmap = cm.get_cmap(cmap_name)
    for _, row in df_valid.iterrows():
        feature_id = int(row["feature_id"])
        value = row[size_column]
        rgba = cmap(norm(value))
        rgb = np.array(rgba[:3])
        feature_pixels = labeled_mask == feature_id
        overlay[feature_pixels] = (
            (1 - alpha) * overlay[feature_pixels]
            + alpha * rgb
        )

    return np.clip(overlay, 0, 1)

def make_population_annotation_overlay(
    img,
    df_features,
    pixel_size_nm,
    annotation_mode="number",
    size_column="feret_diameter_max_nm",
    cmap_name="turbo",
    font_scale=0.4,
    thickness=1
):
    """
    Number or circle annotations colored by size.
    Optional QA overlay.
    """
    img_uint8 = (
        255 * exposure.rescale_intensity(img, out_range=(0, 1))
    ).astype(np.uint8)
    overlay = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    if df_features is None or df_features.empty:
        return overlay
    if size_column not in df_features.columns:
        return overlay
    df_valid = df_features.dropna(subset=[size_column]).copy()
    if df_valid.empty:
        return overlay
    values = df_valid[size_column].values
    if np.isclose(values.min(), values.max()):
        norm = mcolors.Normalize(
            vmin=values.min() - 1e-6,
            vmax=values.max() + 1e-6
        )
    else:
        norm = mcolors.Normalize(
            vmin=values.min(),
            vmax=values.max()
        )
    cmap = cm.get_cmap(cmap_name)
    for _, row in df_valid.iterrows():
        rgba = cmap(norm(row[size_column]))
        color_rgb = (
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255)
        )
        cx = int(round(row["centroid_x_px"]))
        cy = int(round(row["centroid_y_px"]))
        feature_id = int(row["feature_id"])
        if annotation_mode == "number":
            cv2.putText(
                overlay,
                text=str(feature_id),
                org=(cx, cy),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color_rgb,
                thickness=thickness,
                lineType=cv2.LINE_AA
            )
        elif annotation_mode == "circle":
            radius_px = int(round((row[size_column] / pixel_size_nm) / 2))
            radius_px = max(radius_px, 2)
            cv2.circle(
                overlay,
                center=(cx, cy),
                radius=radius_px,
                color=color_rgb,
                thickness=2
            )
        else:
            raise ValueError("annotation_mode must be 'number' or 'circle'")

    return overlay

def save_colormap_figure(
    original_img,
    overlay,
    df_features,
    output_path,
    size_column="feret_diameter_max_nm",
    cmap_name="cividis",
    colorbar_label="Feret diameter (nm)",
    base_alpha=0.5,
    dpi=300
):
    """
    Save side-by-side figure:
        Original | Segmented overlay + side colorbar
    base_alpha controls grayscale base visibility.
    Colored segmentation remains fully opaque.
    """
    # ----------------------------
    # Prepare original grayscale RGB
    # ----------------------------
    original = original_img.astype(float)
    if original.max() > 1:
        original = original / 255.0
    base_rgb = np.dstack([original, original, original])
    # Make base image dimmer / more transparent-looking
    faded_base = base_alpha * base_rgb
    # ----------------------------
    # Prepare overlay
    # ----------------------------
    overlay_rgb = overlay.astype(float)
    if overlay_rgb.max() > 1:
        overlay_rgb = overlay_rgb / 255.0
    # Identify colored segmented pixels by comparing overlay to grayscale base
    diff = np.abs(overlay_rgb - base_rgb).sum(axis=2)
    colored_pixels = diff > 0.03
    # Start from faded base, but keep colored segmentation fully visible
    display_overlay = faded_base.copy()
    display_overlay[colored_pixels] = overlay_rgb[colored_pixels]
    display_overlay = np.clip(display_overlay, 0, 1)
    # ----------------------------
    # Setup color normalization
    # ----------------------------
    has_valid_features = (
        df_features is not None
        and not df_features.empty
        and size_column in df_features.columns
        and not df_features[size_column].dropna().empty
    )
    if has_valid_features:
        values = df_features[size_column].dropna()
        if np.isclose(values.min(), values.max()):
            norm = mcolors.Normalize(
                vmin=values.min() - 1e-6,
                vmax=values.max() + 1e-6
            )
        else:
            norm = mcolors.Normalize(
                vmin=values.min(),
                vmax=values.max()
            )
        cmap = plt.colormaps[cmap_name]
    # ----------------------------
    # Plot side-by-side
    # ----------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        gridspec_kw={"width_ratios": [1, 1]}
    )
    axes[0].imshow(base_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    im = axes[1].imshow(display_overlay)
    axes[1].set_title("Segmented overlay")
    axes[1].axis("off")
    # Put colorbar beside the segmented image only
    if has_valid_features:
        sm = cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=axes[1],
            fraction=0.046,
            pad=0.04
        )
        cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    )
    plt.close()

# ------------------------------------------------------------
# Summary and plotting
# ------------------------------------------------------------
def summarize_features_by_population(
    df_all,
    min_features_for_summary=None,
    min_median_size_for_summary=None,
    size_column_for_filter="feret_diameter_max_nm"
):
    """
    Summarize accepted features by image and population.
    Optional summary-level filtering.
    """
    expected_columns = [
        "image",
        "feature_population",
        "n_features",
        "mean_equiv_diameter_nm",
        "median_equiv_diameter_nm",
        "std_equiv_diameter_nm",
        "mean_feret_diameter_max_nm",
        "median_feret_diameter_max_nm",
        "std_feret_diameter_max_nm",
        "mean_major_axis_nm",
        "median_major_axis_nm",
        "mean_minor_axis_nm",
        "median_minor_axis_nm",
        "mean_aspect_ratio",
        "median_aspect_ratio",
        "mean_circularity",
        "median_circularity",
        "mean_solidity",
        "median_solidity",
        "total_area_nm2"
    ]
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=expected_columns)
    summary = (
        df_all
        .groupby(["image", "feature_population"])
        .agg(
            n_features=("feature_id", "count"),
            mean_equiv_diameter_nm=("equivalent_diameter_nm", "mean"),
            median_equiv_diameter_nm=("equivalent_diameter_nm", "median"),
            std_equiv_diameter_nm=("equivalent_diameter_nm", "std"),
            mean_feret_diameter_max_nm=("feret_diameter_max_nm", "mean"),
            median_feret_diameter_max_nm=("feret_diameter_max_nm", "median"),
            std_feret_diameter_max_nm=("feret_diameter_max_nm", "std"),
            mean_major_axis_nm=("major_axis_length_nm", "mean"),
            median_major_axis_nm=("major_axis_length_nm", "median"),
            mean_minor_axis_nm=("minor_axis_length_nm", "mean"),
            median_minor_axis_nm=("minor_axis_length_nm", "median"),
            mean_aspect_ratio=("aspect_ratio_major_minor", "mean"),
            median_aspect_ratio=("aspect_ratio_major_minor", "median"),
            mean_circularity=("circularity", "mean"),
            median_circularity=("circularity", "median"),
            mean_solidity=("solidity", "mean"),
            median_solidity=("solidity", "median"),
            total_area_nm2=("area_nm2", "sum")
        )
        .reset_index()
    )
    if min_features_for_summary is not None:
        summary = summary[
            summary["n_features"] >= min_features_for_summary
        ].copy()
    if min_median_size_for_summary is not None:
        median_col = {
            "feret_diameter_max_nm": "median_feret_diameter_max_nm",
            "equivalent_diameter_nm": "median_equiv_diameter_nm",
            "major_axis_length_nm": "median_major_axis_nm",
            "minor_axis_length_nm": "median_minor_axis_nm"
        }[size_column_for_filter]
        summary = summary[
            summary[median_col] >= min_median_size_for_summary
        ].copy()
    return summary

def plot_summary_by_population(
    summary,
    output_path=None,
    title="TEM Feature Summary by Population",
    size_metric="feret"
):
    """
    Analyst-grade summary plot.
    """
    if summary is None or summary.empty:
        print("No summary data to plot.")
        return
    summary_plot = summary.copy()
    summary_plot["label"] = (
        summary_plot["image"].astype(str)
        + "\n"
        + summary_plot["feature_population"].astype(str)
    )
    x = np.arange(len(summary_plot))
    if size_metric == "feret":
        mean_col = "mean_feret_diameter_max_nm"
        median_col = "median_feret_diameter_max_nm"
        std_col = "std_feret_diameter_max_nm"
        size_title = "Feret diameter"
    else:
        mean_col = "mean_equiv_diameter_nm"
        median_col = "median_equiv_diameter_nm"
        std_col = "std_equiv_diameter_nm"
        size_title = "Equivalent diameter"
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].bar(x, summary_plot["n_features"])
    axes[0].set_title("Feature count")
    axes[0].set_ylabel("Count")
    means = summary_plot[mean_col].values
    stds = summary_plot[std_col].fillna(0).values
    upper_error = 2 * stds
    lower_error = np.minimum(2 * stds, means)
    yerr = np.vstack([lower_error, upper_error])  
    axes[1].bar(
        x,
        means,
        yerr=yerr,
        capsize=5
        )
    axes[1].set_title(f"Mean {size_title} ± 2 SD")
    axes[1].set_ylabel("nm")
    axes[2].bar(x, summary_plot[median_col])
    axes[2].set_title(f"Median {size_title}")
    axes[2].set_ylabel("nm")
    axes[3].bar(x, summary_plot["total_area_nm2"])
    axes[3].set_title("Total segmented area")
    axes[3].set_ylabel("nm²")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(summary_plot["label"], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

def summarize_features_combined_by_image(
    df_all,
    min_features_for_summary=None,
    min_median_size_for_summary=None,
    size_column_for_filter="feret_diameter_max_nm"
):
    """
    Combine all accepted feature populations within each image for final reporting.
    Example: darkest + second_darkest -> one summary row per image.
    """
    expected_columns = [
        "image",
        "n_features",
        "mean_equiv_diameter_nm",
        "median_equiv_diameter_nm",
        "std_equiv_diameter_nm",
        "mean_feret_diameter_max_nm",
        "median_feret_diameter_max_nm",
        "std_feret_diameter_max_nm",
        "mean_major_axis_nm",
        "median_major_axis_nm",
        "mean_minor_axis_nm",
        "median_minor_axis_nm",
        "mean_aspect_ratio",
        "median_aspect_ratio",
        "mean_circularity",
        "median_circularity",
        "mean_solidity",
        "median_solidity",
        "total_area_nm2"
    ]
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=expected_columns)
    summary = (
        df_all
        .groupby("image")
        .agg(
            n_features=("feature_id", "count"),
            mean_equiv_diameter_nm=("equivalent_diameter_nm", "mean"),
            median_equiv_diameter_nm=("equivalent_diameter_nm", "median"),
            std_equiv_diameter_nm=("equivalent_diameter_nm", "std"),
            mean_feret_diameter_max_nm=("feret_diameter_max_nm", "mean"),
            median_feret_diameter_max_nm=("feret_diameter_max_nm", "median"),
            std_feret_diameter_max_nm=("feret_diameter_max_nm", "std"),
            mean_major_axis_nm=("major_axis_length_nm", "mean"),
            median_major_axis_nm=("major_axis_length_nm", "median"),
            mean_minor_axis_nm=("minor_axis_length_nm", "mean"),
            median_minor_axis_nm=("minor_axis_length_nm", "median"),
            mean_aspect_ratio=("aspect_ratio_major_minor", "mean"),
            median_aspect_ratio=("aspect_ratio_major_minor", "median"),
            mean_circularity=("circularity", "mean"),
            median_circularity=("circularity", "median"),
            mean_solidity=("solidity", "mean"),
            median_solidity=("solidity", "median"),
            total_area_nm2=("area_nm2", "sum")
        )
        .reset_index()
    )

    if min_features_for_summary is not None:
        summary = summary[
            summary["n_features"] >= min_features_for_summary
        ].copy()
    if min_median_size_for_summary is not None:
        median_col = {
            "feret_diameter_max_nm": "median_feret_diameter_max_nm",
            "equivalent_diameter_nm": "median_equiv_diameter_nm",
            "major_axis_length_nm": "median_major_axis_nm",
            "minor_axis_length_nm": "median_minor_axis_nm"
        }[size_column_for_filter]
        summary = summary[
            summary[median_col] >= min_median_size_for_summary
        ].copy()
    return summary

def plot_summary_combined_by_image(
    summary,
    output_path=None,
    title="TEM Feature Summary",
    size_metric="feret",
    errorbar_color="purple",
    annotate_median_panel=True
):
    if summary is None or summary.empty:
        print("No summary data to plot.")
        return
    sample_style = {
        "Backbone random": {
            "color": "lightgray",
            "hatch": None
        },
        "Sidechain random 1.5": {
            "color": "skyblue",
            "hatch": "///"
        },
        "Sidechain random 1.8": {
            "color": "skyblue",
            "hatch": None
        },
        "Sidechain block": {
            "color": "red",
            "hatch": None
        },
        "Backbone block": {
            "color": "navajowhite",
            "hatch": None
        },
        "Nafion 212": {
            "color": "black",
            "hatch": None
        }
    }
    desired_order = list(sample_style.keys())
    summary_plot = summary.copy()
    summary_plot["image"] = summary_plot["image"].astype(str)
    existing_order = [
        name for name in desired_order
        if name in summary_plot["image"].values
    ]
    unmatched = [
        name for name in summary_plot["image"].values
        if name not in desired_order
    ]
    final_order = existing_order + unmatched
    summary_plot["image"] = pd.Categorical(
        summary_plot["image"],
        categories=final_order,
        ordered=True
    )

    summary_plot = summary_plot.sort_values("image").copy()
    summary_plot["image"] = summary_plot["image"].astype(str)
    x = np.arange(len(summary_plot))

    if size_metric == "feret":
        mean_col = "mean_feret_diameter_max_nm"
        median_col = "median_feret_diameter_max_nm"
        std_col = "std_feret_diameter_max_nm"
        size_title = "Feret diameter"
    else:
        mean_col = "mean_equiv_diameter_nm"
        median_col = "median_equiv_diameter_nm"
        std_col = "std_equiv_diameter_nm"
        size_title = "Equivalent diameter"
    bar_colors = [
        sample_style.get(str(name), {"color": "gray"})["color"]
        for name in summary_plot["image"]
    ]
    bar_hatches = [
        sample_style.get(str(name), {"hatch": None})["hatch"]
        for name in summary_plot["image"]
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8))

    def styled_bar(ax, values, ylabel, panel_title):
        bars = ax.bar(
            x,
            values,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8
        )
        for bar, hatch in zip(bars, bar_hatches):
            if hatch is not None:
                bar.set_hatch(hatch)

        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)

        return bars

    # 1. Feature count
    styled_bar(
        axes[0],
        summary_plot["n_features"],
        "Count",
        "Feature count"
    )
    # 2. Mean ± 2 SD with capped lower error
    means = summary_plot[mean_col].values
    stds = summary_plot[std_col].fillna(0).values
    upper_error = 2 * stds
    lower_error = np.minimum(2 * stds, means)
    yerr = np.vstack([
        lower_error,
        upper_error
    ])
    bars = axes[1].bar(
        x,
        means,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8
    )
    for bar, hatch in zip(bars, bar_hatches):
        if hatch is not None:
            bar.set_hatch(hatch)
    axes[1].errorbar(
        x,
        means,
        yerr=yerr,
        fmt="none",
        ecolor=errorbar_color,
        elinewidth=1.8,
        capsize=5,
        capthick=1.8
    )
    axes[1].set_title(f"Mean {size_title} ± 2 SD")
    axes[1].set_ylabel("nm")
    # 3. Median size
    styled_bar(
        axes[2],
        summary_plot[median_col],
        "nm",
        f"Median {size_title}"
    )
    # Annotate median panel
    if annotate_median_panel:
        required_cols = [
            "median_equiv_diameter_nm",
            "median_circularity",
        ]
        missing_cols = [
            col for col in required_cols
            if col not in summary_plot.columns
        ]
        if len(missing_cols) == 0:
            y_max = summary_plot[median_col].max()

            for i, row in summary_plot.reset_index(drop=True).iterrows():
                median_equiv = row["median_equiv_diameter_nm"]
                median_circ = row["median_circularity"]

                if pd.isna(median_equiv) or pd.isna(median_circ):
                    continue

                annotation_text = (
                    f"({median_equiv:.1f}, "
                    f"{median_circ:.1f})"
                )
                bar_height = row[median_col]
                axes[2].text(
                    i,
                    bar_height + 0.03 * y_max,
                    annotation_text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

            axes[2].set_ylim(
                0,
                y_max * 1.35
            )
        else:
            print(
                "Skipping median-panel annotations. Missing columns:",
                missing_cols
            )
    # 4. Total area
    styled_bar(
        axes[3],
        summary_plot["total_area_nm2"],
        "nm²",
        "Total segmented area"
    )
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            summary_plot["image"].astype(str),
            rotation=45,
            ha="right"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=1.03
    )
    if annotate_median_panel:
        fig.text(
            0.5,
            -0.04,
            "Median panel annotation: "
            "(median equivalent diameter in nm, median circularity)",
            ha="center",
            fontsize=10
        )
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()