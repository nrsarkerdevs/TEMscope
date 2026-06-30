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
def read_grayscale_image(
    path,
    normalization_method="robust_percentile",
    p_low=1,
    p_high=99
):
    """
    Read grayscale, RGB, or RGBA image and return normalized grayscale image in [0, 1].

    normalization_method:
        "minmax"              = standard min-max normalization
        "robust_percentile"   = percentile-clipped normalization, recommended
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

    if normalization_method == "minmax":
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)

    elif normalization_method == "robust_percentile":
        low = np.percentile(img, p_low)
        high = np.percentile(img, p_high)

        img = np.clip(img, low, high)
        img = (img - low) / (high - low + 1e-12)

    else:
        raise ValueError(
            "normalization_method must be 'minmax' or 'robust_percentile'"
        )

    return img

def read_grayscale_image_raw_display(path):
    """
    Read grayscale, RGB, or RGBA image and return grayscale image scaled only
    for display, without percentile/CLAHE normalization.

    This is intended for reporting the actual input image appearance.
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

    # Preserve relative raw contrast; only scale to 0–1 for matplotlib display
    if img.max() > 1:
        img_display = img / img.max()
    else:
        img_display = img.copy()

    return img_display

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

def add_dark_pixel_rescue_mask(
    raw_mask,
    img_proc,
    dark_percentile=8,
    connection_radius_px=1
):
    """
    Add very dark pixels based on an intensity percentile threshold.

    This helps recover visually dark domains that KMeans does not assign
    to the selected darkest class.

    dark_percentile:
        5 to 10 is conservative.
        Higher values include more dark pixels but may add background texture.
    """

    threshold = np.percentile(img_proc, dark_percentile)

    rescue_mask = img_proc <= threshold

    combined_mask = raw_mask.astype(bool) | rescue_mask.astype(bool)

    combined_mask = morphology.binary_closing(
        combined_mask,
        morphology.disk(connection_radius_px)
    )

    return combined_mask, rescue_mask, threshold


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

    # ------------------------------------------------------------
# Feature-to-feature distance analysis
# ------------------------------------------------------------

def compute_nearest_neighbor_distances(
    df_features,
    pixel_size_column="pixel_size_nm",
    x_col="centroid_x_px",
    y_col="centroid_y_px",
    size_col="equivalent_diameter_nm",
    group_cols=("image", "feature_population")
):
    """
    Compute nearest-neighbor distances between segmented features.

    Distances reported:
        nearest_centroid_distance_nm:
            center-to-center distance between each feature and its nearest neighbor

        nearest_edge_gap_nm:
            approximate edge-to-edge gap:
            centroid distance - radius_i - radius_j
            where radius is estimated from equivalent diameter / 2

    Notes:
        - Uses feature centroids.
        - Edge gap is approximate for irregular domains.
        - Requires at least 2 features per group.
    """

    if df_features is None or df_features.empty:
        return pd.DataFrame()

    required_cols = list(group_cols) + [
        x_col,
        y_col,
        size_col,
        pixel_size_column,
        "feature_id"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df_features.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    distance_records = []

    for group_key, df_group in df_features.groupby(list(group_cols)):
        df_group = df_group.reset_index(drop=True).copy()

        if len(df_group) < 2:
            continue

        pixel_size_nm = float(df_group[pixel_size_column].iloc[0])

        coords_px = df_group[[x_col, y_col]].values.astype(float)
        coords_nm = coords_px * pixel_size_nm

        sizes_nm = df_group[size_col].values.astype(float)
        radii_nm = sizes_nm / 2.0

        # Pairwise centroid distances
        diff = coords_nm[:, None, :] - coords_nm[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

        # Ignore self-distance
        np.fill_diagonal(dist_matrix, np.inf)

        nearest_idx = np.argmin(dist_matrix, axis=1)
        nearest_centroid_dist_nm = dist_matrix[
            np.arange(len(df_group)),
            nearest_idx
        ]

        nearest_edge_gap_nm = (
            nearest_centroid_dist_nm
            - radii_nm
            - radii_nm[nearest_idx]
        )

        nearest_edge_gap_nm = np.maximum(nearest_edge_gap_nm, 0)

        for i, row in df_group.iterrows():
            nn_i = nearest_idx[i]
            nn_row = df_group.iloc[nn_i]

            record = {
                "image": row["image"],
                "feature_population": row["feature_population"],
                "feature_id": row["feature_id"],
                "nearest_neighbor_feature_id": nn_row["feature_id"],
                "nearest_centroid_distance_nm": nearest_centroid_dist_nm[i],
                "nearest_edge_gap_nm": nearest_edge_gap_nm[i],
                "feature_equiv_diameter_nm": row[size_col],
                "nearest_neighbor_equiv_diameter_nm": nn_row[size_col],
                "centroid_x_nm": row[x_col] * pixel_size_nm,
                "centroid_y_nm": row[y_col] * pixel_size_nm,
                "nearest_centroid_x_nm": nn_row[x_col] * pixel_size_nm,
                "nearest_centroid_y_nm": nn_row[y_col] * pixel_size_nm
            }

            # Preserve group metadata if present
            for col in df_group.columns:
                if col in [
                    "n_classes",
                    "population_mode",
                    "size_filter_column",
                    "min_feature_size_nm"
                ]:
                    record[col] = row[col]

            distance_records.append(record)

    return pd.DataFrame(distance_records)


def summarize_nearest_neighbor_distances(
    df_distances,
    group_cols=("image", "feature_population")
):
    """
    Summarize nearest-neighbor distances by image and population.
    """

    expected_cols = [
        "image",
        "feature_population",
        "n_features_with_distance",
        "mean_nearest_centroid_distance_nm",
        "median_nearest_centroid_distance_nm",
        "std_nearest_centroid_distance_nm",
        "mean_nearest_edge_gap_nm",
        "median_nearest_edge_gap_nm",
        "std_nearest_edge_gap_nm"
    ]

    if df_distances is None or df_distances.empty:
        return pd.DataFrame(columns=expected_cols)

    summary = (
        df_distances
        .groupby(list(group_cols))
        .agg(
            n_features_with_distance=("feature_id", "count"),
            mean_nearest_centroid_distance_nm=(
                "nearest_centroid_distance_nm",
                "mean"
            ),
            median_nearest_centroid_distance_nm=(
                "nearest_centroid_distance_nm",
                "median"
            ),
            std_nearest_centroid_distance_nm=(
                "nearest_centroid_distance_nm",
                "std"
            ),
            mean_nearest_edge_gap_nm=(
                "nearest_edge_gap_nm",
                "mean"
            ),
            median_nearest_edge_gap_nm=(
                "nearest_edge_gap_nm",
                "median"
            ),
            std_nearest_edge_gap_nm=(
                "nearest_edge_gap_nm",
                "std"
            )
        )
        .reset_index()
    )

    return summary


def plot_nearest_neighbor_distance_summary(
    distance_summary,
    output_path=None,
    distance_metric="centroid",
    title="Nearest-Neighbor Feature Spacing"
):
    """
    Plot mean and median nearest-neighbor distance by image/population.

    distance_metric:
        "centroid" -> center-to-center nearest-neighbor distance
        "edge"     -> approximate edge-to-edge nearest-neighbor gap
    """

    if distance_summary is None or distance_summary.empty:
        print("No distance summary data to plot.")
        return

    df_plot = distance_summary.copy()

    df_plot["label"] = (
        df_plot["image"].astype(str)
        + "\n"
        + df_plot["feature_population"].astype(str)
    )

    x = np.arange(len(df_plot))

    sample_style = {
        "Backbone random": {"color": "lightgray", "hatch": None},
        "Sidechain random 1.5": {"color": "skyblue", "hatch": "///"},
        "Sidechain random 1.8": {"color": "skyblue", "hatch": None},
        "Sidechain block": {"color": "red", "hatch": None},
        "Backbone block": {"color": "navajowhite", "hatch": None},
        "Nafion 212": {"color": "black", "hatch": None},
    }

    bar_colors = [
        sample_style.get(str(name), {"color": "gray"})["color"]
        for name in df_plot["image"]
    ]

    bar_hatches = [
        sample_style.get(str(name), {"hatch": None})["hatch"]
        for name in df_plot["image"]
    ]


    if distance_metric == "centroid":
        mean_col = "mean_nearest_centroid_distance_nm"
        median_col = "median_nearest_centroid_distance_nm"
        std_col = "std_nearest_centroid_distance_nm"
        ylabel = "Nearest centroid distance, nm"
        panel_label = "centroid-to-centroid"
    elif distance_metric == "edge":
        mean_col = "mean_nearest_edge_gap_nm"
        median_col = "median_nearest_edge_gap_nm"
        std_col = "std_nearest_edge_gap_nm"
        ylabel = "Nearest edge gap, nm"
        panel_label = "edge-to-edge gap"
    else:
        raise ValueError("distance_metric must be 'centroid' or 'edge'")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    means = df_plot[mean_col].values
    stds = df_plot[std_col].fillna(0).values

    upper_error = stds
    lower_error = np.minimum(stds, means)

    yerr = np.vstack([lower_error, upper_error])

    bars0 = axes[0].bar(
        x,
        means,
        yerr=yerr,
        capsize=5,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8
    )

    for bar, hatch in zip(bars0, bar_hatches):
        if hatch is not None:
            bar.set_hatch(hatch)

    axes[0].set_title(f"Mean nearest-neighbor {panel_label} ± SD")
    axes[0].set_ylabel(ylabel)

    bars1 = axes[1].bar(
        x,
        df_plot[median_col].values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8
    )

    for bar, hatch in zip(bars1, bar_hatches):
        if hatch is not None:
            bar.set_hatch(hatch)

    axes[1].set_title(f"Median nearest-neighbor {panel_label}")
    axes[1].set_ylabel(ylabel)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["label"], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.04)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_feature_diameter_vs_distance(
    feature_summary,
    distance_summary,
    output_path=None,
    statistic="median",
    diameter_metric="equivalent",
    distance_metric="centroid",
    distance_population=None,
    title="Feature Diameter vs Feature-to-Feature Distance",
    errorbar_color="purple"
):
    """
    Plot feature diameter and nearest-neighbor feature-to-feature distance
    side by side for each sample.

    If statistic="mean", error bars show ±1 SD.
    If statistic="median", no SD error bars are plotted.
    """

    if feature_summary is None or feature_summary.empty:
        print("No feature summary data to plot.")
        return

    if distance_summary is None or distance_summary.empty:
        print("No distance summary data to plot.")
        return

    if statistic not in ["mean", "median"]:
        raise ValueError("statistic must be 'mean' or 'median'")

    # ----------------------------
    # Choose diameter metric
    # ----------------------------
    if diameter_metric == "equivalent":
        diameter_col = f"{statistic}_equiv_diameter_nm"
        diameter_std_col = "std_equiv_diameter_nm"
        diameter_label = f"{statistic.capitalize()} equivalent diameter"
    elif diameter_metric == "feret":
        diameter_col = f"{statistic}_feret_diameter_max_nm"
        diameter_std_col = "std_feret_diameter_max_nm"
        diameter_label = f"{statistic.capitalize()} Feret diameter"
    else:
        raise ValueError("diameter_metric must be 'equivalent' or 'feret'")

    # ----------------------------
    # Choose distance metric
    # ----------------------------
    if distance_metric == "centroid":
        distance_col = f"{statistic}_nearest_centroid_distance_nm"
        distance_std_col = "std_nearest_centroid_distance_nm"
        distance_label = (
            f"{statistic.capitalize()} nearest-neighbor centroid distance"
        )
    elif distance_metric == "edge":
        distance_col = f"{statistic}_nearest_edge_gap_nm"
        distance_std_col = "std_nearest_edge_gap_nm"
        distance_label = (
            f"{statistic.capitalize()} nearest-neighbor edge gap"
        )
    else:
        raise ValueError("distance_metric must be 'centroid' or 'edge'")

    # ----------------------------
    # Check columns
    # ----------------------------
    required_feature_cols = ["image", diameter_col]
    required_distance_cols = ["image", distance_col]

    if statistic == "mean":
        required_feature_cols.append(diameter_std_col)
        required_distance_cols.append(distance_std_col)

    missing_feature_cols = [
        col for col in required_feature_cols
        if col not in feature_summary.columns
    ]

    missing_distance_cols = [
        col for col in required_distance_cols
        if col not in distance_summary.columns
    ]

    if missing_feature_cols:
        raise ValueError(
            f"Missing feature summary columns: {missing_feature_cols}"
        )

    if missing_distance_cols:
        raise ValueError(
            f"Missing distance summary columns: {missing_distance_cols}"
        )

    # ----------------------------
    # Sample colors
    # ----------------------------
    sample_style = {
        "Backbone random": {"color": "lightgray", "hatch": None},
        "Sidechain random 1.5": {"color": "skyblue", "hatch": "///"},
        "Sidechain random 1.8": {"color": "skyblue", "hatch": None},
        "Sidechain block": {"color": "red", "hatch": None},
        "Backbone block": {"color": "navajowhite", "hatch": None},
        "Nafion 212": {"color": "black", "hatch": None},
    }

    desired_order = list(sample_style.keys())

    feature_plot = feature_summary.copy()
    distance_plot = distance_summary.copy()

    feature_plot["image"] = feature_plot["image"].astype(str)
    distance_plot["image"] = distance_plot["image"].astype(str)

    # ----------------------------
    # Optional distance population filter
    # ----------------------------
    if distance_population is not None:
        if "feature_population" not in distance_plot.columns:
            raise ValueError(
                "distance_population was provided, but distance_summary "
                "does not contain a 'feature_population' column."
            )

        distance_plot = distance_plot[
            distance_plot["feature_population"] == distance_population
        ].copy()

    # ----------------------------
    # Prepare feature table
    # ----------------------------
    if statistic == "mean":
        feature_plot = feature_plot[
            ["image", diameter_col, diameter_std_col]
        ].rename(
            columns={
                diameter_col: "diameter_value_nm",
                diameter_std_col: "diameter_std_nm"
            }
        )
    else:
        feature_plot = feature_plot[
            ["image", diameter_col]
        ].rename(
            columns={
                diameter_col: "diameter_value_nm"
            }
        )
        feature_plot["diameter_std_nm"] = 0.0

    # ----------------------------
    # Prepare distance table
    # If multiple distance rows remain per image, aggregate them.
    # ----------------------------
    if statistic == "mean":
        distance_plot = (
            distance_plot
            .groupby("image", as_index=False)
            .agg(
                distance_value_nm=(distance_col, "mean"),
                distance_std_nm=(distance_std_col, "mean")
            )
        )
    else:
        distance_plot = (
            distance_plot
            .groupby("image", as_index=False)
            .agg(
                distance_value_nm=(distance_col, "mean")
            )
        )
        distance_plot["distance_std_nm"] = 0.0

    # ----------------------------
    # Merge
    # ----------------------------
    plot_df = pd.merge(
        feature_plot,
        distance_plot,
        on="image",
        how="inner"
    )

    if plot_df.empty:
        print("No overlapping samples between feature_summary and distance_summary.")
        return

    existing_order = [
        name for name in desired_order
        if name in plot_df["image"].values
    ]

    unmatched = [
        name for name in plot_df["image"].values
        if name not in desired_order
    ]

    final_order = existing_order + unmatched

    plot_df["image"] = pd.Categorical(
        plot_df["image"],
        categories=final_order,
        ordered=True
    )

    plot_df = plot_df.sort_values("image").copy()
    plot_df["image"] = plot_df["image"].astype(str)

    # ----------------------------
    # Plot
    # ----------------------------
    x = np.arange(len(plot_df))
    width = 0.36

    bar_colors = [
        sample_style.get(str(name), {"color": "gray"})["color"]
        for name in plot_df["image"]
    ]

    bar_hatches = [
        sample_style.get(str(name), {"hatch": None})["hatch"]
        for name in plot_df["image"]
    ]

    fig, ax = plt.subplots(figsize=(12, 5.8))

    # Error bars only for mean plot
    if statistic == "mean":
        diameter_yerr = plot_df["diameter_std_nm"].fillna(0).values
        distance_yerr = plot_df["distance_std_nm"].fillna(0).values
    else:
        diameter_yerr = None
        distance_yerr = None

    bars_diameter = ax.bar(
        x - width / 2,
        plot_df["diameter_value_nm"],
        width,
        yerr=diameter_yerr,
        capsize=5 if statistic == "mean" else 0,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        label=diameter_label,
        error_kw={
            "ecolor": errorbar_color,
            "elinewidth": 1.6,
            "capthick": 1.6
        } if statistic == "mean" else None
    )

    bars_distance = ax.bar(
        x + width / 2,
        plot_df["distance_value_nm"],
        width,
        yerr=distance_yerr,
        capsize=5 if statistic == "mean" else 0,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        hatch="///",
        label=distance_label,
        error_kw={
            "ecolor": errorbar_color,
            "elinewidth": 1.6,
            "capthick": 1.6
        } if statistic == "mean" else None
    )

    # Preserve sample-specific hatch on feature-diameter bars
    for bar, hatch in zip(bars_diameter, bar_hatches):
        if hatch is not None:
            bar.set_hatch(hatch)

    # Distance bars always get additional hatch
    for bar, hatch in zip(bars_distance, bar_hatches):
        if hatch is not None:
            bar.set_hatch(hatch + "///")
        else:
            bar.set_hatch("///")

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["image"].astype(str),
        rotation=45,
        ha="right"
    )

    ax.set_ylabel("Length scale (nm)")

    if statistic == "mean":
        ax.set_title(title + " | Mean ± SD")
    else:
        ax.set_title(title + " | Median")

    ax.legend(frameon=False)

    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def plot_nearest_centroid_distance_histograms(
    df_distances,
    output_path=None,
    bins=20,
    distance_col="nearest_centroid_distance_nm",
    group_col="image",
    population_col="feature_population",
    population_filter=None,
    title="Nearest-Neighbor Centroid Distance Distribution"
):
    """
    Plot histogram of nearest-neighbor centroid distances for each sample.

    Parameters
    ----------
    df_distances:
        Output from compute_nearest_neighbor_distances(...)

    bins:
        Number of histogram bins.

    distance_col:
        Column to plot. Default: nearest_centroid_distance_nm

    population_filter:
        Optional. Example: "darkest"
        Use this when df_distances contains multiple feature populations.
    """

    if df_distances is None or df_distances.empty:
        print("No distance data to plot.")
        return

    if distance_col not in df_distances.columns:
        raise ValueError(f"{distance_col} not found in df_distances.")

    df_plot = df_distances.copy()

    if population_filter is not None:
        if population_col not in df_plot.columns:
            raise ValueError(
                f"population_filter was provided, but {population_col} "
                "is not in df_distances."
            )

        df_plot = df_plot[
            df_plot[population_col] == population_filter
        ].copy()

    if df_plot.empty:
        print("No distance data left after population filtering.")
        return

    sample_style = {
        "Backbone random": {"color": "lightgray", "hatch": None},
        "Sidechain random 1.5": {"color": "skyblue", "hatch": "///"},
        "Sidechain random 1.8": {"color": "skyblue", "hatch": None},
        "Sidechain block": {"color": "red", "hatch": None},
        "Backbone block": {"color": "navajowhite", "hatch": None},
        "Nafion 212": {"color": "black", "hatch": None},
    }

    desired_order = list(sample_style.keys())

    samples = [
        name for name in desired_order
        if name in df_plot[group_col].astype(str).values
    ]

    unmatched = [
        name for name in df_plot[group_col].astype(str).unique()
        if name not in desired_order
    ]

    samples = samples + unmatched

    n_samples = len(samples)

    if n_samples == 0:
        print("No samples found.")
        return

    # Make one row of histograms
    fig, axes = plt.subplots(
        1,
        n_samples,
        figsize=(4.2 * n_samples, 4.2),
        sharey=True
    )

    if n_samples == 1:
        axes = [axes]

    # Use common bin edges across samples for fair comparison
    all_values = df_plot[distance_col].dropna().values

    bin_edges = np.linspace(
        np.nanmin(all_values),
        np.nanmax(all_values),
        bins + 1
    )

    for ax, sample in zip(axes, samples):
        df_sample = df_plot[
            df_plot[group_col].astype(str) == sample
        ].copy()

        values = df_sample[distance_col].dropna().values

        color = sample_style.get(
            sample,
            {"color": "gray"}
        )["color"]

        hatch = sample_style.get(
            sample,
            {"hatch": None}
        )["hatch"]

        counts, edges, patches = ax.hist(
            values,
            bins=bin_edges,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85
        )

        if hatch is not None:
            for patch in patches:
                patch.set_hatch(hatch)

        mean_value = np.mean(values)
        median_value = np.median(values)
        std_value = np.std(values, ddof=1)

        ax.axvline(
            mean_value,
            linestyle="--",
            linewidth=1.4,
            color="black",
            label=f"Mean = {mean_value:.2f} nm"
        )

        ax.axvline(
            median_value,
            linestyle=":",
            linewidth=1.6,
            color="black",
            label=f"Median = {median_value:.2f} nm"
        )

        ax.set_title(
            f"{sample}\n"
            f"n={len(values)}, SD={std_value:.2f} nm",
            fontsize=10
        )

        ax.set_xlabel("Nearest centroid distance (nm)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("Feature count")

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=1.05
    )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()