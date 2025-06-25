"""Utilities for eucal notebooks."""

import os
from datetime import datetime
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rasterio
import requests  # type: ignore
import torch
import xarray as xr
from dateutil.parser import parse as parse_datetime  # type: ignore
from matplotlib import patches
from matplotlib.ticker import MultipleLocator
from pyproj import Transformer
from rasterio import features
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy import ndimage as ndi
from shapely.geometry import MultiPolygon, Polygon
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing
from skimage.segmentation import watershed

# Factor to scale EMIT radiance values to model's expected range
EMIT_SCALING_FACTOR = 0.1
PREDICTION_NAMES = ["marginal", "likelihood", "conditional"]
WIND_SOURCE_BASE_URL = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{y}/M{m}/D{d}/GEOS.fp.asm.inst3_2d_asm_Nx.{ymd}_{h}00.V01.nc4"
WIND_ERROR = 0.5


class Colorbar(str, Enum):
    """How the colorbar should be set."""

    SHARE = "share"
    INDIVIDUAL = "individual"
    EXTENT = "extent"


#######################################
###### MODEL INFERENCE FUNCTIONS ######
#######################################


def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> xr.DataArray:
    """Run inference for a given model and scene.

    Args:
        model: The model to be used for predictions.
        x: The scene data, prepared for use with `model`.

    Returns:
        The predictions as an `xarray.DataArray`, containing "marginal", "likelihood", and "conditional"
        entries along the "prediction" dimension.
    """
    model.eval()
    with torch.no_grad():
        raw_predictions = model(x)

    n_samples, _, n_rows, n_cols = raw_predictions.shape

    likelihood_logit = raw_predictions[:, 0:1]
    conditional_pred = raw_predictions[:, 1:2]

    sigmoid = torch.nn.Sigmoid()  # inverse logit to get probabilities between 0 and 1
    likelihood_probability = sigmoid(likelihood_logit)
    marginal_pred = likelihood_probability * conditional_pred

    predictions = xr.DataArray(
        np.concatenate([marginal_pred, likelihood_probability, conditional_pred], axis=1),
        dims=["sample", "prediction", "y", "x"],
        coords={
            "sample": np.arange(n_samples),
            "prediction": PREDICTION_NAMES,
            "y": np.arange(n_rows),
            "x": np.arange(n_cols),
        },
    )

    # if we are not running in batch mode, squeeze this array to drop the "samples" dimension
    predictions = predictions.squeeze()
    return predictions


#######################################
##### SENTINEL-2 IMAGE PROCESSING #####
#######################################


def get_s2_rgb_bands(x_tensor: np.ndarray, bands: list[str]) -> np.ndarray:
    """Extract and stack RGB bands from input tensor for Sentinel-2.

    Args:
        x_tensor: Input tensor containing band data with shape (bands, height, width).
        bands: List of band names of x_tensor.

    Returns:
        RGB image array with shape (height, width, 3).
    """
    rgb_indices = [bands.index(band) for band in ("B04", "B03", "B02")]
    rgb_image = np.transpose(x_tensor[rgb_indices], (1, 2, 0)) / 10000  # Normalize
    return rgb_image


def get_s2_band_ratio(x_tensor: np.ndarray, bands: list[str]) -> np.ndarray:
    """Calculate B12/B11 band ratio for Sentinel-2.

    Args:
        x_tensor: Input tensor containing band data with shape (bands, height, width).
        bands: List of band names of x_tensor.

    Returns:
        Band ratio array with shape (height, width).
    """
    b11 = x_tensor[bands.index("B11")]
    b12 = x_tensor[bands.index("B12")]
    return b12 / b11


#######################################
###### LANDSAT IMAGE PROCESSING #######
#######################################


def get_landsat_rgb_bands(x_tensor: np.ndarray, bands: list[str]) -> np.ndarray:
    """Extract and stack RGB bands from input tensor for Landsat.

    Args:
        x_tensor: Input tensor containing band data with shape (bands, height, width).
        bands: List of band names of x_tensor.

    Returns
    -------
        RGB image array with shape (height, width, 3).
    """
    rgb_indices = [bands.index(band) for band in ("red", "green", "blue")]
    rgb_image = np.transpose(x_tensor[rgb_indices], (1, 2, 0)) / 10000  # Normalize
    return rgb_image


def get_landsat_band_ratio(x_tensor: np.ndarray, bands: list[str]) -> np.ndarray:
    """Calculate swir22/swir16 band ratio for Landsat.

    Args:
        x_tensor: Input tensor containing band data with shape (bands, height, width).
        bands: List of band names of x_tensor.

    Returns
    -------
        Band ratio array with shape (height, width).
    """
    swir16 = x_tensor[bands.index("swir16")]
    swir22 = x_tensor[bands.index("swir22")]
    return swir22 / swir16


#######################################
######### VISUALIZATION FUNCTIONS #####
#######################################


def plot_detected_plumes(
    marginal_retrieval: np.ndarray,
    plumes: list,
) -> None:
    """
    Plot a marginal retrieval map with overlayed detected plume bounding boxes and annotations.

    Parameters:
    - marginal_retrieval (np.ndarray): 2D array of retrieval values.
    - plumes (list of dict): Each dict should contain 'bbox', 'Q', 'likelihood_score', and 'source_location'.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # Show marginal retrieval background
    im = ax.imshow(marginal_retrieval, vmax=1.0, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Marginal Retrieval (mol/m²)")

    # Overlay bounding boxes and annotations
    for plume in plumes:
        min_row, min_col, max_row, max_col = plume["bbox"]
        Q = plume["Q"]
        L = plume["likelihood_score"]

        # Draw bounding box
        rect = patches.Rectangle(
            (min_col, min_row), max_col - min_col, max_row - min_row, linewidth=1.5, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        # Annotate with Q and likelihood
        ax.text(
            max_col,
            max_row + 1,
            f"Q={Q:.1f} kg/hr\nL={L * 100:.0f}%",
            color="white",
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

    ax.set_title("Detected Methane Plumes with Estimated Emission Rates")
    plt.tight_layout()
    plt.show()


def plot_segmentation_results(
    binary_probability: np.ndarray, mask: np.ndarray, regions: list, subplot_props: dict
) -> None:
    """Plot segmentation outputs: binary probability, mask, and regions with bounding boxes."""

    fig_scaling = 4
    n_rows = 1
    n_cols = len(subplot_props)

    plt.figure(figsize=(n_cols * fig_scaling, n_rows * fig_scaling), layout="constrained")

    for i, (key, props) in enumerate(subplot_props.items(), start=1):
        plt.subplot(n_rows, n_cols, i)

        if key == "binary_probability":
            data = binary_probability
        elif key == "watershed_mask":
            data = mask
        elif key == "regions":
            data = binary_probability  # used again to show regions + overlay

        plt.imshow(data, **props.get("imshow_kwargs", {}))
        plt.colorbar()
        plt.title(props["title"])
        plt.xticks(np.arange(32, data.shape[1], 32))
        plt.yticks(np.arange(32, data.shape[0], 32))
        plt.grid()

        # Overlay bounding boxes if this is the 'regions' plot
        if key == "regions":
            ax = plt.gca()
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                width = max_col - min_col
                height = max_row - min_row

                rect = patches.Rectangle((min_col, min_row), width, height, fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(rect)

                ax.text(
                    min_col,
                    min_row,
                    f"Region {region.label}",
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.5),
                )

    plt.show()


def plot_predictions(yhat: xr.DataArray, subplot_props: dict, *, units: str) -> None:
    """Plot prediction results."""

    fig_scaling = 4
    n_rows = 1
    n_cols = len(subplot_props)

    plt.figure(figsize=(n_cols * fig_scaling, n_rows * fig_scaling), layout="constrained")

    for i, (prediction_type, props) in enumerate(subplot_props.items(), start=1):
        data = yhat.sel(prediction=prediction_type)
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(data, **props.get("imshow_kwargs", {}))
        plt.colorbar()
        plt.title(props["title"].format(units=units))
        plt.xticks(np.arange(32, data.x.size, 32))
        plt.yticks(np.arange(32, data.y.size, 32))
        plt.grid()


def grid16(ax: matplotlib.axes.Axes | None = None) -> None:
    """Apply a grid with 16x16 blocks to the current active plot."""
    if ax is None:
        ax = plt.gca()

    ax.xaxis.set_major_locator(MultipleLocator(32))
    ax.xaxis.set_minor_locator(MultipleLocator(16))

    ax.yaxis.set_major_locator(MultipleLocator(32))
    ax.yaxis.set_minor_locator(MultipleLocator(16))

    ax.grid(True, which="both")


def plot_rgb_timeseries(
    rgb_main: np.ndarray,
    rgb_ref1: np.ndarray,
    rgb_ref2: np.ndarray,
    date_main: str,
    date_ref1: str,
    date_ref2: str,
) -> None:
    """Plot RGB timeseries for main and reference images.

    Args:
        rgb_main: Main RGB image array
        rgb_ref1: First reference RGB image array
        rgb_ref2: Second reference RGB image array
        date_main: Date string for main image
        date_ref1: Date string for first reference image
        date_ref2: Date string for second reference image
    """
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(
        f"RGB Timeseries - Main Date: {date_main}",
        fontsize=24,
        y=0.97,
        x=0.5,
    )

    # First reference image
    plt.subplot(1, 3, 1)
    plt.title(
        f"""RGB (t=ref1 {date_ref1})
            Min {rgb_ref1.min():.3f}, Max {rgb_ref1.max():.3f}, Mean {rgb_ref1.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_ref1 / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    plt.grid(True)

    # Second reference image
    plt.subplot(1, 3, 2)
    plt.title(
        f"""RGB (t=ref2 {date_ref2})
            Min {rgb_ref2.min():.3f}, Max {rgb_ref2.max():.3f}, Mean {rgb_ref2.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_ref2 / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    plt.grid(True)

    # Main image
    plt.subplot(1, 3, 3)
    plt.title(
        f"""RGB (t=main {date_main})
            Min {rgb_main.min():.3f}, Max {rgb_main.max():.3f}, Mean {rgb_main.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(
        (rgb_main / 0.35 * 255).clip(0, 255).astype(np.uint8),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_band_ratio_timeseries(
    ratio_main: np.ndarray,
    ratio_ref1: np.ndarray,
    ratio_ref2: np.ndarray,
    date_main: str,
    date_ref1: str,
    date_ref2: str,
    satellite: str,
    ratio_colorbar: Colorbar | tuple[float, float] = Colorbar.SHARE,
) -> None:
    """Plot band ratio timeseries for main and reference images.

    Args:
        ratio_main: Main band ratio array
        ratio_ref1: First reference band ratio array
        ratio_ref2: Second reference band ratio array
        date_main: Date string for main image
        date_ref1: Date string for first reference image
        date_ref2: Date string for second reference image
        ratio_colorbar: Colorbar configuration (SHARE, INDIVIDUAL, or tuple of min/max values)
    """
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(
        f"Band Ratio Timeseries - Main Date: {date_main}",
        fontsize=24,
        y=0.97,
        x=0.5,
    )

    band_ratio_name = "swir22/swir16" if satellite == "landsat" else "B12/B11"

    # Determine colorbar ranges
    match ratio_colorbar:
        case Colorbar.SHARE:
            # Use the min/max of all images for consistent colors
            vmin = min(ratio_ref1.min(), ratio_ref2.min(), ratio_main.min())
            vmax = max(ratio_ref1.max(), ratio_ref2.max(), ratio_main.max())
            vmin_ref1 = vmin_ref2 = vmin_main = vmin
            vmax_ref1 = vmax_ref2 = vmax_main = vmax

        case Colorbar.INDIVIDUAL:
            vmin_ref1, vmax_ref1 = ratio_ref1.min(), ratio_ref1.max()
            vmin_ref2, vmax_ref2 = ratio_ref2.min(), ratio_ref2.max()
            vmin_main, vmax_main = ratio_main.min(), ratio_main.max()

        case (float(), float()):
            vmin, vmax = ratio_colorbar
            if vmin >= vmax:
                raise ValueError("Colorbar min must be less than max")
            vmin_ref1 = vmin_ref2 = vmin_main = vmin
            vmax_ref1 = vmax_ref2 = vmax_main = vmax

        case _:
            raise ValueError("ratio_colorbar must be Colorbar enum or tuple of (min, max)")

    # Plot first reference ratio
    plt.subplot(1, 3, 1)
    plt.title(
        f"""{band_ratio_name} Ratio (t=ref1 {date_ref1})
            Min {ratio_ref1.min():.3f}, Max {ratio_ref1.max():.3f}, Mean {ratio_ref1.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_ref1, interpolation="nearest", vmin=vmin_ref1, vmax=vmax_ref1)
    plt.grid(True)
    plt.colorbar()

    # Plot second reference ratio
    plt.subplot(1, 3, 2)
    plt.title(
        f"""{band_ratio_name} Ratio (t=ref2 {date_ref2})
            Min {ratio_ref2.min():.3f}, Max {ratio_ref2.max():.3f}, Mean {ratio_ref2.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_ref2, interpolation="nearest", vmin=vmin_ref2, vmax=vmax_ref2)
    plt.grid(True)
    plt.colorbar()

    # Plot main ratio
    plt.subplot(1, 3, 3)
    plt.title(
        f"""{band_ratio_name} Ratio (t=main {date_main})
            Min {ratio_main.min():.3f}, Max {ratio_main.max():.3f}, Mean {ratio_main.mean():.3f}""",
        fontsize=15,
    )
    plt.imshow(ratio_main, interpolation="nearest", vmin=vmin_main, vmax=vmax_main)
    plt.grid(True)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


#######################################
######### WIND DATA PROCESSING ########
#######################################


def generate_geos_wind_data_url(sensing_time: datetime) -> str:
    """Generate URL for GEOS-FP wind data.

    File format description: https://gmao.gsfc.nasa.gov/pubs/docs/Lucchesi1203.pdf

    Args:
        sensing_time: The time for which to generate the URL.

    Returns:
        URL string for the GEOS-FP data file.
    """
    target_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    closest_hour = min(target_hours, key=lambda x: abs(sensing_time.hour - x))
    hour = f"{closest_hour:02d}"

    formatted_date = sensing_time.strftime("%Y%m%d")
    formatted_url = WIND_SOURCE_BASE_URL.format(
        y=sensing_time.year,
        m=sensing_time.strftime("%m"),
        d=sensing_time.strftime("%d"),
        h=hour,
        ymd=formatted_date,
    )
    return formatted_url


def download_wind_data_from_GEOS_FP(sensing_time_str: str) -> xr.Dataset | None:
    """Download wind data from GEOS_FP for a given date.

    Args:
        sensing_time_str: String representation of the sensing time.

    Returns:
        xarray Dataset containing wind data, or None if download fails.
    """
    sensing_time = parse_datetime(sensing_time_str)
    url = generate_geos_wind_data_url(sensing_time)

    print(f"{url=}")
    response = requests.get(url)

    tmp_file = "/tmp/geos_fp_data.nc4"
    try:
        with open(tmp_file, "wb") as f:
            f.write(response.content)

        ds = xr.open_dataset(tmp_file)
        os.remove(tmp_file)
        return ds

    except Exception as e:
        print(f"Failed to download wind data. Reason: {e}")
        return None


def get_wind_vectors_from_wind_tile(wind_ds: xr.Dataset, plume_lat: float, plume_lon: float) -> tuple[float, float]:
    """Extract the U and V components from the wind dataset for the specific lat and lon.

    Args:
        wind_ds: Wind dataset from GEOS-FP.
        plume_lat: Latitude of the plume.
        plume_lon: Longitude of the plume.

    Returns:
        Tuple of (u_wind_component, v_wind_component).
    """
    u_wind_component = wind_ds["U10M"].sel(lat=plume_lat, lon=plume_lon, method="nearest").values.item()
    v_wind_component = wind_ds["V10M"].sel(lat=plume_lat, lon=plume_lon, method="nearest").values.item()

    return u_wind_component, v_wind_component


def get_wind_speed_for_plume(
    wind_ds: xr.Dataset,
    plume_source_lat: float,
    plume_source_lon: float,
) -> tuple[float, float, float, float, float]:
    """Get the wind vector for the plume source location and calculate the effective wind speeds.

    Args:
        wind_ds: Wind dataset from GEOS-FP.
        plume_source_lat: Latitude of the plume source.
        plume_source_lon: Longitude of the plume source.

    Returns:
        Tuple of (u_eff, u_eff_high, u_eff_low, wind_direction, u_wind_component).
    """
    u_wind_component, v_wind_component = get_wind_vectors_from_wind_tile(wind_ds, plume_source_lat, plume_source_lon)

    wind_speed = calc_effective_wind_speed(u_wind_component, v_wind_component)
    wind_direction = calc_wind_direction(u_wind_component, v_wind_component)

    # Calculate the wind error margin
    wind_low, wind_high = calc_wind_error(wind_speed, wind_error=WIND_ERROR)

    u_eff, u_eff_high, u_eff_low = calc_u_eff(wind_speed, wind_low, wind_high)

    return u_eff, u_eff_high, u_eff_low, wind_direction, u_wind_component


def calc_effective_wind_speed(u10m: float, v10m: float) -> float:
    """Compute the total wind speed in m/s from the zonal (u) and meridional (v) components at 10 m height.

    Calculated as the magnitude of the 2D wind vector: sqrt(u10m^2 + v10m^2).

    Args:
        u10m: Zonal wind component at 10 m, in m/s. Positive values indicate eastward flow.
        v10m: Meridional wind component at 10 m, in m/s. Positive values indicate northward flow.

    Returns:
        The wind speed at 10 m height, in m/s.
    """
    return np.sqrt(np.square(u10m) + np.square(v10m))


def calc_u_eff(wind_speed: float, wind_low: float, wind_high: float) -> tuple[float, float, float]:
    """Calculate effective wind speed and its uncertainty bounds using Varon et al. (2021) fitted parameters.

    The effective wind speed, u_eff, is modeled as:
        u_eff = alpha * wind_speed + beta
    where alpha=0.33 and beta=0.45 are empirically derived for Sentinel 2 based on LES Simulations.
    A minimum effective wind speed of 0.01 m/s is enforced.

    Args:
        wind_speed: Central estimate of wind speed in m/s.
        wind_low: Lower bound of wind speed in m/s.
        wind_high: Upper bound of wind speed in m/s.

    Returns:
        Tuple of (effective wind speed, upper bound, lower bound) in m/s.

    References:
        Varon 2021: https://amt.copernicus.org/articles/14/2771/2021/
    """
    # Model parameters from Varon 2021
    alpha, beta = 1.0, 0.0

    # Calculate effective wind speeds
    u_eff = max(alpha * wind_speed + beta, 0.01)
    u_eff_low = max(alpha * wind_low + beta, 0.009)
    u_eff_high = max(alpha * wind_high + beta, 0.011)

    return u_eff, u_eff_high, u_eff_low


def calc_wind_direction(u10m: float, v10m: float) -> float:
    """Compute wind direction in degrees from the zonal (u) and meridional (v) wind components at 10 m height.

    Args:
        u10m: Zonal wind component at 10 m, in m/s.
        v10m: Meridional wind component at 10 m, in m/s.

    Returns:
        Wind direction in degrees, measured from north.
    """
    return float(np.degrees(np.arctan2(u10m, v10m)))


def calc_wind_error(wind_speed: float, wind_error: float) -> tuple[float, float]:
    """Calculate lower and upper bounds of the wind speed given a fractional error.

    Args:
        wind_speed: Central estimate of wind speed in m/s.
        wind_error: Fractional error (e.g., 0.1 for ±10% error).

    Returns:
        Tuple of (wind_low, wind_high) in m/s.
    """
    wind_low: float = wind_speed * (1 - wind_error)
    wind_high: float = wind_speed * (1 + wind_error)
    return wind_low, wind_high


#######################################
######### PLUME MASKING ##############
#######################################


def retrieval_mask_using_watershed_algo(
    retrieval: np.ndarray,
    marker_distance: int,
    marker_threshold: float,
    watershed_floor_threshold: float,
    closing_footprint_size: int,
):
    """
    Apply watershed algorithm to a retrieval given marker coordinates and watershed parameters.

    This function is abstracted by the singledispatch watershed functions, which identify the marker coordinates
    and, in the case of quantile-based thresholding, the floor threshold.
    """
    # This function locates local maxima in the source array, which serve as starting points
    # (markers) for the watershed segmentation algorithm. For CV, these markers represent areas with
    # high probability of methane presence.
    marker_coords = peak_local_max(retrieval, min_distance=marker_distance, threshold_abs=marker_threshold)

    # Create a binary mask of the markers
    markers = np.zeros(retrieval.shape, dtype=bool)
    markers[tuple(marker_coords.T)] = True
    markers, _ = ndi.label(markers)

    segmentation_mask = np.logical_and(retrieval > watershed_floor_threshold, retrieval > 0.0)

    mask = watershed(
        -retrieval,
        markers,
        mask=segmentation_mask,
    )
    # Watershed returns integer labels for segmented plume masks whereas we want a binary plume mask
    # Conver these integers to 1s (numpy where has arguments: condition, value if true, value if false)
    mask = np.where(mask != 0, 1, 0).astype(bool)

    if closing_footprint_size > 0:
        closing_footprint = np.ones((closing_footprint_size, closing_footprint_size), dtype=bool)
        mask = binary_closing(mask, footprint=closing_footprint)

    return mask


def mask_and_segment_plumes(
    binary_probability: np.ndarray,
    marker_distance: int = 1,
    marker_threshold: float = 0.1,
    watershed_floor_threshold: float = 0.075,
    closing_footprint_size: int = 0,
) -> tuple[np.ndarray, list[regionprops]]:
    """
    Mask the retrieval array using watershed algorithm and segment into individual plumes.

    Returns:
        tuple: (mask, list of region properties for each plume)
    """
    mask = retrieval_mask_using_watershed_algo(
        binary_probability,
        marker_distance,
        marker_threshold,
        watershed_floor_threshold,
        closing_footprint_size,
    )

    labeled_mask = label(mask, connectivity=2)
    regions = list(regionprops(labeled_mask))
    return mask, regions


#######################################
######### PLUME QUANTIFICATION# #######
#######################################


def get_plume_source_location_v2(
    plume_retrieval: np.ndarray, source_crs: CRS, source_transform: Affine
) -> tuple[float, float]:
    """Calculate the latitude and longitude coordinates of the maximum methane retrieval pixel.

    This function:
    1. Identifies the pixel with the maximum value in `plume_retrieval`.
    2. Converts that pixel's row/column indices to real-world coordinates in `source_crs`.
    3. Transforms those coordinates into geographic coordinates (EPSG:4326).

    Args:
        plume_retrieval: A 2D array of methane retrieval or enhancement data.
        source_crs: The coordinate reference system of the `plume_retrieval`.
        source_transform: The affine transform for converting array indices to real-world coordinates.

    Returns:
        Tuple of (latitude, longitude).
    """
    assert len(plume_retrieval.shape) == 2, "Plume retrieval must have 2 dimensions for source location calculation"  # noqa: PLR2004 (magic-number-comparison)

    # Calculate the grid coordinates of the maximum retrieval pixel
    source_index = np.unravel_index(plume_retrieval.argmax(), plume_retrieval.shape)

    # Convert the grid (array) coordinates to real world coordinates in the source CRS
    transformer = rasterio.transform.AffineTransformer(source_transform)
    source_x_coord, source_y_coord = transformer.xy(source_index[0], source_index[1])

    # Transform real world coordinates in source CRS to latitude and longitude in CRS: 4326
    coordinate_transformer = Transformer.from_crs(source_crs.to_epsg(), 4326)
    latitude, longitude = coordinate_transformer.transform(source_x_coord, source_y_coord)

    return latitude, longitude


def calc_Q_IME(u_eff: float, L: float, IME: float) -> float:
    """Estimate the methane emission rate (Q) from effective wind speed (u_eff), effective plume size (L), and IME.

    This function implements:
    Q = (u_eff / L) x IME x 57.75286,
    where 57.75286 is the conversion factor from mol/s to kg/hr for CH₄.

    Args:
        u_eff: Effective wind speed in m/s.
        L: Effective plume size in meters.
        IME: Integrated Methane Enhancement in moles.

    Returns:
        Methane emission rate, in kg/hr.
    """
    return (u_eff / L) * IME * 57.75286


def calculate_major_axis_quantification(
    plume_mask: npt.NDArray,
    retrieval: npt.NDArray,
    pixel_width: float,
    wind_speed: float,
) -> tuple[float, float, float]:
    """Calculate methane quantification using the major axis length method.

    Args:
        plume_mask: Binary mask of the plume.
        retrieval: Conditional or marginal retrieval values (mol/m2).
        pixel_width: Width of a pixel in meters.
        wind_speed: Wind speed in m/s.

    Returns:
        Tuple of (major_axis_length, IME, quantification).
    """
    pixel_area = pixel_width**2
    # Assert that plume_mask is a binary mask (contains only 0s and 1s)
    assert np.array_equal(plume_mask, plume_mask.astype(bool)), (
        "plume_mask must be a binary mask containing only 0s and 1s"
    )

    # Calculate IME
    IME = retrieval[plume_mask].sum() * pixel_area

    # Get plume length using major axis
    contours = features.shapes(plume_mask.astype(np.uint8), mask=plume_mask)
    polygons = [Polygon([(j[0], j[1]) for j in i[0]["coordinates"][0]]) for i in contours]

    plume_polygon = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]

    min_rect = plume_polygon.minimum_rotated_rectangle
    coords = np.array(min_rect.exterior.coords)
    sides = np.array([np.linalg.norm(coords[i] - coords[i - 1]) for i in range(1, len(coords))])
    major_axis_length = float(np.max(sides) * pixel_width)

    # Calculate quantification
    major_axis_quantification = calc_Q_IME(wind_speed, major_axis_length, IME)

    return major_axis_length, IME, major_axis_quantification
