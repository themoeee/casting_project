"""Utilities for reading and analyzing Buehler XML files.

Typical notebook usage::

    from useful_py_function import load_xml_data, show_xml_contents, plot_casting_overview

    xml_data = load_xml_data("data/measurement.xml.gz")
    show_xml_contents(xml_data)
    plot_casting_overview(xml_data)

The module supports both uncompressed ``.xml`` files and compressed
``.xml.gz`` files. Importing the module does not automatically load a file or
open a plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
from os import PathLike
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PathType = str | PathLike[str]
SUMMARY_COLUMNS = [
    "short_name",
    "long_name",
    "unit",
    "samples",
    "duration_s",
    "min",
    "max",
    "mean",
]


@dataclass
class XMLCurve:
    """A single measurement curve stored in the XML file.

    ``data`` contains at least the columns ``time_us``, ``time_s``, and
    ``value``. ``time_s`` starts at zero for each curve. Additional XML sample
    attributes, such as ``CAdtIdx``, are preserved as well.
    """

    short_name: str
    long_name: str
    unit: str
    data: pd.DataFrame
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Return a readable label for tables and plots."""

        names = self.long_name or self.short_name or "Unnamed curve"
        if self.short_name and self.short_name != self.long_name:
            names = f"{names} ({self.short_name})"
        return f"{names} {self.unit}".strip()

    @property
    def sample_count(self) -> int:
        return len(self.data)


@dataclass
class XMLData:
    """Contents of an XML file: file metadata and all measurement curves."""

    file_path: Path
    metadata: dict[str, str]
    curves: list[XMLCurve]


def _local_name(tag: str) -> str:
    """Remove an optional XML namespace from a tag name."""

    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str, default: str = "") -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return (child.text or "").strip()
    return default


def _open_xml(file_path: PathType) -> tuple[Path, ET.Element]:
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"XML file not found: {path}")

    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as stream:
                root = ET.parse(stream).getroot()
        else:
            root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"'{path}' could not be read as XML/XML.GZ.") from error

    return path.resolve(), root


def _extract_file_metadata(root: ET.Element) -> dict[str, str]:
    metadata = {f"xml_{key}": value for key, value in root.attrib.items()}

    def visit(element: ET.Element) -> None:
        # Curves and samples are extracted separately and do not belong in the
        # general file metadata.
        if _local_name(element.tag) in {"curveObject", "aSample"}:
            return
        children = list(element)
        text = (element.text or "").strip()
        if not children and text:
            key = _local_name(element.tag)
            if key in metadata:
                suffix = 2
                while f"{key}_{suffix}" in metadata:
                    suffix += 1
                key = f"{key}_{suffix}"
            metadata[key] = text
        for child in children:
            visit(child)

    visit(root)
    return metadata


def _extract_curve(curve_element: ET.Element) -> XMLCurve:
    curve_metadata: dict[str, str] = {}
    for child in curve_element:
        if _local_name(child.tag) == "samples":
            continue
        if not list(child):
            curve_metadata[_local_name(child.tag)] = (child.text or "").strip()

    records = [
        dict(sample.attrib)
        for sample in curve_element.iter()
        if _local_name(sample.tag) == "aSample"
    ]
    frame = pd.DataFrame.from_records(records)
    frame.rename(columns={"timeUs": "time_us", "CAy": "value"}, inplace=True)

    if frame.empty:
        frame = pd.DataFrame(columns=["time_us", "time_s", "value"])
    else:
        if "time_us" not in frame or "value" not in frame:
            raise ValueError("A curveObject curve has no timeUs or CAy attribute.")
        frame["time_us"] = pd.to_numeric(frame["time_us"], errors="raise").astype("int64")
        frame["value"] = pd.to_numeric(frame["value"], errors="raise")
        for column in frame.columns.difference(["time_us", "value"]):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().all():
                frame[column] = numeric
        frame.insert(1, "time_s", (frame["time_us"] - frame["time_us"].iloc[0]) / 1e6)

    return XMLCurve(
        short_name=curve_metadata.get("shortText", ""),
        long_name=curve_metadata.get("longText", ""),
        unit=curve_metadata.get("unitText", ""),
        data=frame,
        metadata=curve_metadata,
    )


def load_xml_data(file_path: PathType) -> XMLData:
    """Read a complete Buehler XML or XML.GZ file.

    Returns:
        An :class:`XMLData` object. The measurements for each curve are stored
        as a pandas DataFrame in ``result.curves[i].data``.
    """

    path, root = _open_xml(file_path)
    curve_elements = [
        element for element in root.iter() if _local_name(element.tag) == "curveObject"
    ]
    curves = [_extract_curve(element) for element in curve_elements]
    return XMLData(path, _extract_file_metadata(root), curves)


def select_xml_curves(
    xml_data: XMLData,
    pattern: str | None = None,
    names: str | Iterable[str] | None = None,
) -> list[XMLCurve]:
    """Select curves using a search string or exact names.

    ``pattern`` performs a case-insensitive search in the short name, long
    name, and unit. ``names`` accepts short or long names. Both filters can be
    combined.
    """

    if names is None:
        requested_names = None
    elif isinstance(names, str):
        requested_names = {names.casefold()}
    else:
        requested_names = {name.casefold() for name in names}

    search = pattern.casefold() if pattern is not None else None
    selected: list[XMLCurve] = []
    for curve in xml_data.curves:
        searchable = " ".join((curve.short_name, curve.long_name, curve.unit)).casefold()
        name_matches = requested_names is None or bool(
            {curve.short_name.casefold(), curve.long_name.casefold()} & requested_names
        )
        if name_matches and (search is None or search in searchable):
            selected.append(curve)
    return selected


def get_xml_curve(xml_data: XMLData, name: str) -> XMLCurve:
    """Return exactly one curve by its short or long name."""

    matches = select_xml_curves(xml_data, names=name)
    if not matches:
        available = ", ".join(curve.short_name for curve in xml_data.curves)
        raise KeyError(f"Curve '{name}' not found. Available curves: {available}")
    if len(matches) > 1:
        labels = ", ".join(curve.label for curve in matches)
        raise ValueError(f"The name '{name}' is ambiguous: {labels}")
    return matches[0]


def summarize_xml_curves(
    xml_data: XMLData,
    pattern: str | None = None,
) -> pd.DataFrame:
    """Create a compact summary of all available measurement curves."""

    rows = []
    for curve in select_xml_curves(xml_data, pattern=pattern):
        values = curve.data["value"]
        rows.append(
            {
                "short_name": curve.short_name,
                "long_name": curve.long_name,
                "unit": curve.unit,
                "samples": curve.sample_count,
                "duration_s": curve.data["time_s"].iloc[-1] if curve.sample_count else np.nan,
                "min": values.min() if curve.sample_count else np.nan,
                "max": values.max() if curve.sample_count else np.nan,
                "mean": values.mean() if curve.sample_count else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)


def show_xml_contents(
    source: XMLData | PathType,
    pattern: str | None = None,
    show_metadata: bool = True,
) -> pd.DataFrame:
    """Print file metadata and a list of the available curves.

    The curve summary is also returned as a DataFrame so it can be used for
    further analysis in a notebook.
    """

    xml_data = source if isinstance(source, XMLData) else load_xml_data(source)
    print(f"File: {xml_data.file_path}")
    print(f"Measurement curves: {len(xml_data.curves)}")
    if show_metadata and xml_data.metadata:
        print("\nMetadata:")
        for key, value in xml_data.metadata.items():
            print(f"  {key}: {value}")

    summary = summarize_xml_curves(xml_data, pattern=pattern)
    print("\nAvailable curves:")
    print(summary.to_string(index=False) if not summary.empty else "  No matching curves.")
    return summary


def xml_to_long_dataframe(
    xml_data: XMLData,
    pattern: str | None = None,
) -> pd.DataFrame:
    """Combine selected curves into a long, analysis-friendly table."""

    curves = select_xml_curves(xml_data, pattern=pattern)
    if not curves:
        return pd.DataFrame(
            columns=["time_us", "time_s", "value", "short_name", "long_name", "unit"]
        )

    curves_with_samples = [curve for curve in curves if curve.sample_count]
    if not curves_with_samples:
        return pd.DataFrame(
            columns=["time_us", "time_s", "value", "short_name", "long_name", "unit"]
        )
    global_start = min(int(curve.data["time_us"].iloc[0]) for curve in curves_with_samples)
    frames = []
    for curve in curves:
        if not curve.sample_count:
            continue
        frame = curve.data.copy()
        frame["time_s"] = (frame["time_us"] - global_start) / 1e6
        frame["short_name"] = curve.short_name
        frame["long_name"] = curve.long_name
        frame["unit"] = curve.unit
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def xml_to_wide_dataframe(
    xml_data: XMLData,
    pattern: str | None = None,
    column: str = "short_name",
    interpolate: bool = False,
) -> pd.DataFrame:
    """Place measurement curves side by side using their absolute timestamps.

    Different sampling rates produce NaN values by default. With
    ``interpolate=True``, gaps within each measurement range are linearly
    interpolated; values outside the range are not extrapolated.
    """

    if column not in {"short_name", "long_name", "label"}:
        raise ValueError("column must be 'short_name', 'long_name', or 'label'.")

    curves = select_xml_curves(xml_data, pattern=pattern)
    series = []
    used_names: dict[str, int] = {}
    for curve in curves:
        if not curve.sample_count:
            continue
        base_name = getattr(curve, column)
        used_names[base_name] = used_names.get(base_name, 0) + 1
        name = base_name
        if used_names[base_name] > 1:
            name = f"{base_name} [{used_names[base_name]}]"
        values = curve.data.set_index("time_us")["value"].rename(name)
        series.append(values)

    if not series:
        return pd.DataFrame(columns=["time_s"])

    result = pd.concat(series, axis=1, join="outer").sort_index()
    if interpolate:
        result = result.interpolate(method="index", limit_area="inside")
    start_us = int(result.index.min())
    result.insert(0, "time_s", (result.index - start_us) / 1e6)
    result.index.name = "time_us"
    return result.reset_index()


def export_xml_to_csv(
    source: XMLData | PathType,
    output_path: PathType | None = None,
    pattern: str | None = None,
    interpolate: bool = False,
) -> Path:
    """Export selected XML curves to a wide-format CSV file."""

    xml_data = source if isinstance(source, XMLData) else load_xml_data(source)
    if output_path is None:
        name = xml_data.file_path.name
        if name.lower().endswith(".xml.gz"):
            name = name[:-7]
        elif name.lower().endswith(".xml"):
            name = name[:-4]
        output = xml_data.file_path.with_name(f"{name}.csv")
    else:
        output = Path(output_path).expanduser()

    frame = xml_to_wide_dataframe(xml_data, pattern=pattern, interpolate=interpolate)
    if len(frame.columns) <= 1:
        raise ValueError("No matching measurement curves found for CSV export.")
    frame.to_csv(output, index=False)
    return output.resolve()


def plot_xml_data(
    source: XMLData | PathType,
    pattern: str | None = None,
    names: str | Iterable[str] | None = None,
    separate_units: bool = False,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
):
    """Plot selected time series in the style of ``plot_xml_gz_file.py``.

    Args:
        source: Previously loaded XML data or a file path.
        pattern: Substring in the short name, long name, or unit.
        names: Optional exact short or long names.
        separate_units: Create a separate y-axis for each unit. This is usually
            more meaningful when plotting different physical quantities.
        show: Call ``plt.show()``. Set to ``False`` to customize or save the
            plot first.

    Returns:
        Matplotlib ``(figure, axes)``.
    """

    xml_data = source if isinstance(source, XMLData) else load_xml_data(source)
    curves = [
        curve
        for curve in select_xml_curves(xml_data, pattern=pattern, names=names)
        if curve.sample_count
    ]
    if not curves:
        raise ValueError("No matching measurement curves with data found.")

    global_start = min(int(curve.data["time_us"].iloc[0]) for curve in curves)
    if separate_units:
        units = list(dict.fromkeys(curve.unit or "no unit" for curve in curves))
    else:
        units = ["all"]

    if figsize is None:
        figsize = (16, max(5, 3.5 * len(units)))
    figure, axes = plt.subplots(len(units), 1, figsize=figsize, sharex=True, squeeze=False)
    axes_flat = axes[:, 0]

    for axis_index, unit in enumerate(units):
        axis = axes_flat[axis_index]
        current_curves = curves if not separate_units else [
            curve for curve in curves if (curve.unit or "no unit") == unit
        ]
        for curve in current_curves:
            time_s = (curve.data["time_us"] - global_start) / 1e6
            axis.plot(time_s, curve.data["value"], label=curve.label)
        axis.grid(True)
        axis.legend(loc="best")
        axis.set_ylabel(unit if separate_units else "Value")

    axes_flat[-1].set_xlabel("Cycle time [s]")
    axes_flat[0].set_title("Time series from the XML file")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes_flat


def plot_casting_overview(
    source: XMLData | PathType,
    figsize: tuple[float, float] = (16, 12),
    show: bool = True,
):
    """Plot the main casting signals in three vertically stacked panels.

    The panels contain:

    1. actual plunger stroke (``s I a``),
    2. nominal and actual casting-cylinder velocity (``v I`` and ``v I a``),
    3. metal, vacuum, primary-cylinder, and secondary-cylinder pressures.

    Vacuum pressure is plotted on a secondary y-axis because it is measured in
    mbar, whereas the other pressure signals are measured in bar.

    Returns:
        ``(figure, axes)`` where ``axes`` is a dictionary containing the
        ``stroke``, ``velocity``, ``pressure``, and ``vacuum`` axes.
    """

    xml_data = source if isinstance(source, XMLData) else load_xml_data(source)
    curves_by_name = {
        " ".join(curve.short_name.casefold().split()): curve
        for curve in xml_data.curves
        if curve.sample_count
    }

    def get_curves(*short_names: str) -> list[XMLCurve]:
        return [
            curves_by_name[name]
            for name in short_names
            if name in curves_by_name
        ]

    stroke_curves = get_curves("s i a")
    velocity_curves = get_curves("v i", "v i a")
    pressure_curves = get_curves("p im", "p im a", "p i.1 a", "p i.2 a")
    vacuum_curves = get_curves("p vacd1 a")
    selected_curves = (
        stroke_curves + velocity_curves + pressure_curves + vacuum_curves
    )
    if not selected_curves:
        raise ValueError("None of the expected casting curves were found in the XML file.")

    global_start = min(
        int(curve.data["time_us"].iloc[0]) for curve in selected_curves
    )
    figure, plot_axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    stroke_axis, velocity_axis, pressure_axis = plot_axes

    def plot_curves(axis, curves: Sequence[XMLCurve]) -> None:
        for curve in curves:
            normalized_name = " ".join(curve.short_name.casefold().split())
            line_style = "--" if normalized_name in {"v i", "p im"} else "-"
            time_s = (curve.data["time_us"] - global_start) / 1e6
            axis.plot(
                time_s,
                curve.data["value"],
                label=curve.label,
                linestyle=line_style,
            )

    plot_curves(stroke_axis, stroke_curves)
    stroke_axis.set_title("Plunger stroke – actual signal only")
    stroke_axis.set_ylabel("Stroke [mm]")

    plot_curves(velocity_axis, velocity_curves)
    velocity_axis.set_title("Casting-cylinder velocity")
    velocity_axis.set_ylabel("Velocity [m/s]")

    plot_curves(pressure_axis, pressure_curves)
    pressure_axis.set_title("Pressures")
    pressure_axis.set_ylabel("Pressure [bar]")

    vacuum_axis = pressure_axis.twinx()
    plot_curves(vacuum_axis, vacuum_curves)
    vacuum_axis.set_ylabel("Vacuum pressure [mbar]")
    vacuum_axis.set_ylim(0, 1000)
    vacuum_axis.grid(False)

    for axis in plot_axes:
        axis.grid(True)
        handles, labels = axis.get_legend_handles_labels()
        if handles and axis is not pressure_axis:
            axis.legend(handles, labels, loc="best")

    pressure_handles, pressure_labels = pressure_axis.get_legend_handles_labels()
    vacuum_handles, vacuum_labels = vacuum_axis.get_legend_handles_labels()
    if pressure_handles or vacuum_handles:
        pressure_axis.legend(
            pressure_handles + vacuum_handles,
            pressure_labels + vacuum_labels,
            loc="best",
        )

    pressure_axis.set_xlabel("Cycle time [s]")
    figure.suptitle(xml_data.file_path.name)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    if show:
        plt.show()

    axes = {
        "stroke": stroke_axis,
        "velocity": velocity_axis,
        "pressure": pressure_axis,
        "vacuum": vacuum_axis,
    }
    return figure, axes


def find_nearest_index(array: Sequence[float], value: float) -> int:
    """Return the index of the array element closest to ``value``."""

    values = np.asarray(array, dtype=float)
    if values.size == 0:
        raise ValueError("array must not be empty.")
    return int(np.nanargmin(np.abs(values - value)))


def time_from_stroke_velocity(
    stroke: Sequence[float],
    velocity: Sequence[float],
) -> np.ndarray:
    """Calculate time points from stroke and velocity using the trapezoidal rule.

    Stroke and velocity must have the same length. Stroke must increase
    monotonically, and the sum of adjacent velocities must not be zero. Units
    must be consistent, for example m and m/s.
    """

    stroke_array = np.asarray(stroke, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    if stroke_array.ndim != 1 or velocity_array.ndim != 1:
        raise ValueError("stroke and velocity must be one-dimensional arrays.")
    if len(stroke_array) != len(velocity_array):
        raise ValueError("stroke and velocity must have the same length.")
    if len(stroke_array) == 0:
        return np.array([], dtype=float)
    if np.any(np.diff(stroke_array) < 0):
        raise ValueError("stroke must increase monotonically.")

    velocity_sums = velocity_array[1:] + velocity_array[:-1]
    if np.any(np.isclose(velocity_sums, 0.0)):
        raise ValueError("Adjacent velocities sum to zero; time cannot be calculated.")
    delta_t = 2 * np.diff(stroke_array) / velocity_sums
    return np.concatenate(([0.0], np.cumsum(delta_t)))


def acceleration_from_velocity(
    velocity: Sequence[float],
    time: Sequence[float],
) -> np.ndarray:
    """Calculate the acceleration dv/dt of a velocity curve."""

    velocity_array = np.asarray(velocity, dtype=float)
    time_array = np.asarray(time, dtype=float)
    if velocity_array.ndim != 1 or time_array.ndim != 1:
        raise ValueError("velocity and time must be one-dimensional arrays.")
    if len(velocity_array) != len(time_array):
        raise ValueError("velocity and time must have the same length.")
    if len(time_array) < 2:
        raise ValueError("At least two time points are required.")
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("time must increase strictly monotonically.")
    return np.gradient(velocity_array, time_array)


def stroke_from_time_velocity(
    time: Sequence[float],
    velocity: Sequence[float],
    initial_stroke: float = 0.0,
) -> np.ndarray:
    """Integrate a velocity curve into stroke using the trapezoidal rule."""

    time_array = np.asarray(time, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    if time_array.ndim != 1 or velocity_array.ndim != 1:
        raise ValueError("time and velocity must be one-dimensional arrays.")
    if len(time_array) != len(velocity_array):
        raise ValueError("time and velocity must have the same length.")
    if len(time_array) == 0:
        return np.array([], dtype=float)
    if np.any(np.diff(time_array) < 0):
        raise ValueError("time must increase monotonically.")

    delta_s = 0.5 * (velocity_array[1:] + velocity_array[:-1]) * np.diff(time_array)
    return np.concatenate(([initial_stroke], initial_stroke + np.cumsum(delta_s)))


# Compatible name for existing code from plot_xml_gz_file.py.
def plot_compressed_xml(file_path: PathType, pattern: str | None = None):
    """Backward-compatible wrapper around :func:`plot_xml_data`."""

    return plot_xml_data(file_path, pattern=pattern)


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect and optionally plot a Buehler XML file from the command line."""

    import argparse
    import sys

    # The XML metadata may contain characters such as the Greek capital sigma.
    # Windows often defaults to CP1252, which cannot print those characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    default_file = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "ddm"
        / "ddm_y_FliesslaengerformBuehler_0003_ok.xml.gz"
    )
    parser = argparse.ArgumentParser(description="Inspect curves in a Buehler XML file.")
    parser.add_argument(
        "file",
        nargs="?",
        default=default_file,
        help=f"Path to the XML or XML.GZ file (default: {default_file}).",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        help="Substring in the short name, long name, or unit to select curves.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print the file contents without opening a plot.",
    )
    args = parser.parse_args(argv)

    xml_data = load_xml_data(args.file)
    show_xml_contents(xml_data, pattern=args.pattern)
    if not args.no_plot:
        if args.pattern:
            plot_xml_data(xml_data, pattern=args.pattern)
        else:
            plot_casting_overview(xml_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
