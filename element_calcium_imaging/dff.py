"""dF/F0 from suite2p fluorescence, as one explicit recipe per parameter set.

This module is the single implementation behind ``imaging.DffParamSet`` / ``imaging.Dff0``
and the input of CASCADE in ``imaging.Activity.make``. It is pure (numpy/scipy only, no
DataJoint), so it can be tested offline and called identically from the schema, from
notebooks (``adamacs.helpers.stack_helpers.calculate_dFF`` wraps it) and from a worker.

The steps run in a fixed order; every step is one column group of ``DffParamSet``:

1. dark      F, Fneu <- F - dark, Fneu - dark
2. neuropil  Fc <- F - a*Fneu + a*offset(Fneu)            offset: none | median | mean
3. detrend   Fc <- Fc - trend(Fc)                         none | linear | highpass
4. baseline  F0 <- baseline(Fc)                           running_percentile | maximin | global_percentile
5. floor     how F0 <= 0 is handled                       none | flag | nan | fraction_of_median_F | quantile_of_F0
6. dF/F0     (Fc - F0) / F0

Reproducing the CASCADE input of the legacy code
------------------------------------------------
Before this module, ``imaging.Activity.make`` on BUGPU (installed from e75c3fd and edited in
place 2026-09-03) computed, for every ``cascade_inference*`` method:

* dark = mean over frames [d0, d1) of the all-mask average F, where [d0, d1) is the
  shutter-closed window (``shutter`` events, 0.09 s margin, nearest 2P frame), or frames
  [12, 24) when that fails. Before 2026-09-03 the window computation always failed.
* dF = (F - a*Fneu) + a*nanmedian(Fneu), a = 0 for ``*no_neuropil*`` else 0.7;
* F0 = scipy.ndimage.percentile_filter(dF_row, p, size=int(fps*60)), p = 11 for ``*_11``
  else 8, default mode ('reflect');
* dFF = (dF - F0) / F0, no floor.

The functions below follow that code line by line for the corresponding recipes, so a
recipe's dF/F0 can be compared with what CASCADE received.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import uuid

import numpy as np

# --------------------------------------------------------------------------------------
# Recipe registry: which methods exist for each step, and which columns each one needs.
# A method that is not in this registry cannot be inserted into DffParamSet.
# --------------------------------------------------------------------------------------

#: All recipe columns, in table order, with their default.
RECIPE_COLUMNS = {
    "dark_method": None,
    "dark_frame_start": None,
    "dark_frame_stop": None,
    "dark_margin_s": None,
    "neuropil_factor": None,
    "neuropil_offset": None,
    "detrend_method": "none",
    "detrend_window_s": None,
    "baseline_method": None,
    "baseline_percentile": None,
    "baseline_window_s": None,
    "baseline_smooth_s": None,
    "baseline_step": 1,
    "f0_floor_method": None,
    "f0_floor_value": None,
    "extra": None,
}

#: step column -> {method: (required columns, optional columns)}
REGISTRY = {
    "dark_method": {
        "none": ((), ()),
        "frames": (("dark_frame_start", "dark_frame_stop"), ()),
        # shutter window; dark_frame_start/stop name the fallback when the shutter events
        # are missing (the lab code falls back to frames 12-24)
        "shutter": (("dark_margin_s",), ("dark_frame_start", "dark_frame_stop")),
    },
    "neuropil_offset": {"none": ((), ()), "median": ((), ()), "mean": ((), ())},
    "detrend_method": {
        "none": ((), ()),
        "linear": ((), ()),
        "highpass": (("detrend_window_s",), ()),
    },
    "baseline_method": {
        "running_percentile": (("baseline_percentile", "baseline_window_s"), ("baseline_step",)),
        "maximin": (("baseline_smooth_s", "baseline_window_s"), ()),
        "global_percentile": (("baseline_percentile",), ()),
    },
    "f0_floor_method": {
        "none": ((), ()),
        "flag": ((), ()),
        "nan": ((), ()),
        "fraction_of_median_F": (("f0_floor_value",), ()),
        "quantile_of_F0": (("f0_floor_value",), ()),
    },
}

_ALWAYS_REQUIRED = ("dark_method", "neuropil_factor", "neuropil_offset", "detrend_method",
                    "baseline_method", "f0_floor_method")


def normalise_recipe(params: dict) -> dict:
    """Validate a recipe and return it with every column present, in table order.

    Raises ``ValueError`` on unknown keys, unknown methods, missing required columns and
    columns the chosen methods would ignore (so a recipe can never name a parameter that
    is not applied). ``neuropil_offset`` is normalised to 'none' when the factor is 0,
    because the offset is then multiplied by zero.
    """
    unknown = set(params) - set(RECIPE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown recipe columns: {sorted(unknown)}")
    rec = {k: params.get(k, v) for k, v in RECIPE_COLUMNS.items()}
    for col in _ALWAYS_REQUIRED:
        if rec[col] is None:
            raise ValueError(f"{col} is required")
    if float(rec["neuropil_factor"]) == 0.0:
        rec["neuropil_offset"] = "none"
    used = set(_ALWAYS_REQUIRED) | {"extra"}
    for step, methods in REGISTRY.items():
        method = rec[step]
        if method not in methods:
            raise ValueError(f"{step}={method!r} is not one of {sorted(methods)}")
        required, optional = methods[method]
        for col in required:
            if rec[col] is None:
                raise ValueError(f"{step}={method!r} needs {col}")
        used |= set(required) | set(optional)
    ignored = [k for k in RECIPE_COLUMNS
               if k not in used and rec[k] is not None and rec[k] != RECIPE_COLUMNS[k]]
    if ignored:
        raise ValueError(f"columns not used by the chosen methods: {ignored}")
    # canonical numeric types, so 8 and 8.0 hash the same
    for k in ("dark_margin_s", "neuropil_factor", "detrend_window_s", "baseline_percentile",
              "baseline_window_s", "baseline_smooth_s", "f0_floor_value"):
        if rec[k] is not None:
            rec[k] = float(rec[k])
    for k in ("dark_frame_start", "dark_frame_stop", "baseline_step"):
        if rec[k] is not None:
            rec[k] = int(rec[k])
    return rec


def recipe_hash(rec: dict) -> uuid.UUID:
    """UUID from the normalised recipe (all columns, sorted keys)."""
    blob = json.dumps(normalise_recipe(rec), sort_keys=True, default=str).encode()
    return uuid.UUID(hashlib.md5(blob).hexdigest())


# --------------------------------------------------------------------------------------
# Step 1: dark signal
# --------------------------------------------------------------------------------------

def _closest_index(series, target):
    """Index of the element of the sorted ``series`` closest to ``target``.

    Same tie rule as the lab's ``closest_timestamp`` (ties go to the earlier element).
    """
    index = bisect.bisect_left(series, target)
    if index == 0:
        return 0
    if index == len(series):
        return len(series) - 1
    before, after = series[index - 1], series[index]
    return index if after - target < target - before else index - 1


def shutter_dark_frames(frame_times, shutter_starts, shutter_ends, margin_s):
    """[first, stop) frames of the shutter-closed window, as the lab code computes them.

    The window runs from the end of the first shutter event + margin to the start of the
    second shutter event - margin. Raises ``ValueError`` if there are fewer than two
    shutter events or no frame times.
    """
    frame_times = np.asarray(frame_times, dtype=float)
    starts = np.sort(np.asarray(shutter_starts, dtype=float))
    ends = np.sort(np.asarray(shutter_ends, dtype=float))
    if len(starts) < 2 or len(ends) < 1 or frame_times.size == 0:
        raise ValueError("need >= 2 shutter events and frame times")
    t0 = float(ends[0]) + margin_s
    t1 = float(starts[1]) - margin_s
    series = frame_times.tolist()
    return _closest_index(series, t0), _closest_index(series, t1)


def dark_signal(F, first, stop):
    """Mean over frames [first, stop) of the average trace over all masks."""
    avg = np.mean(F, axis=0)
    return float(np.mean(avg[first:stop]))


def resolve_dark(rec, F, frame_times=None, shutter_starts=None, shutter_ends=None):
    """Return (dark_value, first, stop, source) for one curation."""
    method = rec["dark_method"]
    if method == "none":
        return 0.0, None, None, "none"
    if method == "frames":
        a, b = rec["dark_frame_start"], rec["dark_frame_stop"]
        return dark_signal(F, a, b), a, b, "frames"
    try:
        a, b = shutter_dark_frames(frame_times, shutter_starts, shutter_ends, rec["dark_margin_s"])
        return dark_signal(F, a, b), a, b, "shutter"
    except (ValueError, TypeError, IndexError):
        a, b = rec["dark_frame_start"], rec["dark_frame_stop"]
        if a is None or b is None:
            raise ValueError("shutter window unavailable and the recipe names no fallback frames")
        return dark_signal(F, a, b), a, b, "fallback_frames"


# --------------------------------------------------------------------------------------
# Steps 2-4
# --------------------------------------------------------------------------------------

def neuropil_correct(F, Fneu, factor, offset):
    """F - a*Fneu + a*offset(Fneu), row-wise."""
    out = F - factor * Fneu
    if offset == "median":
        out = out + factor * np.nanmedian(Fneu, axis=1, keepdims=True)
    elif offset == "mean":
        out = out + factor * np.nanmean(Fneu, axis=1, keepdims=True)
    return out


def detrend(X, method, fps, window_s=None):
    """Remove a slow trend before the baseline, keeping each row's mean level."""
    if method == "none":
        return X
    if method == "linear":
        t = np.arange(X.shape[1], dtype=float)
        t = t - t.mean()
        slope = (X - X.mean(axis=1, keepdims=True)) @ t / (t @ t)
        return X - slope[:, None] * t[None, :]
    if method == "highpass":
        from scipy.ndimage import uniform_filter1d
        size = max(3, int(round(window_s * fps)))
        trend = uniform_filter1d(X, size=size, axis=1, mode="nearest")
        return X - trend + X.mean(axis=1, keepdims=True)
    raise ValueError(method)


def baseline(X, rec, fps, n_jobs=1):
    """F0 per row according to the recipe's baseline columns."""
    from scipy.ndimage import percentile_filter
    method = rec["baseline_method"]
    if method == "global_percentile":
        return np.repeat(np.percentile(X, rec["baseline_percentile"], axis=1, keepdims=True),
                         X.shape[1], axis=1)
    if method == "maximin":
        from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
        win = max(3, int(round(rec["baseline_window_s"] * fps)))
        s = gaussian_filter1d(X, sigma=rec["baseline_smooth_s"] * fps, axis=1, mode="nearest")
        return maximum_filter1d(minimum_filter1d(s, win, axis=1, mode="nearest"), win, axis=1,
                                mode="nearest")
    # running percentile. step == 1: exactly the lab/CASCADE call
    # (percentile_filter(row, p, size=int(fps*window)), default mode 'reflect').
    p, step = rec["baseline_percentile"], rec["baseline_step"] or 1
    size = int(fps * rec["baseline_window_s"])
    if step == 1:
        def one(row):
            return percentile_filter(row, p, size=size)
        if n_jobs == 1:
            return np.array([one(r) for r in X])
        from joblib import Parallel, delayed
        return np.array(Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(r) for r in X))
    # step > 1: the faster subsample-and-interpolate approximation (leondrift recipes)
    win = max(3, int(round(rec["baseline_window_s"] * fps)) | 1)
    xs = X[:, ::step]
    b = percentile_filter(xs, p, size=(1, max(3, win // step) | 1), mode="nearest")
    idx = np.arange(X.shape[1]) / step
    return np.array([np.interp(idx, np.arange(xs.shape[1]), r) for r in b])


# --------------------------------------------------------------------------------------
# Steps 5-6 and the whole recipe
# --------------------------------------------------------------------------------------

def empty_rois(F):
    """Rows of F that carry no fluorescence: all zero or non-finite in every frame (suite2p: every pixel of the
    ROI overlaps another ROI, so none is used for F)."""
    F = np.asarray(F)
    return ~np.any(np.isfinite(F) & (F != 0), axis=1)


def roi_qc_reject(criterion, F_rows):
    """Rejection flag per ROI for one imaging.RoiQCCriterion; F_rows: sequence of 1-D fluorescence traces."""
    if criterion == "empty_f":
        return [bool(empty_rois(np.asarray(f, dtype=float)[None, :])[0]) for f in F_rows]
    raise ValueError(f"unknown ROI QC criterion {criterion!r}")


def compute_dff0(F, Fneu, fps, rec, frame_times=None, shutter_starts=None, shutter_ends=None,
                 n_jobs=1, return_intermediates=False):
    """dF/F0 of all masks of one curation under one recipe.

    Parameters
    ----------
    F, Fneu : (n_masks, n_frames) arrays, rows in mask order. All masks of the curation are
        needed because the dark signal is computed from their average.
    fps : float, ``scan.ScanInfo.fps``.
    rec : recipe dict (normalised here).
    frame_times, shutter_starts, shutter_ends : DAQ times for the shutter dark window.

    Returns
    -------
    dict with ``dff`` (float32, n_masks x n_frames), ``f0_median``, ``f0_min``,
    ``f0_nonpositive`` and ``f_empty`` (per mask; ``dff`` is NaN where ``f_empty``), and the resolved ``dark_signal``, ``dark_frame_start``,
    ``dark_frame_stop``, ``dark_source``. With ``return_intermediates=True`` also ``dF`` (the
    neuropil-corrected, detrended fluorescence) and ``F0`` (the baseline), both full arrays.
    """
    rec = normalise_recipe(rec)
    F = np.asarray(F)
    Fneu = np.asarray(Fneu)
    dark, d0, d1, source = resolve_dark(rec, F, frame_times, shutter_starts, shutter_ends)
    # same dtype behaviour as the lab code: subtract a scalar from the stored arrays
    Fd = F - np.asarray(dark, dtype=F.dtype)
    Fnd = Fneu - np.asarray(dark, dtype=Fneu.dtype)
    dF = neuropil_correct(Fd, Fnd, rec["neuropil_factor"], rec["neuropil_offset"])
    dF = detrend(dF, rec["detrend_method"], fps, rec["detrend_window_s"])
    F0 = baseline(dF, rec, fps, n_jobs=n_jobs)
    nonpos = ~(np.nanmin(F0, axis=1) > 0)
    # suite2p writes F = 0 in every frame for an ROI whose pixels all overlap other ROIs (overlapping pixels are
    # excluded from F); its "dF/F0" would be pure neuropil (-a*Fneu) on a negative F0, so it is blanked
    empty = empty_rois(F)
    floor = rec["f0_floor_method"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if floor in ("none", "flag"):
            dff = (dF - F0) / F0
        elif floor == "nan":
            dff = (dF - F0) / F0
            dff[F0 <= 0] = np.nan
        elif floor == "fraction_of_median_F":
            level = np.median(F, axis=1, keepdims=True)
            dff = (dF - F0) / np.maximum(F0, rec["f0_floor_value"] * level)
            dff[(level <= 0).ravel()] = np.nan
        elif floor == "quantile_of_F0":
            q = np.quantile(F0, rec["f0_floor_value"], axis=1, keepdims=True)
            dff = (dF - F0) / np.maximum(F0, q)
        else:
            raise ValueError(floor)
    dff[empty] = np.nan
    out = {
        "dff": dff.astype(np.float32),
        "f_empty": empty,
        "f0_median": np.nanmedian(F0, axis=1).astype(float),
        "f0_min": np.nanmin(F0, axis=1).astype(float),
        "f0_nonpositive": nonpos,
        "dark_signal": float(dark),
        "dark_frame_start": d0,
        "dark_frame_stop": d1,
        "dark_source": source,
    }
    if return_intermediates:
        out["dF"], out["F0"] = dF, F0
    return out


# --------------------------------------------------------------------------------------
# The recipes found in the lab's code (DffParamSet rows 1-10 of the proposal)
# --------------------------------------------------------------------------------------

_CASCADE = dict(neuropil_offset="median", detrend_method="none",
                baseline_method="running_percentile", baseline_window_s=60.0, f0_floor_method="none")
_OLD_DARK = dict(dark_method="frames", dark_frame_start=12, dark_frame_stop=24)
_NEW_DARK = dict(dark_method="shutter", dark_margin_s=0.09, dark_frame_start=12, dark_frame_stop=24)

KNOWN_RECIPES = {
    1: ("lab standard (proposed): shutter dark (fallback 12-24), neuropil 0.7 + median, running p8 / 60 s, F0<=0 flagged",
        dict(_CASCADE, **_NEW_DARK, neuropil_factor=0.7, baseline_percentile=8, f0_floor_method="flag")),
    2: ("CASCADE p8 since 2026-09-03 (cascade_inference, _25ms_filter, _global_causal) = lab helper calculate_dFF with CASCADE settings",
        dict(_CASCADE, **_NEW_DARK, neuropil_factor=0.7, baseline_percentile=8)),
    3: ("CASCADE p8 before 2026-09-03: dark frames 12-24",
        dict(_CASCADE, **_OLD_DARK, neuropil_factor=0.7, baseline_percentile=8)),
    4: ("CASCADE p11 since 2026-09-03 (cascade_inference_11)",
        dict(_CASCADE, **_NEW_DARK, neuropil_factor=0.7, baseline_percentile=11)),
    5: ("CASCADE p11 before 2026-09-03 (cascade_inference_11): dark frames 12-24",
        dict(_CASCADE, **_OLD_DARK, neuropil_factor=0.7, baseline_percentile=11)),
    6: ("CASCADE no neuropil p8 before 2026-09-03: dark frames 12-24",
        dict(_CASCADE, **_OLD_DARK, neuropil_factor=0.0, baseline_percentile=8)),
    7: ("CASCADE no neuropil p11 before 2026-09-03: dark frames 12-24",
        dict(_CASCADE, **_OLD_DARK, neuropil_factor=0.0, baseline_percentile=11)),
    8: ("leondrift dff: no dark, neuropil 0.7, running p10 / 60 s (subsampled), floor 5 % of median F",
        dict(dark_method="none", neuropil_factor=0.7, neuropil_offset="none", detrend_method="none",
             baseline_method="running_percentile", baseline_percentile=10, baseline_window_s=60.0,
             baseline_step=30, f0_floor_method="fraction_of_median_F", f0_floor_value=0.05)),
    9: ("leondrift dff_np (Bauer 2024 style): neuropil 0.7 + mean, running p10 / 60 s (subsampled), floor 5 % of median F",
        dict(dark_method="none", neuropil_factor=0.7, neuropil_offset="mean", detrend_method="none",
             baseline_method="running_percentile", baseline_percentile=10, baseline_window_s=60.0,
             baseline_step=30, f0_floor_method="fraction_of_median_F", f0_floor_value=0.05)),
    10: ("suite2p convention: neuropil 0.7, maximin baseline (sigma 10 s, 60 s), F0<=0 flagged",
         dict(dark_method="none", neuropil_factor=0.7, neuropil_offset="none", detrend_method="none",
              baseline_method="maximin", baseline_smooth_s=10.0, baseline_window_s=60.0,
              f0_floor_method="flag")),
}

#: imaging.CascadeRecipe contents: extraction_method -> recipe used by Activity.make from now on.
#: Methods without a row here (no_neuropil variants) are not computed by the new code.
#: Identical to what the in-place-edited code on BUGPU computed since 2026-09-03 (verified,
#: adamacs_ingest PR #25), so existing methods keep producing the same spikes.
CASCADE_RECIPES = {
    "cascade_inference": 2,
    "cascade_inference_25ms_filter": 2,
    "cascade_inference_global_causal": 2,
}

#: Provenance of the LEGACY rows: extraction_method -> (recipe before the 2026-09-03 code change, recipe since)
#: Verified 2026-11-15 by re-running CASCADE (BUGPU, Cascade 4c113e4) on the stored dF/F0:
#: recipes 2 and 3 reproduce cascade_inference / _25ms_filter / _global_causal to float
#: precision (except masks with F0 <= 0). The _11 and no_neuropil_11 rows written in Dec 2025
#: are NOT reproduced by any recipe: their stored traces do not correspond to their own masks
#: (same-mask r vs _25ms_filter ~0 on 20/20 curations, control 0.97). Recipes 5 and 7 describe
#: the code as written, not those rows. All _11 / no_neuropil_11 rows were deleted from
#: imaging on 2026-11-15 (manifest ~/adamacs/out/cascade11_deleted_2026-11-15.csv).
CASCADE_METHOD_RECIPES = {
    "cascade_inference": (3, 2),
    "cascade_inference_25ms_filter": (3, 2),
    "cascade_inference_global_causal": (3, 2),
    "cascade_inference_11": (5, 4),
    "cascade_inference_no_neuropil": (6, None),
    "cascade_inference_no_neuropil_11": (7, None),
}

#: When the in-place edit of element_calcium_imaging/imaging.py on BUGPU took effect (file mtime).
CASCADE_CODE_CHANGE = "2026-09-03 15:22:15"


# --------------------------------------------------------------------------------------
# Version stamp: which code produced a row
# --------------------------------------------------------------------------------------

def element_version() -> str:
    """'<commit[:10]>+<sha256(imaging.py, dff.py)[:12]>' of the installed package.

    The commit comes from pip's ``direct_url.json`` ('local' for an editable/unknown
    install). The file hash changes with any in-place edit, so a row computed by edited
    code can always be told apart from one computed by the commit.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in ("imaging.py", "dff.py"):
        with open(os.path.join(here, name), "rb") as f:
            h.update(f.read())
    commit = "local"
    try:
        from importlib.metadata import PackageNotFoundError, distribution
        for dist_name in ("element-calcium-imaging", "element_calcium_imaging"):
            try:
                info = json.loads(distribution(dist_name).read_text("direct_url.json") or "{}")
                commit = (info.get("vcs_info") or {}).get("commit_id", "local")[:10]
                break
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    return f"{commit}+{h.hexdigest()[:12]}"
