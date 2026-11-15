"""Offline tests for element_calcium_imaging.dff.

They pin the recipe validation (a parameter that is not applied can never be stored) and
the line-by-line equivalence with the CASCADE input code in imaging.Activity.make.
Agreement with the stored CASCADE traces is measured on the server
(adamacs_ingest scripts/verify_dff0_cascade.py).
"""
import pytest

np = pytest.importorskip("numpy", reason="needs numpy; CI installs pytest only")
pytest.importorskip("scipy", reason="needs scipy")

from element_calcium_imaging import dff as D  # noqa: E402


def _recipe(**kw):
    r = dict(dark_method="none", neuropil_factor=0.7, neuropil_offset="median",
             baseline_method="running_percentile", baseline_percentile=8, baseline_window_s=60.0,
             f0_floor_method="none")
    r.update(kw)
    return r


# ---------------------------------------------------------------- recipe validation

def test_unknown_column_is_refused():
    with pytest.raises(ValueError, match="unknown"):
        D.normalise_recipe(_recipe(f0_floor_quantile=0.01))


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="baseline_method"):
        D.normalise_recipe(_recipe(baseline_method="wavelet"))


def test_missing_required_column_is_refused():
    r = _recipe()
    del r["baseline_percentile"]
    with pytest.raises(ValueError, match="needs baseline_percentile"):
        D.normalise_recipe(r)


def test_ignored_column_is_refused():
    # a floor value with floor 'none' would be stored but never applied
    with pytest.raises(ValueError, match="not used"):
        D.normalise_recipe(_recipe(f0_floor_value=0.05))


def test_zero_neuropil_normalises_offset_and_hash():
    a = D.recipe_hash(_recipe(neuropil_factor=0, neuropil_offset="median"))
    b = D.recipe_hash(_recipe(neuropil_factor=0.0, neuropil_offset="none"))
    assert a == b


def test_int_and_float_hash_the_same():
    assert D.recipe_hash(_recipe(baseline_percentile=8)) == D.recipe_hash(_recipe(baseline_percentile=8.0))


def test_known_recipes_are_valid_and_distinct():
    hashes = {D.recipe_hash(r) for _, r in D.KNOWN_RECIPES.values()}
    assert len(hashes) == len(D.KNOWN_RECIPES)


# ---------------------------------------------------------------- dark window

def test_shutter_window_uses_nearest_frames_and_margin():
    frames = np.arange(0, 10, 0.1)               # 10 Hz
    a, b = D.shutter_dark_frames(frames, [0.0, 2.0], [0.5, 2.5], 0.09)
    assert (a, b) == (6, 19)                     # 0.59 s -> frame 6, 1.91 s -> frame 19


def test_missing_shutter_falls_back_and_says_so():
    F = np.ones((3, 50))
    rec = D.normalise_recipe(_recipe(dark_method="shutter", dark_margin_s=0.09,
                                     dark_frame_start=12, dark_frame_stop=24))
    v, a, b, src = D.resolve_dark(rec, F, np.arange(50) / 10, [1.0], [1.5])
    assert (a, b, src) == (12, 24, "fallback_frames") and v == 1.0


def test_shutter_without_fallback_raises():
    rec = D.normalise_recipe(_recipe(dark_method="shutter", dark_margin_s=0.09))
    with pytest.raises(ValueError, match="fallback"):
        D.resolve_dark(rec, np.ones((2, 10)), np.arange(10.0), [], [])


# ---------------------------------------------------------------- equivalence with Activity.make

def _activity_make_dff(Fall, Fneu_all, framerate, neu, percentile, mean_darksignal):
    """The CASCADE branch of imaging.Activity.make (BUGPU copy, 2026-09-03), verbatim maths."""
    from scipy.ndimage import percentile_filter
    smoothing_window = framerate * 60
    Fall = Fall - mean_darksignal
    Fneu_all = Fneu_all - mean_darksignal
    dF = (Fall - neu * Fneu_all) + (np.nanmedian(Fneu_all, axis=1, keepdims=True) * neu)
    F0 = np.array([percentile_filter(dF[i, :], percentile, size=int(smoothing_window))
                   for i in range(dF.shape[0])])
    return (dF - F0) / F0


@pytest.mark.parametrize("neu,pct", [(0.7, 8), (0.7, 11), (0.0, 8)])
def test_frames_recipe_equals_activity_make(neu, pct):
    rng = np.random.default_rng(0)
    fps = 29.97
    n = int(fps * 200)
    F = (300 + 50 * rng.random((4, n))).astype(np.float32)
    Fneu = (200 + 20 * rng.random((4, n))).astype(np.float32)
    rec = _recipe(dark_method="frames", dark_frame_start=12, dark_frame_stop=24,
                  neuropil_factor=neu, baseline_percentile=pct)
    out = D.compute_dff0(F, Fneu, fps, rec)
    dark = np.mean(np.mean(F, axis=0)[12:24])
    ref = _activity_make_dff(F, Fneu, fps, neu, pct, dark)
    assert out["dark_source"] == "frames"
    np.testing.assert_allclose(out["dff"], ref.astype(np.float32), rtol=0, atol=1e-6)


def test_nonpositive_f0_is_flagged_not_hidden():
    fps = 10.0
    F = np.vstack([np.full(2000, 100.0), np.full(2000, 1.0)]).astype(np.float32)
    Fneu = np.vstack([np.full(2000, 50.0), np.full(2000, 10.0)]).astype(np.float32)
    out = D.compute_dff0(F, Fneu, fps, _recipe(neuropil_offset="none", f0_floor_method="flag"))
    assert out["f0_nonpositive"].tolist() == [False, True]


def test_nan_floor_blanks_where_f0_nonpositive():
    fps = 10.0
    F = np.vstack([np.full(2000, 1.0)]).astype(np.float32)
    Fneu = np.vstack([np.full(2000, 10.0)]).astype(np.float32)
    out = D.compute_dff0(F, Fneu, fps, _recipe(neuropil_offset="none", f0_floor_method="nan"))
    assert np.isnan(out["dff"]).all()


def test_linear_detrend_removes_slope_keeps_mean():
    X = np.vstack([np.linspace(0, 10, 101) + 5]).astype(float)
    Y = D.detrend(X, "linear", fps=1.0)
    assert np.allclose(Y, Y.mean()) and np.isclose(Y.mean(), X.mean())


# ---------------------------------------------------------------- tables' contents

def test_cascade_recipes_point_at_known_recipes():
    assert set(D.CASCADE_RECIPES.values()) <= set(D.KNOWN_RECIPES)


def test_existing_methods_keep_the_post_fix_recipe():
    # cascade_inference, _25ms_filter and _global_causal were computed with recipe 2 since
    # 2026-09-03 (verified by recomputation, adamacs_ingest PR #25): the new make must not change them
    for m in ("cascade_inference", "cascade_inference_25ms_filter", "cascade_inference_global_causal"):
        assert D.CASCADE_RECIPES[m] == 2
    rec = D.normalise_recipe(D.KNOWN_RECIPES[2][1])
    assert (rec["dark_method"], rec["neuropil_factor"], rec["neuropil_offset"], rec["baseline_percentile"],
            rec["baseline_window_s"], rec["f0_floor_method"]) == ("shutter", 0.7, "median", 8.0, 60.0, "none")


def test_element_version_has_commit_and_file_hash():
    v = D.element_version()
    assert "+" in v and len(v.split("+")[1]) == 12


# ---------------------------------------------------------------- review regressions

def _dffparamset_definition():
    import os
    src = open(os.path.join(os.path.dirname(D.__file__), "imaging.py")).read()
    i = src.index("class DffParamSet(dj.Lookup):")
    j = src.index('"""', src.index('definition = """', i) + len('definition = """'))
    return src[i:j]


def test_recipe_numerics_are_stored_as_double():
    # MySQL FLOAT is single precision: 0.7 would come back as 0.699999988..., changing the
    # recipe hash on a round trip and the parameters the database-backed make computes with
    definition = _dffparamset_definition()
    numeric = [c for c, default in D.RECIPE_COLUMNS.items()
               if c not in ("dark_frame_start", "dark_frame_stop", "baseline_step", "extra")
               and not c.endswith("_method") and c != "neuropil_offset"]
    for col in numeric:
        line = next(l for l in definition.splitlines() if l.strip().startswith(col))
        assert ": double" in line, line


def test_single_precision_round_trip_would_change_the_hash():
    rec = D.normalise_recipe(D.KNOWN_RECIPES[2][1])
    rounded = dict(rec, neuropil_factor=float(np.float32(rec["neuropil_factor"])))
    assert D.recipe_hash(rec) != D.recipe_hash(rounded)
