"""Phase 1 qie_only launch stats — pass/fail gate aggregation."""
from semantic_search.config.models import QieOnlyLaunchConfig
from semantic_search.measurement.qie_only_launch import (
    QieOnlyLaunchStats,
    reset_qie_only_launch_stats_for_tests,
)


def _cfg(**overrides):
    values = {
        "min_sample_size": 5,
        "fail_rate_max": 0.05,
        "p95_latency_ms_max": 3000.0,
        "find_skipped_rate_max": 0.15,
        "hard_params_empty_rate_max": 0.40,
        "latency_window": 2000,
    }
    values.update(overrides)
    return QieOnlyLaunchConfig(**values)


def _stats(**overrides):
    cfg = _cfg(**overrides)
    return QieOnlyLaunchStats(
        min_sample_size=cfg.min_sample_size,
        fail_rate_max=cfg.fail_rate_max,
        p95_latency_ms_max=cfg.p95_latency_ms_max,
        find_skipped_rate_max=cfg.find_skipped_rate_max,
        hard_params_empty_rate_max=cfg.hard_params_empty_rate_max,
        latency_window=cfg.latency_window,
    )


def test_insufficient_sample_until_min():
    stats = _stats(min_sample_size=5)
    for _ in range(3):
        stats.record_complete(
            latency_ms=100.0,
            find_skipped_count=0,
            hard_params_empty=0,
            source="L0_llm",
        )
    snap = stats.snapshot()
    assert snap["ml_owned_overall"] == "insufficient_sample"
    assert snap["gates"]["qi_availability"]["status"] == "insufficient_sample"


def test_pass_when_rates_inside_thresholds():
    stats = _stats(min_sample_size=5)
    for _ in range(10):
        stats.record_complete(
            latency_ms=120.0,
            find_skipped_count=0,
            hard_params_empty=0,
            source="L0_llm",
        )
    snap = stats.snapshot()
    assert snap["ml_owned_overall"] == "pass"
    assert snap["gates"]["qi_availability"]["status"] == "pass"
    assert snap["gates"]["find_skipped_rate"]["status"] == "pass"
    assert snap["gates"]["hard_params_empty_rate"]["status"] == "pass"
    assert snap["gates"]["qi_latency_p95_ms"]["status"] == "pass"
    assert snap["gates"]["fos_wiring"]["status"] == "external"
    assert snap["phase1_launch_overall"] == "pending_external_gates"


def test_p95_latency_gate_ignores_single_extreme_outlier():
    stats = _stats(min_sample_size=20, p95_latency_ms_max=3000.0)
    for _ in range(19):
        stats.record_complete(
            latency_ms=200.0,
            find_skipped_count=0,
            hard_params_empty=0,
            source="L0_llm",
        )
    stats.record_complete(
        latency_ms=10_000.0,
        find_skipped_count=0,
        hard_params_empty=0,
        source="L0_llm",
    )
    snap = stats.snapshot()
    assert snap["rates"]["p95_latency_ms"] == 200.0
    assert snap["gates"]["qi_latency_p95_ms"]["status"] == "pass"


def test_fail_on_high_error_rate():
    stats = _stats(min_sample_size=5, fail_rate_max=0.05)
    for _ in range(8):
        stats.record_complete(
            latency_ms=100.0,
            find_skipped_count=0,
            hard_params_empty=0,
            source="L0_llm",
        )
    for _ in range(2):
        stats.record_failed(status=503, reason="configuration_error")
    snap = stats.snapshot()
    assert snap["rates"]["fail_rate"] == 0.2
    assert snap["gates"]["qi_availability"]["status"] == "fail"
    assert snap["ml_owned_overall"] == "fail"


def test_singleton_reset_for_tests():
    a = reset_qie_only_launch_stats_for_tests(_cfg())
    a.record_complete(
        latency_ms=10.0,
        find_skipped_count=1,
        hard_params_empty=1,
        source="L0_regex",
    )
    b = reset_qie_only_launch_stats_for_tests(_cfg())
    assert b.snapshot()["counts"]["complete"] == 0
