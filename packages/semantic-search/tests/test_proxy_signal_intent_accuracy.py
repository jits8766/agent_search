from types import SimpleNamespace

from semantic_search.contracts import FeedbackSignal
from semantic_search.measurement.evaluator import ProxySignalEvaluator


_THRESHOLDS = SimpleNamespace(correct_intent_confidence_min=0.92)


class _Store:
    def __init__(self, min_sample_size: int, observations=None):
        self._min_sample_size = min_sample_size
        self._observations = observations or []

    def snapshot(self):
        return self._observations

    def min_sample_size(self):
        return self._min_sample_size


class _Signals:
    def __init__(self, signals):
        self._signals = signals

    def recent(self, limit: int):
        return self._signals[-limit:]

    def stats(self):
        return {
            "filter_override": 0,
            "result_click": 0,
            "calibration_label": len(self._signals),
        }


def _signal(is_correct, tier="L1_semantic"):
    return FeedbackSignal(
        signal_id=FeedbackSignal.new_signal_id(),
        request_id=FeedbackSignal.new_signal_id(),
        signal_type="calibration_label",
        payload={"is_correct": is_correct, "tier": tier},
        signal_origin="offline_eval",
    )


def _evaluator(signals, min_sample_size=1, observations=None):
    return ProxySignalEvaluator(
        config=SimpleNamespace(),
        store=_Store(min_sample_size, observations=observations),
        signal_store=_Signals(signals),
        sanitizer=SimpleNamespace(),
        circuit_breaker=SimpleNamespace(),
        cache_stats_fn=lambda: {},
    )


def test_intent_classification_accuracy_from_calibration_labels():
    evaluator = _evaluator(
        [
            _signal(True, tier="L0_entity"),
            _signal(False, tier="L0_entity"),
            _signal("true", tier="L1_semantic"),
            _signal(1, tier="L1_semantic"),
        ],
        min_sample_size=2,
    )

    sig = evaluator._signal_intent_classification_accuracy()

    assert sig.name == "intent_classification_accuracy"
    assert sig.status == "ok"
    assert sig.value == 0.75
    assert sig.sample_size == 4
    assert sig.details["correct"] == 3
    assert sig.details["by_tier"]["L0_entity"] == {"correct": 1, "total": 2}
    assert sig.details["by_tier"]["L1_semantic"] == {"correct": 2, "total": 2}


def test_intent_classification_accuracy_insufficient_without_labels():
    evaluator = _evaluator([], min_sample_size=2)

    sig = evaluator._signal_intent_classification_accuracy()

    assert sig.name == "intent_classification_accuracy"
    assert sig.status == "insufficient_data"
    assert sig.value is None
    assert sig.sample_size == 0


def test_cost_per_correct_intent_query_joins_labels_to_observations():
    labels = [
        _signal(True, tier="L2_llm"),
        _signal(True, tier="L2_llm"),
        _signal(False, tier="L2_llm"),
    ]
    observations = [
        SimpleNamespace(
            request_id=labels[0].request_id,
            confidence=0.93,
            decision_cost_usd=0.03,
        ),
        SimpleNamespace(
            request_id=labels[1].request_id,
            confidence=0.98,
            decision_cost_usd=0.05,
        ),
        SimpleNamespace(
            request_id=labels[2].request_id,
            confidence=0.99,
            decision_cost_usd=0.11,
        ),
    ]
    evaluator = _evaluator(labels, min_sample_size=2, observations=observations)

    sig = evaluator._signal_cost_per_correct_intent_query(observations, _THRESHOLDS)

    assert sig.name == "cost_per_correct_intent_query_usd"
    assert sig.status == "ok"
    assert sig.sample_size == 2
    assert sig.value == 0.04
    assert sig.details["matched_labels"] == 3
    assert sig.details["correct_high_confidence"] == 2
    assert sig.details["confidence_min"] == 0.92


def test_cost_per_correct_intent_query_requires_confidence_floor():
    label = _signal(True, tier="L2_llm")
    observations = [
        SimpleNamespace(
            request_id=label.request_id,
            confidence=0.91,
            decision_cost_usd=0.03,
        ),
    ]
    evaluator = _evaluator([label], min_sample_size=1, observations=observations)

    sig = evaluator._signal_cost_per_correct_intent_query(observations, _THRESHOLDS)

    assert sig.name == "cost_per_correct_intent_query_usd"
    assert sig.status == "insufficient_data"
    assert sig.sample_size == 0
