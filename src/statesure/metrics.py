"""Evaluation against human labels.

Metrics are computed per installation, recipe version, judge, stage and signal.
They are never pooled across installations for promotion.

Key rules:

* A sample a person could not judge is excluded. A sample the judge did not
  answer (abstained, invalid, error) is kept: for ``recall_all`` it counts as a
  miss.
* Samples are weighted by ``1 / inclusion_probability`` times the stratum's
  stored/labeled ratio, so the order in which people label does not bias
  estimates. Wilson intervals use Kish's effective sample size.
* Probability thresholds are chosen on one set of days and measured on the
  other.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import tzinfo
from statistics import median
from typing import Any

from .confirm import ChangeEvent
from .readings import Outcome, SignalReading
from .recipe import Recipe, SignalSpec
from .store import EVAL_STAGES, HUMAN_UNCERTAIN, SampleRecord

Z95 = 1.959963984540054
CALIBRATION_BINS = 10


@dataclass(frozen=True)
class Ratio:
    """A weighted proportion with a Wilson 95% interval."""

    value: float | None
    interval: tuple[float, float] | None
    n: int
    n_eff: float

    @property
    def lower(self) -> float | None:
        return self.interval[0] if self.interval else None


@dataclass(frozen=True)
class EvalKey:
    install_fingerprint: str
    recipe_id: str
    recipe_fingerprint: str
    judge_id: str
    stage: str
    signal: str


@dataclass(frozen=True)
class PresenceMetrics:
    positive_value: str
    positives: int
    negatives: int
    tp: int
    fp: int
    fn: int
    tn: int
    miss_by_abstain: int
    miss_by_failure: int
    miss_by_wrong: int
    recall_all: Ratio
    recall_when_decided: Ratio
    precision: Ratio


@dataclass(frozen=True)
class Calibration:
    kind: str
    """``positive`` for presence signals, ``top1`` otherwise."""
    bins: tuple[tuple[float, float, int, int, float], ...]
    """(low, high, count, hits, mean_probability) per bin."""
    ece: float | None
    brier: float | None
    n: int


@dataclass(frozen=True)
class SignalMetrics:
    key: EvalKey
    scored: int
    counts: Mapping[str, int]
    coverage: Ratio
    accuracy_when_decided: Ratio
    confusion: Mapping[str, int]
    labeled_by_stratum: Mapping[str, int]
    presence: PresenceMetrics | None = None
    calibration: Calibration | None = None


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float | None
    selection_days: int
    evaluation_days: int
    selection_support: int
    evaluation_coverage: Ratio | None = None
    evaluation_recall_all: Ratio | None = None
    evaluation_precision: Ratio | None = None


@dataclass(frozen=True)
class PromotionVerdict:
    status: str
    """``eligible``, ``not_enough_data`` or ``below_threshold``."""
    unmet: tuple[str, ...]
    action: str = "manual_review_required"


@dataclass(frozen=True)
class EventMetrics:
    events: int
    labeled_events: int
    false_events: int
    latency_seconds_median: float | None
    latency_seconds_max: float | None


@dataclass
class _Row:
    weight: float
    truth: str | int
    reading: SignalReading | None
    stratum: str


@dataclass
class Report:
    signals: list[SignalMetrics] = field(default_factory=list)
    thresholds: dict[str, ThresholdChoice] = field(default_factory=dict)
    promotions: dict[str, PromotionVerdict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": [asdict(item) for item in self.signals],
            "thresholds": {key: asdict(value) for key, value in self.thresholds.items()},
            "promotions": {key: asdict(value) for key, value in self.promotions.items()},
        }


def wilson(p: float, n: float) -> tuple[float, float] | None:
    """Wilson score interval for proportion ``p`` with (effective) size ``n``."""
    if n <= 0:
        return None
    z2 = Z95 * Z95
    scale = 1 + z2 / n
    center = (p + z2 / (2 * n)) / scale
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / scale
    return (max(0.0, center - half), min(1.0, center + half))


def ratio(pairs: Iterable[tuple[float, bool]]) -> Ratio:
    """Weighted proportion of successes among ``(weight, success)`` pairs."""
    items = list(pairs)
    total = sum(weight for weight, _ in items)
    if not items or total <= 0:
        return Ratio(None, None, len(items), 0.0)
    hits = sum(weight for weight, success in items if success)
    squares = sum(weight * weight for weight, _ in items)
    n_eff = total * total / squares
    p = hits / total
    return Ratio(p, wilson(p, n_eff), len(items), n_eff)


def stratum_weights(records: Sequence[SampleRecord], signal: str) -> dict[str, float]:
    """Per-sample weights: 1/inclusion probability times the stratum's
    stored/labeled ratio for ``signal``. Strata are counted per installation and
    recipe version. Keyed by sample_id."""
    stored: Counter[tuple[str, str, str]] = Counter()
    labeled: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        group = (record.install_fingerprint, record.recipe_fingerprint, record.stratum)
        stored[group] += 1
        if signal in record.labels:
            labeled[group] += 1
    weights: dict[str, float] = {}
    for record in records:
        group = (record.install_fingerprint, record.recipe_fingerprint, record.stratum)
        if labeled[group]:
            weights[record.sample_id] = (
                (1.0 / record.inclusion_probability) * stored[group] / labeled[group]
            )
    return weights


def evaluate_signal(
    recipe: Recipe,
    records: Sequence[SampleRecord],
    key: EvalKey,
    weights: Mapping[str, float],
) -> SignalMetrics:
    """Metrics for one evaluation key."""
    spec = recipe.signal(key.signal)
    rows: list[_Row] = []
    labeled_by_stratum: Counter[str] = Counter()
    for record in records:
        if (
            record.install_fingerprint != key.install_fingerprint
            or record.recipe_fingerprint != key.recipe_fingerprint
        ):
            continue
        label = record.labels.get(key.signal)
        if label is None or not label.assessable or label.truth in (None, HUMAN_UNCERTAIN):
            continue
        labeled_by_stratum[record.stratum] += 1
        rows.append(
            _Row(
                weight=weights.get(record.sample_id, 1.0),
                truth=label.truth,
                reading=record.readings.get((key.judge_id, key.stage, key.signal)),
                stratum=record.stratum,
            )
        )

    counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    for row in rows:
        kind = _classify(row)
        counts[kind] += 1
        confusion[f"{row.truth}|{_ai_column(row.reading)}"] += 1

    coverage = ratio((row.weight, _decided(row.reading)) for row in rows)
    accuracy = ratio(
        (row.weight, row.reading.value == row.truth)
        for row in rows
        if row.reading is not None and _decided(row.reading)
    )
    presence = _presence_metrics(recipe, spec, rows) if spec.type == "presence" else None
    calibration = _calibration(recipe, spec, rows)
    return SignalMetrics(
        key=key,
        scored=len(rows),
        counts={kind: counts[kind] for kind in ("correct", "wrong", "abstained", "failed")},
        coverage=coverage,
        accuracy_when_decided=accuracy,
        confusion=dict(sorted(confusion.items())),
        labeled_by_stratum=dict(labeled_by_stratum),
        presence=presence,
        calibration=calibration,
    )


def choose_min_probability(
    recipe: Recipe,
    records: Sequence[SampleRecord],
    *,
    install_fingerprint: str,
    judge_id: str,
    signal: str,
    tz: tzinfo,
    target_precision: float,
    min_support: int = 10,
) -> ThresholdChoice:
    """Pick the smallest probability threshold whose precision lower bound meets
    the target on selection days, and report it on the other days.

    Days (in ``tz``) are sorted and alternately assigned to selection and
    evaluation, so the threshold is never measured on the data it was chosen on.
    Uses unweighted counts of ``primary`` readings that carry a distribution.
    """
    spec = recipe.signal(signal)
    if spec.type != "presence":
        raise ValueError("thresholds are supported for presence signals only")
    positive = _positive_value(recipe, spec)
    by_day: dict[Any, list[tuple[float | None, bool]]] = defaultdict(list)
    for record in records:
        if record.install_fingerprint != install_fingerprint:
            continue
        label = record.labels.get(signal)
        if label is None or not label.assessable or label.truth in (None, HUMAN_UNCERTAIN):
            continue
        reading = record.readings.get((judge_id, "primary", signal))
        p_positive = None
        if reading is not None and reading.distribution and positive in reading.distribution:
            p_positive = float(reading.distribution[positive])
        by_day[record.captured_at.astimezone(tz).date()].append(
            (p_positive, label.truth == positive)
        )

    days = sorted(by_day)
    selection = [row for day in days[0::2] for row in by_day[day]]
    evaluation = [row for day in days[1::2] for row in by_day[day]]
    chosen: float | None = None
    support = 0
    for step in range(50, 100):
        threshold = step / 100
        predicted = [(1.0, truth) for p, truth in selection if p is not None and p >= threshold]
        result = ratio(predicted)
        if len(predicted) >= min_support and result.lower is not None:
            if result.lower >= target_precision:
                chosen, support = threshold, len(predicted)
                break
    if chosen is None:
        return ThresholdChoice(None, len(days[0::2]), len(days[1::2]), 0)

    def decided_positive(p: float | None) -> bool:
        return p is not None and p >= chosen

    def decided_any(p: float | None) -> bool:
        return p is not None and p != 0.5 and max(p, 1 - p) >= chosen

    return ThresholdChoice(
        threshold=chosen,
        selection_days=len(days[0::2]),
        evaluation_days=len(days[1::2]),
        selection_support=support,
        evaluation_coverage=ratio((1.0, decided_any(p)) for p, _ in evaluation),
        evaluation_recall_all=ratio((1.0, decided_positive(p)) for p, truth in evaluation if truth),
        evaluation_precision=ratio((1.0, truth) for p, truth in evaluation if decided_positive(p)),
    )


def evaluate_promotion(
    recipe: Recipe,
    confirmed: SignalMetrics,
    *,
    probabilistic: bool = False,
    threshold: ThresholdChoice | None = None,
) -> PromotionVerdict:
    """Check the recipe's promotion rule. Never promotes by itself."""
    promotion = recipe.promotion
    if promotion is None or confirmed.presence is None:
        return PromotionVerdict("not_enough_data", ("no_promotion_rule",))
    if confirmed.key.stage != "confirmed" or confirmed.key.signal != promotion.signal:
        raise ValueError("promotion is evaluated on the confirmed stage of its signal")
    presence = confirmed.presence
    data_gaps: list[str] = []
    if presence.positives < promotion.min_positive:
        data_gaps.append("min_positive")
    if presence.negatives < promotion.min_negative:
        data_gaps.append("min_negative")
    if confirmed.labeled_by_stratum.get("ordinary", 0) < 1:
        data_gaps.append("ordinary_labels")
    if probabilistic and (threshold is None or threshold.threshold is None):
        data_gaps.append("probability_threshold")
    if data_gaps:
        return PromotionVerdict("not_enough_data", tuple(data_gaps))

    below: list[str] = []
    if (presence.recall_all.lower or 0.0) < promotion.recall_lower_bound:
        below.append("recall_lower_bound")
    if (presence.precision.lower or 0.0) < promotion.precision_lower_bound:
        below.append("precision_lower_bound")
    if probabilistic and threshold is not None:
        recall = threshold.evaluation_recall_all
        precision = threshold.evaluation_precision
        if recall is None or (recall.lower or 0.0) < promotion.recall_lower_bound:
            below.append("heldout_recall_lower_bound")
        if precision is None or (precision.lower or 0.0) < promotion.precision_lower_bound:
            below.append("heldout_precision_lower_bound")
    if below:
        return PromotionVerdict("below_threshold", tuple(below))
    return PromotionVerdict("eligible", ())


def event_metrics(events: Iterable[ChangeEvent], records: Sequence[SampleRecord]) -> EventMetrics:
    """Check confirmed events against the label of the sample that confirmed them."""
    by_sample = {record.sample_id: record for record in records}
    total = labeled = false = 0
    latencies: list[float] = []
    for event in events:
        total += 1
        latencies.append((event.confirmed_at - event.first_seen_at).total_seconds())
        record = by_sample.get(event.confirming_sample_id)
        signal = event.key[4]
        label = record.labels.get(signal) if record else None
        if label is None or not label.assessable or label.truth in (None, HUMAN_UNCERTAIN):
            continue
        labeled += 1
        if label.truth != event.to_value:
            false += 1
    return EventMetrics(
        events=total,
        labeled_events=labeled,
        false_events=false,
        latency_seconds_median=median(latencies) if latencies else None,
        latency_seconds_max=max(latencies) if latencies else None,
    )


def build_report(
    recipe: Recipe,
    records: Sequence[SampleRecord],
    *,
    tz: tzinfo | None = None,
    judge_id: str | None = None,
) -> Report:
    """Metrics for every (installation, judge, stage, signal) of one recipe."""
    records = [r for r in records if r.recipe_id == recipe.id]
    report = Report()
    installs = sorted({r.install_fingerprint for r in records})
    judges = sorted({key[0] for r in records for key in r.readings})
    if judge_id is not None:
        judges = [j for j in judges if j == judge_id]
    weights = {signal: stratum_weights(records, signal) for signal in recipe.signal_names}
    for install in installs:
        for judge in judges:
            for stage in EVAL_STAGES:
                for signal in recipe.signal_names:
                    has_readings = any(
                        (judge, stage, signal) in r.readings
                        for r in records
                        if r.install_fingerprint == install
                    )
                    if not has_readings:
                        continue
                    key = EvalKey(install, recipe.id, recipe.fingerprint, judge, stage, signal)
                    report.signals.append(evaluate_signal(recipe, records, key, weights[signal]))
            promotion = recipe.promotion
            if promotion is None:
                continue
            confirmed = next(
                (
                    m
                    for m in report.signals
                    if m.key.install_fingerprint == install
                    and m.key.judge_id == judge
                    and m.key.stage == "confirmed"
                    and m.key.signal == promotion.signal
                ),
                None,
            )
            if confirmed is None:
                continue
            probabilistic = any(
                (reading := r.readings.get((judge, "primary", promotion.signal))) is not None
                and reading.distribution is not None
                for r in records
                if r.install_fingerprint == install
            )
            threshold = None
            label = f"{install}/{judge}/{promotion.signal}"
            if probabilistic and tz is not None:
                threshold = choose_min_probability(
                    recipe,
                    records,
                    install_fingerprint=install,
                    judge_id=judge,
                    signal=promotion.signal,
                    tz=tz,
                    target_precision=promotion.precision_lower_bound,
                )
                report.thresholds[label] = threshold
            report.promotions[label] = evaluate_promotion(
                recipe, confirmed, probabilistic=probabilistic, threshold=threshold
            )
    return report


def _decided(reading: SignalReading | None) -> bool:
    return reading is not None and reading.is_decided


def _classify(row: _Row) -> str:
    reading = row.reading
    if reading is None or reading.outcome in (Outcome.INVALID, Outcome.ERROR):
        return "failed"
    if reading.outcome is Outcome.ABSTAINED:
        return "abstained"
    return "correct" if reading.value == row.truth else "wrong"


def _ai_column(reading: SignalReading | None) -> str:
    if reading is None or reading.outcome in (Outcome.INVALID, Outcome.ERROR):
        return "failed"
    if reading.outcome is Outcome.ABSTAINED:
        return "abstained"
    return str(reading.value)


def _positive_value(recipe: Recipe, spec: SignalSpec) -> str:
    if recipe.promotion is not None and recipe.promotion.signal == spec.name:
        return recipe.promotion.value
    return "present"


def _presence_metrics(recipe: Recipe, spec: SignalSpec, rows: Sequence[_Row]) -> PresenceMetrics:
    positive = _positive_value(recipe, spec)
    tp = fp = fn = tn = by_abstain = by_failure = by_wrong = 0
    for row in rows:
        kind = _classify(row)
        decided_positive = kind in ("correct", "wrong") and row.reading.value == positive
        if row.truth == positive:
            if decided_positive:
                tp += 1
            else:
                fn += 1
                if kind == "abstained":
                    by_abstain += 1
                elif kind == "failed":
                    by_failure += 1
                else:
                    by_wrong += 1
        elif decided_positive:
            fp += 1
        elif kind in ("correct", "wrong"):
            tn += 1
    positives = [row for row in rows if row.truth == positive]
    return PresenceMetrics(
        positive_value=positive,
        positives=len(positives),
        negatives=len(rows) - len(positives),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        miss_by_abstain=by_abstain,
        miss_by_failure=by_failure,
        miss_by_wrong=by_wrong,
        recall_all=ratio(
            (row.weight, _decided(row.reading) and row.reading.value == positive)
            for row in positives
        ),
        recall_when_decided=ratio(
            (row.weight, row.reading.value == positive)
            for row in positives
            if _decided(row.reading)
        ),
        precision=ratio(
            (row.weight, row.truth == positive)
            for row in rows
            if _decided(row.reading) and row.reading.value == positive
        ),
    )


def _calibration(recipe: Recipe, spec: SignalSpec, rows: Sequence[_Row]) -> Calibration | None:
    points: list[tuple[float, bool, float]] = []
    for row in rows:
        reading = row.reading
        if reading is None or not reading.distribution:
            continue
        distribution = {str(k): float(v) for k, v in reading.distribution.items()}
        truth = str(row.truth)
        brier = sum((p - (1.0 if k == truth else 0.0)) ** 2 for k, p in distribution.items())
        if spec.type == "presence":
            positive = _positive_value(recipe, spec)
            p = distribution.get(positive)
            if p is None:
                continue
            points.append((p, truth == positive, brier))
        else:
            top = max(distribution, key=lambda k: distribution[k])
            points.append((distribution[top], top == truth, brier))
    if not points:
        return None
    bins: list[tuple[float, float, int, int, float]] = []
    ece = 0.0
    for index in range(CALIBRATION_BINS):
        low, high = index / CALIBRATION_BINS, (index + 1) / CALIBRATION_BINS
        members = [
            (p, hit)
            for p, hit, _ in points
            if min(int(p * CALIBRATION_BINS), CALIBRATION_BINS - 1) == index
        ]
        if not members:
            bins.append((low, high, 0, 0, 0.0))
            continue
        hits = sum(1 for _, hit in members if hit)
        mean_p = sum(p for p, _ in members) / len(members)
        ece += len(members) / len(points) * abs(mean_p - hits / len(members))
        bins.append((low, high, len(members), hits, mean_p))
    return Calibration(
        kind="positive" if spec.type == "presence" else "top1",
        bins=tuple(bins),
        ece=ece,
        brier=sum(b for _, _, b in points) / len(points),
        n=len(points),
    )
