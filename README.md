# statesure

Measure whether a home-state judgment can be trusted before you automate on it.

statesure takes fixed-choice judgments about your home from camera snapshots
(for example "is there a package at the door?" or "is the trash bin out?"),
checks them against human labels, and only lets a judgment drive automations
after it has passed an explicit accuracy gate that you approve.

It is designed to run next to [Frigate](https://frigate.video/) and to publish
results to Home Assistant over MQTT discovery. The judge is pluggable: a local
vision-language model behind an OpenAI-compatible API, a Jev-style decision
server such as djev, or Frigate's own state classification results.

> **Status: pre-alpha.** The offline core (phase 1) exists: recipes, reading
> normalization, confirmation, a local label store, metrics and evaluation cards.
> It does not capture images or call any model yet.

## Try the offline core

```bash
python -m pip install -e ".[dev]"
statesure validate recipes/*.yaml
statesure replay --observations observations.jsonl --recipe recipes/package_at_door.yaml
statesure report --db labels.sqlite --recipe recipes/package_at_door.yaml --tz UTC
```

These commands never open a network connection.

## Run one cycle

Copy `examples/install.example.yaml`, keep the copy private, and point it at your
Frigate (or RTSP / image URL) and a local judge:

```bash
statesure run-once --config install.yaml
```

Each run captures one frame, asks the judge, re-checks with crops when needed,
appends the structured result to an observation log (no pixels), updates the
confirmed state and, for samples chosen for review, keeps the image with an
expiry. A warning is printed if a judge or source is outside your network.

## Constrained output can bias answers

`json_schema: true` asks an OpenAI-compatible server to constrain the reply to
the recipe's schema. In a first real trial with a local 8B vision model, that
setting made the model answer "present" for three frames that a person labeled
"absent"; the same prompt without the constraint answered "absent" for all three.
Three frames prove little, but it is a reminder of why statesure exists: the
setting is off by default, it is part of the judge's identity, and you should
compare both settings with labels before trusting either.

## Label samples

```bash
statesure review --config install.yaml
```

Open the printed `http://127.0.0.1:18120/` address. The page listens on loopback
only; from another device, use an SSH tunnel. It never shows the judge's answer,
so your labels are not anchored to it. Expired images are not shown.

## What it does (planned)

- **Recipes**: a small YAML file per question, with fixed answer choices.
  No free text, no questions about people.
- **Abstain instead of guessing**: poor visibility or low confidence is recorded
  as "no answer", never as "absent".
- **Confirm over time**: a state changes only after the same value is seen in
  consecutive samples.
- **Measure with human labels**: per-stage coverage, recall (abstentions count
  as misses), precision, confidence intervals, and probability calibration.
- **Promote explicitly**: a judgment stays in shadow mode until it meets the
  recipe's thresholds and you approve it.
- **Share results without images**: evaluation cards contain counts only.

## Privacy

- Images are processed in memory. They are stored locally, with an expiry,
  only when you turn on evaluation.
- No telemetry. Nothing is sent anywhere you did not configure.
- Please never attach real home images to issues or pull requests.

## Roadmap

1. Core: recipes, normalization, confirmation, label store, metrics (offline CLI) — done
2. Judges and sources: OpenAI-compatible VLM, Jev-style API, Frigate, RTSP — in progress
3. Daemon and MQTT discovery output
4. Promotion workflow, evaluation cards, first recipes
5. First release

## License

Apache-2.0. See [LICENSE](LICENSE).
