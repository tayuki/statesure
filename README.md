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

> **Status: pre-alpha.** Nothing is usable yet. The design is being written in
> the open; see the roadmap below.

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

1. Core: recipes, normalization, confirmation, label store, metrics (offline CLI)
2. Judges and sources: OpenAI-compatible VLM, Jev-style API, Frigate, RTSP
3. Daemon and MQTT discovery output
4. Promotion workflow, evaluation cards, first recipes
5. First release

## License

Apache-2.0. See [LICENSE](LICENSE).
