# Security LLM real-time reasoning benchmark plan

This is a separately budgeted evaluation track. The incremental executable
foundation is in [`security/`](security/); dataset acquisition and full public
plus unseen campaigns remain outstanding. It is not part of the
completed Gemma quality score or synthetic serving benchmark.

Qwen3.6 35B is skipped. The active model matrix is Gemma 4 E4B, 12B, and 26B
A4B, run as separate campaigns with identical telemetry and settings.

## Research question

Can locally deployable 26–35B models act as a real-time reasoning layer above
conventional security telemetry: detecting weak signals, correlating events
across sources, reconstructing attacks, and warning defenders early enough to
change the outcome?

The LLM is the reasoning engine, not the log parser. Fast rules, IOC matching,
statistics, baselines, and an entity/time correlator should reduce raw
telemetry into an evolving incident state before model analysis.

## Capability ladder

1. **Understand:** recognise security events such as abnormal PowerShell.
2. **Detect:** identify anomalous authentication or process behaviour.
3. **Correlate:** connect identity, endpoint, DNS, and network events across
   time and entities.
4. **Investigate:** reconstruct a credential-theft or lateral-movement chain.
5. **Anticipate:** predict likely next attacker actions.
6. **Prevent:** recommend a justified action before damage occurs.

Levels 3–6 are the primary research contribution; conventional detection
already covers many L1/L2 cases.

## Proposed architecture

```text
Endpoint / identity / network / DNS / cloud telemetry
                         │
                 normalisation layer
                         │
             rules, Sigma, IOC, baselines, anomaly
                         │
                  event correlator
                         │
                 entity + time graph
                         │
                  context builder
                         │
              20–100 useful observations
                         │
                    26–35B LLM
                         │
       detect · investigate · predict · recommend
                         │
                    incident state
                         │
                  alert / response
```

Maintain state instead of repeatedly sending thousands of raw events:

```json
{
  "entity": "HOST-127",
  "risk": 91,
  "observations": [
    "PowerShell launched",
    "Encoded command executed",
    "External domain contacted",
    "User authenticated to SERVER-32"
  ],
  "hypotheses": ["Possible compromise followed by lateral movement"],
  "evidence": ["E1042", "E1045", "E1078", "E1081"]
}
```

## Dataset stages

### Stage A — OTRF controlled laboratory benchmark

Start with 10–20 benign and malicious scenarios covering PowerShell,
credential activity, persistence, lateral movement, suspicious process chains,
C2, and remote execution. Use [OTRF Security
Datasets](https://github.com/OTRF/Security-Datasets) and MITRE ATT&CK as the
common technique taxonomy.

### Stage B — LANL enterprise-scale correlation

Replay the [LANL multi-source enterprise
events](https://csr.lanl.gov/data/cyber1/) over 58 consecutive days. Hide attack
signals among long benign periods and test whether the model finds a weak
indicator, authentication anomaly, process activity, network event, and
lateral movement as one developing incident.

### Stage C — DARPA OpTC attack reconstruction

Use [DARPA OpTC data](https://github.com/FiveDirections/OpTC-data) to test
multi-stage reconstruction: initial access, discovery, execution, credential
access, lateral movement, persistence, collection, and exfiltration.

## Leakage and evaluation tracks

Public datasets provide reproducibility, but their schemas or examples may
have appeared in model training. Maintain two tracks:

- **Public:** OTRF, LANL, and OpTC for repeatable comparisons.
- **Unseen:** fresh attack telemetry generated in an isolated cyber range.

For public data, remap usernames, hosts, IPs, timestamps, filenames, labels,
attack names, and ground-truth fields. Never tell the model which dataset it is
analysing.

## Experimental modes

Compare architectures as well as models:

- **Raw events:** every event directly to the LLM (control, expected to be
  noisy and expensive).
- **Windowed:** 5-second, 30-second, and 2-minute event windows.
- **Triggered:** rules or anomaly detection select interesting events.
- **Stateful:** new events plus incident state and entity history produce an
  updated state object.
- **Agentic investigation:** the model can call bounded tools such as
  `get_process_tree(host)`, `get_auth_history(user)`, `get_connections(host)`,
  `get_dns_history(host)`, `get_asset_information(host)`, and
  `get_related_alerts(entity)`.

Keep system prompt, decoding, context, schema, replay speed, and tool access
identical across candidate models. Test quantization separately from model
intelligence (BF16/FP16, FP8, INT8, and INT4 where supported).

## Structured model contract

Require observations, hypotheses, evidence IDs, ATT&CK techniques, related
entities, recommended actions, missing information, predicted next actions,
status, risk score, and confidence in JSON. The model must distinguish evidence
from inference, avoid invented entities, update hypotheses, and abstain when
evidence is insufficient.

Telemetry is untrusted data: include prompt-injection cases such as malicious
command lines or HTTP headers that attempt to override the system instruction.

## Core measurements

- Attack detection precision, recall, and F1.
- Event-correlation precision/recall and attack-chain reconstruction.
- **TTD:** time from first attacker action to the first justified suspicious
  or malicious classification.
- False alerts per hour during long benign replays.
- Evidence grounding rate, unsupported-claim rate, and contradicted-claim rate.
- Appropriate abstention rate and prompt-injection resistance.
- Top-1/top-3 next-action accuracy and ATT&CK-stage prediction.
- Preventability window: time between detection and operational impact.
- TTFT, TPOT, end-to-end analysis latency, output/total tokens per second,
  concurrent investigations, events processed per second, cost per incident,
  GPU utilisation, and failure rate.

Stress replay at 0.5×, 1×, 5×, 10×, 20×, and 50× speed. Compare raw,
filtered, correlated, and stateful context sizes to test whether better context
beats more context. Test long incidents with compact incident memory and add
distractor ratios from 1:10 through 1:10,000.

## Initial targets

These are experimental starting points, not industry guarantees:

- attack detection ≥85%; detection precision ≥80%;
- evidence grounding ≥95%; major hallucination <2%;
- attack-chain reconstruction ≥80%; correlation F1 ≥0.80;
- p95 triggered-analysis latency <5 seconds;
- appropriate abstention ≥90%; prompt-injection success approximately 0%.

Report separate **Security Intelligence** and **Operational** scorecards. Do
not hide the trade-off in a single score: the best deployment is likely on a
Pareto frontier between reasoning quality, prevention window, throughput,
concurrency, context efficiency, and cost per incident.

## Phased rollout

1. Build dataset parser, timestamp replay, canonical JSON, LLM API adapter,
   structured evaluator, and raw artifact capture.
2. Run a basic shootout on 10 attacks and 10 benign scenarios.
3. Add temporal replay and risk-vs-time/TTD measurement.
4. Add multi-source identity, endpoint, network, and DNS correlation.
5. Add stateful incident and entity memory.
6. Add bounded investigation tools and measure evidence selection.
7. Add next-action prediction and intervention/prevention scoring.
8. Stress test replay speed, signal-to-noise, context compression, and cost.
9. Validate the winning architecture on unseen cyber-range telemetry.

The first implementation should be: **OTRF → 10 attack scenarios → timestamp
replay → stateful 30-second correlation → three Gemma 4 models → structured
JSON → automated scoring**, followed by LANL and OpTC evaluation.

## References

- [OTRF Security Datasets](https://github.com/OTRF/Security-Datasets)
- [MITRE ATT&CK](https://attack.mitre.org/)
- [LANL Comprehensive Cyber-Security Events](https://csr.lanl.gov/data/cyber1/)
- [DARPA OpTC data](https://github.com/FiveDirections/OpTC-data)
