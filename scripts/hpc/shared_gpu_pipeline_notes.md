## Shared GPU Pipeline Notes

Implemented on 2026-03-07:

- Fixed local shared-GPU train batching to count one finished game as one episode instead of using trajectory step count.
- Reduced JVM-local shared-GPU score batch timeout so the Python host is the primary batching layer.
- Split shared GPU host execution into separate inference and learner lanes with independent `ProfileContext` instances.
- Added async learner-to-inference weight publish/reload so score traffic no longer waits behind learner scheduling.

Known follow-up issues:

- Score batching is fragmented by strict keys: `profile_id`, `policy_key`, `head_id`, `pick_index`, `min_targets`, `max_targets`, and tensor shape.
- The score path is synchronous end-to-end, so batching depends on cross-game coincidence rather than overlap within one game thread.
- Score, control, and train traffic share one socket/outbound queue in the JVM client.
- Shared-GPU payload building still does a lot of allocation and copying on the flush path.
- Trainer-side request latency still needs better breakdown: local queue wait, host queue wait, service time, and response unpack time.
- Separate learner/inference lanes may need rate limits or fairness controls if one side starts to dominate the GPU.

Suggested next order:

1. Relax score batch keys where the Python side can handle per-row metadata.
2. Split score traffic from control/train traffic in the JVM client.
3. Add explicit fairness/rate controls between learner and inference lanes if the shared GPU begins oscillating.
4. Reduce payload churn and add queue-age instrumentation.
