## Shared GPU Pipeline Notes

Implemented on 2026-03-07:

- Fixed local shared-GPU train batching to count one finished game as one episode instead of using trajectory step count.
- Reduced JVM-local shared-GPU score batch timeout so the Python host is the primary batching layer.

Known follow-up issues:

- Score batching is fragmented by strict keys: `profile_id`, `policy_key`, `head_id`, `pick_index`, `min_targets`, `max_targets`, and tensor shape.
- The score path is synchronous end-to-end, so batching depends on cross-game coincidence rather than overlap within one game thread.
- Shared GPU host scheduling can let train work delay score work.
- Score, control, and train traffic share one socket/outbound queue in the JVM client.
- Shared-GPU payload building still does a lot of allocation and copying on the flush path.
- Trainer-side request latency still needs better breakdown: local queue wait, host queue wait, service time, and response unpack time.

Suggested next order:

1. Relax score batch keys where the Python side can handle per-row metadata.
2. Make host scheduling inference-first when any score work is pending.
3. Split score traffic from control/train traffic in the JVM client.
4. Reduce payload churn and add queue-age instrumentation.
