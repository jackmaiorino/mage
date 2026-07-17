# Phase-0 corpus content locks

`corpus_content_locks_v1.json` is the tracked content identity for the two
formal local Phase-0 replay corpora. It covers `burn_mirror_v6` and
`rally_mirror_v2`. Older corpora remain unblocked by this content-lock gate and
are explicitly reported as untracked by `replay_burn_v2`; they are still
subject to the existing Java-oracle provenance gate.

Schema v1 uses this exact algorithm:

1. The trace set is every recursively discovered file whose basename starts
   with `game_` and whose extension is exactly lowercase `.txt`, matching the
   replay loader. Paths are relative to the corpus root, use `/` separators,
   must be UTF-8, and are sorted by their raw UTF-8 bytes. Missing or additional
   trace paths fail the lock.
2. `size` is the raw byte length. `sha256` is SHA-256 over the raw file bytes,
   encoded as 64 lowercase hexadecimal characters. The same rules cover the
   root `manifest.json`.
3. The aggregate input is a byte string. For each sorted trace append
   `trace`, NUL, path, NUL, base-10 size with no leading zero, NUL, lowercase
   SHA-256, LF. After all traces append `manifest`, NUL, `manifest.json`, NUL,
   size, NUL, SHA-256, LF. There is no BOM, prefix, suffix, or final data beyond
   that LF. `aggregate_sha256` is the lowercase SHA-256 of this byte string.
4. Replay additionally requires the manifest's `corpus` to equal the lock's
   `manifest_corpus` and its `status` to equal `LOCKED` before hashing or
   parsing any trace.

The lock deliberately excludes `live_checkpoints/`: those files are supporting
generation evidence, not replay inputs. The tracked metadata makes local
evidence tamper-evident; the large corpus bytes are still outside Git. An
independent clean clone therefore still needs durable, digest-verified corpus
storage and retrieval before it can reproduce these reference gates.
