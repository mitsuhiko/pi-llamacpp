# pi-llamacpp

Pi provider extension for running Pi self-managed local
[llama.cpp](https://github.com/ggml-org/llama.cpp) inference.

The extension registers Qwen3.6 GGUF models under the `llamacpp` provider,
downloads/builds a matching llama.cpp runtime and downloads the selected GGUF on
first use, starts `llama-server`, and stops it automatically when pi shuts down.

## Models

Currently registered:

- `llamacpp/qwen-3.6-dense-2bit` (27B dense)
- `llamacpp/qwen-3.6-dense-4bit` (27B dense)
- `llamacpp/qwen-3.6-dense-8bit` (27B dense)
- `llamacpp/qwen-3.6-moe-2bit` (35B-A3B MoE)
- `llamacpp/qwen-3.6-moe-4bit` (35B-A3B MoE)
- `llamacpp/qwen-3.6-moe-8bit` (35B-A3B MoE)

The model names describe the architecture:

- `dense` is the Qwen3.6 27B dense model. All parameters participate in every
  token, which makes compute and memory use more direct and predictable.
- `moe` is the Qwen3.6 35B-A3B Mixture-of-Experts model. It has about 35B total
  parameters, but routes each token through only a small active subset of
  experts (about 3B active parameters). MoE can offer more total capacity for a
  similar amount of active compute, but the full expert weights still need to be
  stored and loaded.

The `moe` (35B-A3B) models are downloaded from
[`havenoammo/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/havenoammo/Qwen3.6-35B-A3B-MTP-GGUF)
at revision `44ce525026e7e7d0e0915dc1bf83a783c813e75a`, and the `dense`
(27B) models are downloaded from
[`froggeric/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF)
at revision `431204640c8511573e61a7964a12cc452114a223`. Pinning the
revisions keeps downloads reproducible if upstream `main` moves; set
`LLAMACPP_QWEN_35B_A3B_REVISION`, `LLAMACPP_QWEN_27B_REVISION`, or
`LLAMACPP_QWEN_REVISION` to override.
These files need llama.cpp MTP/NextN support, so the default runtime path builds
a pinned snapshot of [llama.cpp pull request #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
instead of using the stock binary release.

## Install

```sh
pi install https://github.com/mitsuhiko/pi-llamacpp
```

For local development from this checkout:

```sh
./install-pi-extension-local.sh
```

Then restart Pi or run `/reload`.

## Runtime layout

Runtime state is kept under `~/.pi/llamacpp`:

- `source/`: pinned llama.cpp source snapshots built locally (default: [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) snapshot for MTP/NextN support)
- `runtime/`: extracted llama.cpp release archives when `LLAMACPP_RUNTIME_KIND=release`
- `downloads/`: release archives and resumable `.part` files
- `models/havenoammo/Qwen3.6-35B-A3B-MTP-GGUF/`: cached `moe` (35B-A3B) GGUF model files
- `models/froggeric/Qwen3.6-27B-MTP-GGUF/`: cached `dense` (27B) GGUF model files
- `clients/`: active Pi process leases
- `server.json`: managed `llama-server` state
- `log`: download/extract/server/watchdog log

The managed server binds to a random localhost port by default and records the
active endpoint in `server.json`. Set `LLAMACPP_PORT` only if you explicitly
want a fixed port.

## Debugging

Use `/llamacpp` inside Pi to show the live llama.cpp log, `/llamacpp status` for
paths/status, and `/llamacpp stop` to stop the managed server when no other
leases are active.
