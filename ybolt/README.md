# bolt::ybolt

> Bolt-arena-backed binding for the [ycpp](https://github.com/Ugbot/ycpp)
> Yjs CRDT runtime.

A header-only INTERFACE library that plugs ycpp's templated `Allocator`
policy into `bolt::Arena` (~3ns/alloc, no vtable). Every allocation
inside `Doc<A>`, `YArray<A>`, `YText<A>`, `StateVector<A>`, etc.
monomorphises through bolt's arena at the call site.

## Usage

```cpp
#include "bolt/ybolt.h"

bolt::Arena arena;
bolt::ybolt::Doc doc = bolt::ybolt::make_doc(arena, /*client_id=*/1);

doc.text_append("body", "Hello, ");
doc.text_append("body", "world!");

// Sync diffs the same way as raw ycpp:
bolt::Arena scratch;
bolt::ybolt::BoltArenaAllocator scratch_alloc{&scratch};
bolt::ybolt::StateVector peer_sv{&scratch_alloc};
peer.state_vector(&peer_sv);

uint8_t  buf[4096];
ycpp::Writer w{buf, sizeof(buf)};
bolt::ybolt::encode_diff_v1(doc, &peer_sv, &w);
peer.apply_update_v1({buf, w.pos()});
```

## Type aliases

Every ycpp template is pre-instantiated against `BoltArenaAllocator`:

```
bolt::ybolt::Doc / YMap / YArray / YText / StateVector
bolt::ybolt::DeleteSet / StructStore / DecodedUpdate
bolt::ybolt::Awareness / UndoManager / SubDocRegistry
bolt::ybolt::Envelope / MessageKind
```

Plus pinned helpers: `make_doc`, `decode_update_v1`, `encode_diff_v1`,
`emit_sync_step1` / `step2` / `update`, `apply_sync_message`.

## A note on updateV2

ybolt inherits ycpp's wire-format support: **updateV1 only**. The
`apply_update_v2` / `encode_diff_v2` symbols exist as stubs in ycpp;
they return `Status::kUnsupportedFormat`.

The decision is deliberate, not a TODO. Yjs JS defaults to v1
(`Y.encodeStateAsUpdate` is v1); v2 is opt-in via
`Y.encodeStateAsUpdateV2`. v2's only benefit is ~30–50% smaller
wire payloads — no new CRDT semantics, no new content kinds.
Implementing it means ~1500 LOC of bit-fiddly RLE / diff-RLE /
optional-RLE / string-pool stream encoders with no behavioural payoff
for the v1-default ecosystem we ship into (browser ↔ gestaltd over
local WebSocket / SSE, where size compression is irrelevant).

We'll land v2 when a real use case shows up. If you need it, open an
issue.

## Built + tested

Both ycpp and ybolt are verified against the real `yjs` npm package
via `extern/ycpp/tools/yjs_interop/test.js`. Drive `ybolt_cli` instead
of `ycpp_cli` and the same 13/13 assertions go green.

## License

MIT (inherits from ycpp + bolt).
