# Column codec chains: prototype and first numbers (2026-11-16)

Branch `codec/column-chain` (off `ts/unified` 52ec514). Prototype only: the bytes are not an MSEG page format and are
not stable. Engineering KB was unreachable when this was written; prior decisions read from
`marbledb/docs/research/layout-decision-2026-10.md` (U10 lightweight hot / zstd cold only, U28 no trial compression,
D5 dictionary scope, §6.4 encoding menu).

## What exists

`include/bolt/codec/bolt_column_codec.h` + `bolt_codec_stream.h`: a column declares an ordered chain of stages
(ClickHouse `CODEC(...)` style), e.g. `delta,for,zstd(3)` or `template,dict,for,zstd`. Each stage accepts one stream
kind (ints, strings, bytes); a stream takes the first remaining stage that accepts it and its children take the rest.
The output is self-describing, so decode needs no chain.

| stage | accepts | children |
|---|---|---|
| `delta` | ints | ints (base kept apart) |
| `for` | ints | bytes: 1,024-value FastLanes blocks, own ref + width each |
| `rle` | ints | values, run lengths (skipped if runs > n/2) |
| `dict` | strings | sorted strings, int codes (skipped if NDV > n/2) |
| `front` | strings | shared-prefix length with the previous row, suffix length, suffix bytes |
| `template(p)` | strings | templates, int variables, string variables. p bit 0: only all-digit tokens are variables; bit 1: `. - _ @ +` also delimit |
| `lz4`, `zstd(level)`, `zstdself` | bytes | kept only if smaller. `zstd` = libzstd, `zstdself` = bolt's own encoder |

Tests: `test_bolt_column_codec` (5 cases): 13 int chains and 16 string chains x 8 sizes x 5-6 data shapes round-trip;
truncated input refuses; a block codec alone reaches string and int payloads. Negative control: dropping the
leading-zero rule of the template split fails the round trip.

Bench: `bench_column_codec <file> [--chunk R] [--str chains] [--int chains]`, 1 thread, every chunk round-trip checked.

## Numbers (M5 Max, 1 thread, load average 3-4.5 from other lanes, best of 3)

### Rally `http_logs`, first 500,000 lines, 133.5 raw bytes per line, 65,536-row chunks

Best chain per column, bytes per row:

| column | raw | lightweight only | with a block codec |
|---|---|---|---|
| `@timestamp` | 8.0 | 0.08 `rle,delta,for` | 0.07 `delta,for,zstd` |
| `clientip` | 9.3 | 1.64 `template,dict,for` | 1.43 `dict,for,zstd` |
| `request` | 40.0 | 3.89 `template,dict,for` | 1.68 `dict,for,zstd` |
| `status` | 8.0 | 0.38 `rle,for` | 0.23 `rle,delta,for,zstd` |
| `size` | 8.0 | 2.57 `for` | 1.74 `zstd` |
| **sum** | | **8.56** | **5.15** |

At 4,096-row chunks the sums are 13.05 and 6.00. VictoriaLogs stores this corpus at 9.6 bytes per line; `otel_logs`
today at 360-385. The sums are payload only: no primary-key index, LSN runs, word filter or page stats, and none of
`otel_logs`' other columns (`log_key`, `entity_id`, ...).

The whole line as one string (132.5 B):

| chain | bytes/row | enc MB/s | dec MB/s |
|---|---|---|---|
| `lz4` | 22.28 | 1440 | 2579 |
| `zstdself` | 20.87 | 649 | 1314 |
| `zstd` | 11.27 | 1579 | 2980 |
| `front,for,zstd` | 9.97 | 1526 | 2888 |
| `zstd(9)` | 8.76 | 238 | 4304 |
| `template,dict,for,zstd` | 8.55 | 266 | 639 |
| `template,dict,for` (no block codec) | 18.30 | 288 | 688 |

### macOS `/var/log/install.log`, 88,136 free-text lines, 146.7 raw bytes per line

| chain | bytes/row | enc MB/s | dec MB/s |
|---|---|---|---|
| `zstdself` | 15.42 | 957 | 2041 |
| `template,dict,for,zstd` | 8.20 | 258 | 580 |
| `zstd` | 6.76 | 2305 | 4925 |
| `front,for,zstd` | 5.50 | 2312 | 5086 |
| `zstd(9)` | 5.36 | 412 | 6863 |

## What the numbers say

- No one chain wins. `request`: plain `dict` beats the template split (1.68 vs 2.33). Whole HTTP lines: the template
  split wins (8.55 vs 11.27). Free text: the template split loses to plain zstd (8.20 vs 6.76) and the prefix delta
  against the previous row wins (5.50). The per-column chain is the requirement, not a nicety.
- Lightweight stages alone reach 8.56 bytes per line on the split columns, under VictoriaLogs' 9.6, with no general
  codec.
- bolt's own zstd encoder (raw literals, predefined FSE tables) gives about half libzstd's ratio (20.87 vs 11.27,
  15.42 vs 6.76). A block codec stage needs Huffman literals in it, or libzstd.
- Chunk size matters most for `dict` and `template` (request 1.68 -> 2.76 at 4,096 rows): the dictionary is paid per
  chunk. This is the case for a dictionary wider than a page (D5's table-level symbols, or one per stripe).
- Encode speed of `dict` / `template` (120-300 MB/s) is the prototype (`std::sort`, byte-at-a-time split), not a limit.

## Not done

- Not an MSEG page format; no predicate evaluation on encoded data; no chooser from stats (chains are declared).
- No ALP, FSST, Gorilla, zstd dictionary; no nulls / validity; ints are int64 only; no floats.
- Decode is bounds-checked and truncation is tested; corrupt (bit-flipped) input is not fuzzed.
- Not measured: ClickHouse on the same data, metrics samples, TPC-H, triples.
