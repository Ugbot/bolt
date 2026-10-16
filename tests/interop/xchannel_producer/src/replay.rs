//! Bounded reference-profile replay; payloads stay opaque to this test writer.
use crate::unix::{fresh_base, invalid, json_string, number, page_size};
use std::env;
use std::fs::File;
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use xchannel::{Writer, WriterBuilder};

const MAX_INPUT: usize = 256 * 1024 * 1024;
const MAX_COUNT: usize = 65_536;
const MAX_FRAME: usize = 1024 * 1024;
const MAX_RATE: u64 = 10_000_000;
const HEADER: usize = 64;
const RECORD_HEADER: usize = 16;
const FIRST_HEADER: u64 = 208;
const ROLL_GUARD: u64 = 56;
const NANOS: u64 = 1_000_000_000;
const MAX_CLOCK_POLLS: usize = 1_000_000_000;
const CHANNEL_NAME: &str = "chukonu-normalized-book-v1";

struct Config {
    base: PathBuf,
    input: PathBuf,
    rate: u64,
    region: u64,
    roll: u64,
    generation: u64,
}

struct Record { offset: usize, length: usize, schedule_ns: u64 }
struct Input {
    bytes: Vec<u8>,
    records: Vec<Record>,
    lag_ns: Vec<u64>,
    message_type: u16,
    total_payload: u64,
    max_frame: usize,
    region_skips: u64,
    file_bytes: u64,
}
struct Clock { last: u64 }
struct ResultStats {
    offered: u64, accepted: u64, elapsed_ns: u64,
    schedule_start_ns: u64, first_attempt_ns: u64, last_commit_ns: u64,
}

fn parse() -> io::Result<Config> {
    let args: Vec<_> = env::args_os().take(9).collect();
    if args.len() != 8 || args[1] != "replay" {
        return Err(invalid("usage: replay BASE INPUT RATE_UPDATES_PER_SEC REGION_SIZE FILE_ROLL_SIZE GENERATION"));
    }
    let config = Config { base: PathBuf::from(&args[2]), input: PathBuf::from(&args[3]),
        rate: number(&args[4])?, region: number(&args[5])?, roll: number(&args[6])?,
        generation: number(&args[7])? };
    if config.rate > MAX_RATE || config.generation == 0 {
        return Err(invalid("replay requires rate<=10000000 (0 unlimited) and nonzero generation"));
    }
    Ok(config)
}

fn preload(path: &Path) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() < HEADER as u64 || metadata.len() > MAX_INPUT as u64 {
        return Err(invalid("replay input must be a regular file of64 bytes through256MiB"));
    }
    let length = metadata.len() as usize;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| invalid("input allocation failed"))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    let mut extra = [0u8; 1];
    if file.read(&mut extra)? != 0 { return Err(invalid("input grew beyond admitted extent")); }
    Ok(bytes)
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("validated header extent"))
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("validated header extent"))
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("validated header extent"))
}

fn header(bytes: &[u8]) -> io::Result<(usize, usize, u16)> {
    assert!(bytes.len() >= HEADER); assert!(bytes.len() <= MAX_INPUT);
    if &bytes[..8] != b"BNFEVT1\0" || u32_at(bytes, 8) != HEADER as u32 ||
        u32_at(bytes, 12) != RECORD_HEADER as u32 || u16_at(bytes, 38) != 0 ||
        bytes[40..64].iter().any(|byte| *byte != 0) {
        return Err(invalid("unsupported replay header, flags or reserved bytes"));
    }
    let count = u64_at(bytes, 16);
    let body = u64_at(bytes, 24);
    let maximum = u32_at(bytes, 32) as usize;
    if count == 0 || count > MAX_COUNT as u64 || body != (bytes.len() - HEADER) as u64 ||
        maximum == 0 || maximum > MAX_FRAME || maximum % 64 != 0 {
        return Err(invalid("replay header count/body/maximum outside admitted bounds"));
    }
    Ok((count as usize, maximum, u16_at(bytes, 36)))
}

fn records(input: &mut Input, count: usize, maximum: usize, rate: u64) -> io::Result<()> {
    assert!(count <= MAX_COUNT); assert!(input.records.is_empty());
    let mut offset = HEADER;
    let mut actual_max = 0;
    for index in 0..count {
        if offset > input.bytes.len() || input.bytes.len() - offset < RECORD_HEADER {
            return Err(invalid("truncated replay record header"));
        }
        let length = u32_at(&input.bytes, offset) as usize;
        if u32_at(&input.bytes, offset + 4) != 0 || u64_at(&input.bytes, offset + 8) != index as u64 ||
            length == 0 || length > MAX_FRAME || length % 64 != 0 {
            return Err(invalid("invalid replay record reserved/index/length"));
        }
        offset += RECORD_HEADER;
        if length > input.bytes.len() - offset { return Err(invalid("truncated replay frame")); }
        let schedule_ns = if rate == 0 { 0 } else {
            (index as u64).checked_mul(NANOS).and_then(|v| v.checked_div(rate))
                .ok_or_else(|| invalid("offered deadline offset overflow"))?
        };
        input.records.push(Record { offset, length, schedule_ns });
        offset += length;
        input.total_payload = input.total_payload.checked_add(length as u64)
            .ok_or_else(|| invalid("payload byte sum overflow"))?;
        actual_max = actual_max.max(length);
    }
    if offset != input.bytes.len() || actual_max != maximum {
        return Err(invalid("replay trailing bytes or advertised maximum mismatch"));
    }
    input.max_frame = actual_max;
    Ok(())
}

fn load(config: &Config) -> io::Result<Input> {
    let bytes = preload(&config.input)?;
    let (count, maximum, message_type) = header(&bytes)?;
    let mut input = Input { bytes, records: Vec::new(), lag_ns: Vec::new(), message_type,
        total_payload: 0, max_frame: 0, region_skips: 0, file_bytes: 0 };
    input.records.try_reserve_exact(count).map_err(|_| invalid("record plan allocation failed"))?;
    input.lag_ns.try_reserve_exact(count).map_err(|_| invalid("lag sample allocation failed"))?;
    input.lag_ns.resize(count, 0);
    records(&mut input, count, maximum, config.rate)?;
    Ok(input)
}

fn geometry(config: &Config, input: &mut Input) -> io::Result<()> {
    let page = page_size()?;
    // Skip.length is UInt32 in v4; admit only representable region geometry.
    if config.region < 4096 || config.region > u32::MAX as u64 || config.region % page != 0 ||
        config.roll < config.region * 2 || config.roll > i64::MAX as u64 ||
        config.roll > usize::MAX as u64 {
        return Err(invalid("replay region/page/Skip width or fixed file extent is invalid"));
    }
    let remainder = config.roll % config.region;
    let allocation = if remainder == 0 { Some(config.roll) }
        else { config.roll.checked_add(config.region - remainder) };
    input.file_bytes = allocation.filter(|v| *v <= i64::MAX as u64 && *v <= usize::MAX as u64)
        .ok_or_else(|| invalid("rounded fixed extent exceeds OS limits"))?;
    let mut position = FIRST_HEADER;
    for record in &input.records {
        let padded = (16 + record.length as u64 + 7) & !7;
        let needed = padded + ROLL_GUARD;
        if needed > config.region || position.checked_add(needed).is_none_or(|end| end > config.roll) {
            return Err(invalid("replay would require an unsupported file roll"));
        }
        if position % config.region + needed > config.region {
            position += config.region - position % config.region;
            input.region_skips += 1;
            if position.checked_add(needed).is_none_or(|end| end > config.roll) {
                return Err(invalid("replay region Skip would require a file roll"));
            }
        }
        position += padded;
    }
    Ok(())
}

fn clock_ns() -> io::Result<u64> {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // A named OS clock shared with the C++ consumer, never Rust Instant/epoch time.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if time.tv_sec < 0 || time.tv_nsec < 0 || time.tv_nsec >= NANOS as libc::c_long {
        return Err(ErrorKind::InvalidData.into());
    }
    (time.tv_sec as u64).checked_mul(NANOS).and_then(|v| v.checked_add(time.tv_nsec as u64))
        .ok_or_else(|| ErrorKind::InvalidData.into())
}

impl Clock {
    fn now(&mut self) -> io::Result<u64> {
        let now = clock_ns()?;
        if now < self.last { return Err(ErrorKind::InvalidData.into()); }
        self.last = now;
        Ok(now)
    }
    fn at_or_after(&mut self, deadline: u64, paced: bool) -> io::Result<u64> {
        for _ in 0..MAX_CLOCK_POLLS {
            let now = self.now()?;
            if !paced || now >= deadline { return Ok(now); }
            std::hint::spin_loop();
        }
        Err(ErrorKind::TimedOut.into())
    }
}

fn create(config: &Config, input: &Input) -> io::Result<Writer> {
    let writer = WriterBuilder::new(&config.base).generation(config.generation)
        .base_record_index(0).region_size(config.region as usize).file_roll_size(config.roll)
        .mtu(input.max_frame as u64).wake_readers(false).channel_name(CHANNEL_NAME)?.build()?;
    if writer.generation() != config.generation || writer.file_sequence() != 0 || writer.next_record_index() != 0 ||
        std::fs::metadata(&config.base)?.len() != input.file_bytes {
        return Err(invalid("replay writer is not the admitted fresh one-file extent"));
    }
    Ok(writer)
}

fn ready(config: &Config, input: &Input) -> io::Result<()> {
    let stdout = io::stdout(); let mut out = stdout.lock();
    let base = config.base.to_str().ok_or_else(|| invalid("resolved replay base is not UTF-8"))?;
    writeln!(out, "{{\"phase\":\"READY\",\"mode\":\"replay\",\"profile\":\"BNFEVT1\",\"xchannel_version\":\"7.0.0\",\"format_version\":4,\"base\":{},\"generation\":{},\"channel_name\":\"{CHANNEL_NAME}\",\"clock\":\"CLOCK_MONOTONIC\",\"count\":{},\"message_type\":{},\"total_payload_bytes\":{},\"rate_updates_per_second\":{},\"region_size\":{},\"file_roll_size\":{},\"allocated_file_bytes\":{},\"region_skip_count\":{}}}",
        json_string(base), config.generation, input.records.len(), input.message_type, input.total_payload,
        config.rate, config.region, config.roll, input.file_bytes, input.region_skips)?;
    out.flush()
}

fn release() -> io::Result<()> {
    let mut byte = [0u8; 1];
    io::stdin().lock().read_exact(&mut byte)?;
    if byte[0] != b'G' { return Err(invalid("replay startup requires exactly one release byte G")); }
    Ok(())
}

fn publish(config: &Config, input: &mut Input, writer: &mut Writer, clock: &mut Clock) -> io::Result<ResultStats> {
    assert!(!input.records.is_empty()); assert!(input.records.len() <= MAX_COUNT);
    let epoch = clock.now()?;
    if epoch.checked_add(input.records.last().expect("nonempty admitted input").schedule_ns).is_none() {
        return Err(ErrorKind::InvalidData.into());
    }
    let mut stats = ResultStats { offered: 0, accepted: 0, elapsed_ns: 0,
        schedule_start_ns: epoch, first_attempt_ns: 0, last_commit_ns: 0 };
    for (index, record) in input.records.iter().enumerate() {
        let source = &input.bytes[record.offset..record.offset + record.length];
        let deadline = epoch + record.schedule_ns;
        // The last named-clock sample is immediately before upstream reserve.
        // Its interval includes reserve/region mapping, copy and commit.
        let stamp = clock.at_or_after(deadline, config.rate != 0)?;
        let destination = writer.try_reserve(record.length)?;
        destination.copy_from_slice(source);
        writer.commit(input.message_type, record.length as u32, stamp)?;
        if index + 1 == input.records.len() { stats.last_commit_ns = clock.now()?; }
        if index == 0 { stats.first_attempt_ns = stamp; }
        stats.offered += 1; stats.accepted += 1;
        if writer.file_sequence() != 0 || writer.next_record_index() != index as u64 + 1 {
            return Err(ErrorKind::InvalidData.into());
        }
        if config.rate != 0 {
            // at_or_after proved this subtraction nonnegative; do not clamp.
            input.lag_ns[index] = stamp.checked_sub(deadline).ok_or(ErrorKind::InvalidData)?;
        }
    }
    stats.elapsed_ns = stats.last_commit_ns.checked_sub(epoch).ok_or(ErrorKind::InvalidData)?;
    Ok(stats)
}

fn summary(config: &Config, input: &Input, stats: &ResultStats) -> io::Result<()> {
    let mut late = 0u64; let mut maximum = 0u64;
    for lag in &input.lag_ns { if *lag != 0 { late += 1; } maximum = maximum.max(*lag); }
    let formula = if config.rate == 0 { "null" }
        else { "\"schedule_start_ns+floor(index*1000000000/rate_updates_per_second)\"" };
    let stdout = io::stdout(); let mut out = stdout.lock();
    writeln!(out, "{{\"phase\":\"COMPLETE\",\"mode\":\"replay\",\"clock\":\"CLOCK_MONOTONIC\",\"count\":{},\"total_payload_bytes\":{},\"offered_updates\":{},\"accepted_updates\":{},\"elapsed_ns\":{},\"rate_updates_per_second\":{},\"schedule_lag_enabled\":{},\"schedule_lag_sample_count\":{},\"late_count\":{},\"max_schedule_lag_ns\":{},\"schedule_start_ns\":{},\"first_attempt_ns\":{},\"last_commit_ns\":{},\"offered_arrival_schedule\":{},\"deadline_formula\":{},\"file_ordinal\":0}}",
        input.records.len(), input.total_payload, stats.offered, stats.accepted, stats.elapsed_ns,
        config.rate, config.rate != 0, if config.rate == 0 { 0 } else { input.lag_ns.len() }, late, maximum,
        stats.schedule_start_ns, stats.first_attempt_ns, stats.last_commit_ns, config.rate != 0, formula)?;
    out.flush()
}

pub(crate) fn run() -> io::Result<()> {
    let mut config = parse()?;
    let mut input = load(&config)?;
    geometry(&config, &mut input)?;
    let mut clock = Clock { last: clock_ns()? };
    config.base = fresh_base(&config.base)?;
    let mut writer = create(&config, &input)?;
    ready(&config, &input)?;
    release()?;
    let stats = publish(&config, &mut input, &mut writer, &mut clock)?;
    summary(&config, &input, &stats)
}
