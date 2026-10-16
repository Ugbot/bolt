//! Separate-process interoperability fixture using the actual upstream writer.
#[cfg(unix)]
mod replay;
#[cfg(unix)]
mod unix {
    use std::env;
    use std::ffi::OsString;
    use std::fs;
    use std::io::{self, ErrorKind, Write};
    use std::path::{Path, PathBuf};
    use xchannel::{Writer, WriterBuilder};

    const MAX_RECORDS: usize = 10_000;
    const MAX_PAYLOAD: usize = 4096;
    const MAX_RETAINED: u64 = 64 * 1024 * 1024;
    const MAX_DIRECTORY_ENTRIES: usize = 100_000;
    const GENERATION: u64 = 0x4732_5843_4834_0001;
    const CHANNEL_NAME: &str = "bolt-xchannel-v4-fixture";
    const FIRST_HEADER: u64 = 208;
    const HEADER_BYTES: u64 = 16;
    const ROLL_GUARD: u64 = 56;
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x100_0000_01b3;

    struct Config {
        base: PathBuf,
        count: usize,
        max_payload: usize,
        region: u64,
        roll: u64,
        explicit_roll_every: usize,
    }

    #[derive(Clone, Copy)]
    struct RecordPlan {
        length: usize,
        file_ordinal: u64,
    }

    struct Plan {
        records: [RecordPlan; MAX_RECORDS],
        last_file: u64,
        skips: u64,
        retained_bytes: u64,
        checksum: u64,
    }

    pub(crate) fn invalid(message: &str) -> io::Error {
        io::Error::new(ErrorKind::InvalidInput, message)
    }

    pub(crate) fn number(arg: &OsString) -> io::Result<u64> {
        let text = arg.to_str().ok_or_else(|| invalid("arguments must be UTF-8"))?;
        if text.is_empty() || text.len() > 20 || !text.bytes().all(|c| c.is_ascii_digit()) {
            return Err(invalid("numeric arguments must be unsigned decimal integers"));
        }
        text.parse().map_err(|_| invalid("integer overflow"))
    }

    fn parse() -> io::Result<Config> {
        let args: Vec<OsString> = env::args_os().take(9).collect();
        if !(args.len() == 7 || args.len() == 8) || args[1] != "fixture" {
            return Err(invalid("usage: fixture BASE COUNT MAX_PAYLOAD REGION_SIZE FILE_ROLL_SIZE [EXPLICIT_ROLL_EVERY]"));
        }
        let base = PathBuf::from(&args[2]);
        let count = number(&args[3])?;
        let max_payload = number(&args[4])?;
        let region = number(&args[5])?;
        let roll = number(&args[6])?;
        let explicit = if args.len() == 8 { number(&args[7])? } else { 0 };
        if count > MAX_RECORDS as u64 || max_payload > MAX_PAYLOAD as u64 || explicit > MAX_RECORDS as u64 {
            return Err(invalid("count/explicit roll interval must be <=10000; payload must be <=4096"));
        }
        Ok(Config { base, count: count as usize, max_payload: max_payload as usize,
            region, roll, explicit_roll_every: explicit as usize })
    }

    #[cfg(unix)]
    pub(crate) fn page_size() -> io::Result<u64> {
        // The fixture refuses an unknown page size instead of guessing 4096.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size <= 0 { return Err(io::Error::last_os_error()); }
        Ok(size as u64)
    }

    fn validate_geometry(config: &Config) -> io::Result<u64> {
        let page = page_size()?;
        if config.region < 4096 || config.region > MAX_RETAINED || config.region % page != 0 {
            return Err(invalid("region_size must be >=4096, <=64MiB and OS-page aligned"));
        }
        if config.roll < config.region * 2 || config.roll > MAX_RETAINED {
            return Err(invalid("file_roll_size must span >=2 regions and be <=64MiB"));
        }
        let needed = padded_record(config.max_payload) + ROLL_GUARD;
        if needed > config.region || FIRST_HEADER + needed > config.roll {
            return Err(invalid("maximum record plus padding/Roll guard cannot fit admitted geometry"));
        }
        Ok(config.roll.div_ceil(config.region) * config.region)
    }

    pub(crate) fn fresh_base(base: &Path) -> io::Result<PathBuf> {
        let text = base.to_str().ok_or_else(|| invalid("base path must be UTF-8"))?;
        if text.is_empty() || text.len() > 4096 { return Err(invalid("base path length must be 1..4096 bytes")); }
        let name = base.file_name().and_then(|v| v.to_str()).ok_or_else(|| invalid("base needs a file name"))?;
        if name.len() > 128 || !name.bytes().all(|v| v.is_ascii_alphanumeric() || b"_.+-".contains(&v)) {
            return Err(invalid("base file name must be <=128 ASCII letters/digits or _.+-"));
        }
        let parent = base.parent().filter(|v| !v.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let parent = fs::canonicalize(parent)?;
        let prefix = format!("{name}.");
        for (index, entry) in fs::read_dir(&parent)?.enumerate() {
            if index >= MAX_DIRECTORY_ENTRIES { return Err(invalid("parent directory exceeds inspection cap")); }
            let entry = entry?;
            let entry_name = entry.file_name();
            let entry_name = entry_name.to_string_lossy();
            if entry_name == name || entry_name.starts_with(&prefix) {
                return Err(io::Error::new(ErrorKind::AlreadyExists, "base or prefixed sibling already exists"));
            }
        }
        let result = parent.join(name);
        if result.to_str().is_none_or(|v| v.len() > 4096) { return Err(invalid("resolved base path must be UTF-8 and <=4096 bytes")); }
        Ok(result)
    }

    fn padded_record(length: usize) -> u64 {
        assert!(length <= MAX_PAYLOAD);
        let bytes = (HEADER_BYTES + length as u64 + 7) & !7;
        assert_eq!(bytes % 8, 0);
        bytes
    }

    fn payload_byte(index: usize, offset: usize) -> u8 {
        assert!(index < MAX_RECORDS);
        assert!(offset < MAX_PAYLOAD);
        ((index * 17 + offset * 31 + (offset >> 3)) & 255) as u8
    }

    fn message_type(index: usize) -> u16 {
        0x4000 + (index % 17) as u16
    }

    fn fold(checksum: &mut u64, bytes: &[u8]) {
        for byte in bytes { *checksum = (*checksum ^ u64::from(*byte)).wrapping_mul(FNV_PRIME); }
    }

    fn reserve_position(config: &Config, length: usize, position: &mut u64,
                        file: &mut u64, skips: &mut u64) -> io::Result<()> {
        assert!(config.region >= 4096);
        assert_eq!(*position % 8, 0);
        let needed = padded_record(length) + ROLL_GUARD;
        // At most file-roll then region-skip, or region-skip then file-roll.
        for _ in 0..3 {
            if *position + needed > config.roll { *file += 1; *position = FIRST_HEADER; }
            else if *position % config.region + needed > config.region {
                *position += config.region - *position % config.region;
                *skips += 1;
            } else { return Ok(()); }
        }
        Err(io::Error::new(ErrorKind::InvalidData, "geometry model did not converge within bound"))
    }

    fn prepare(config: &Config, file_bytes: u64, plan: &mut Plan) -> io::Result<()> {
        let mut position = FIRST_HEADER;
        for index in 0..config.count {
            if index != 0 && config.explicit_roll_every != 0 && index % config.explicit_roll_every == 0 {
                plan.last_file += 1; position = FIRST_HEADER;
            }
            let length = (73 * index + 19) % (config.max_payload + 1);
            reserve_position(config, length, &mut position, &mut plan.last_file, &mut plan.skips)?;
            plan.records[index] = RecordPlan { length, file_ordinal: plan.last_file };
            fold(&mut plan.checksum, &(index as u64).to_le_bytes());
            fold(&mut plan.checksum, &(length as u32).to_le_bytes());
            fold(&mut plan.checksum, &message_type(index).to_le_bytes());
            for offset in 0..length { fold(&mut plan.checksum, &[payload_byte(index, offset)]); }
            position += padded_record(length);
        }
        plan.retained_bytes = (plan.last_file + 1).checked_mul(file_bytes)
            .ok_or_else(|| invalid("retained byte count overflow"))?;
        if plan.retained_bytes > MAX_RETAINED { return Err(invalid("planned retained files exceed64MiB")); }
        Ok(())
    }

    fn publish(config: &Config, plan: &Plan) -> io::Result<()> {
        assert!(config.count <= MAX_RECORDS);
        assert!(config.max_payload <= MAX_PAYLOAD);
        let mut writer: Writer = WriterBuilder::new(&config.base)
            .generation(GENERATION).base_record_index(0)
            .region_size(config.region as usize).file_roll_size(config.roll)
            .channel_name(CHANNEL_NAME)?.build()?;
        if writer.generation() != GENERATION || writer.next_record_index() != 0 {
            return Err(io::Error::new(ErrorKind::InvalidData, "writer did not create expected fresh channel"));
        }
        for index in 0..config.count {
            if index != 0 && config.explicit_roll_every != 0 && index % config.explicit_roll_every == 0 {
                writer.roll_file()?;
            }
            let record = plan.records[index];
            let payload = writer.try_reserve(record.length)?;
            for (offset, byte) in payload.iter_mut().enumerate() { *byte = payload_byte(index, offset); }
            if writer.file_sequence() != record.file_ordinal {
                return Err(io::Error::new(ErrorKind::InvalidData, "upstream file ordinal disagrees with admitted model"));
            }
            writer.commit(message_type(index), record.length as u32, index as u64)?;
            if writer.next_record_index() != index as u64 + 1 {
                return Err(io::Error::new(ErrorKind::InvalidData, "upstream record index did not advance exactly once"));
            }
        }
        if writer.file_sequence() != plan.last_file {
            return Err(io::Error::new(ErrorKind::InvalidData, "final upstream ordinal mismatch"));
        }
        Ok(())
    }

    fn validate_output(config: &Config, plan: &Plan, file_bytes: u64) -> io::Result<()> {
        if plan.last_file > MAX_RECORDS as u64 { return Err(invalid("file ordinal exceeds fixture cap")); }
        let mut total = 0u64;
        for sequence in 0..=plan.last_file {
            let mut path = config.base.clone();
            if sequence != 0 {
                let name = config.base.file_name().and_then(|v| v.to_str())
                    .ok_or_else(|| invalid("non-UTF8 base file name"))?;
                path.set_file_name(format!("{name}.{sequence}"));
            }
            let bytes = fs::metadata(path)?.len();
            if bytes != file_bytes { return Err(io::Error::new(ErrorKind::InvalidData, "upstream segment allocation differs from admitted size")); }
            total = total.checked_add(bytes).ok_or_else(|| invalid("observed retained byte overflow"))?;
        }
        if total != plan.retained_bytes || total > MAX_RETAINED {
            return Err(io::Error::new(ErrorKind::InvalidData, "observed retained byte bound mismatch"));
        }
        Ok(())
    }

    pub(crate) fn json_string(text: &str) -> String {
        let mut out = String::with_capacity(6 * text.len() + 2);
        out.push('"');
        for ch in text.chars() {
            match ch {
                '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
                c if c < ' ' => {
                    const HEX: &[u8; 16] = b"0123456789abcdef";
                    out.push_str("\\u00");
                    out.push(HEX[c as usize >> 4] as char); out.push(HEX[c as usize & 15] as char);
                }
                c => out.push(c),
            }
        }
        out.push('"'); out
    }

    fn manifest(config: &Config, plan: &Plan) -> io::Result<()> {
        let stdout = io::stdout(); let mut out = stdout.lock();
        let base = config.base.to_str().ok_or_else(|| invalid("non-UTF8 resolved base"))?;
        write!(out, "{{\"manifest_version\":1,\"mode\":\"fixture\",\"xchannel_version\":\"7.0.0\",\"format_version\":4,\"generation\":{GENERATION},\"base\":{},\"channel_name\":\"{CHANNEL_NAME}\",", json_string(base))?;
        write!(out, "\"count\":{},\"max_payload\":{},\"region_size\":{},\"file_roll_size\":{},\"explicit_roll_every\":{},", config.count, config.max_payload, config.region, config.roll, config.explicit_roll_every)?;
        write!(out, "\"payload_algorithm\":\"index17_offset31_shift3_v1\",\"length_algorithm\":\"(73*i+19)%(max_payload+1)\",\"message_type_algorithm\":\"16384+i%17\",\"user_meta\":\"record_index\",\"clock\":null,\"checksum_algorithm\":\"fnv1a64_indexLE8_lengthLE4_typeLE2_payload\",\"checksum_hex\":\"{:016x}\",", plan.checksum)?;
        write!(out, "\"consumer_schedule\":\"peek_twice_then_skip_if_index_mod_7_eq_3_else_consume\",\"region_skip_count\":{},\"retained_bytes\":{},\"file_ordinal_expectation\":{{\"first\":0,\"last\":{},\"count\":{},\"record_file_ordinals\":[", plan.skips, plan.retained_bytes, plan.last_file, plan.last_file+1)?;
        for index in 0..config.count {
            if index != 0 { write!(out, ",")?; }
            write!(out, "{}", plan.records[index].file_ordinal)?;
        }
        writeln!(out, "]}}}}")?; out.flush()
    }

    pub fn run() -> io::Result<()> {
        let mut config = parse()?;
        let file_bytes = validate_geometry(&config)?;
        let mut plan = Plan { records: [RecordPlan { length: 0, file_ordinal: 0 }; MAX_RECORDS],
            last_file: 0, skips: 0, retained_bytes: 0, checksum: FNV_OFFSET };
        prepare(&config, file_bytes, &mut plan)?;
        config.base = fresh_base(&config.base)?;
        publish(&config, &plan)?;
        validate_output(&config, &plan, file_bytes)?;
        manifest(&config, &plan)
    }
}

#[cfg(unix)]
fn main() {
    let replay_mode = std::env::args_os().nth(1).is_some_and(|arg| arg == "replay");
    let result = if replay_mode { replay::run() } else { unix::run() };
    if let Err(error) = result {
        let mode = if replay_mode { "replay" } else { "fixture" };
        eprintln!("xchannel {mode} refused: {error}");
        std::process::exit(2);
    }
}

#[cfg(not(unix))]
fn main() {
    eprintln!("xchannel fixture refused: Unix is the only supported test platform");
    std::process::exit(2);
}
