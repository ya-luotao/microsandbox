use std::path::{Path, PathBuf};
#[cfg(not(feature = "prebuilt"))]
use std::time::SystemTime;

use microsandbox_utils::AGENTD_BINARY;
#[cfg(feature = "prebuilt")]
use microsandbox_utils::{PREBUILT_VERSION, agentd_download_url, http_client};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../utils/lib/lib.rs");
    // Invalidate the embedded agentd when its source changes.
    // This won't auto-rebuild agentd (that requires `just build-agentd`),
    // but it forces cargo to re-check that `build/agentd` is fresh.
    println!("cargo:rerun-if-changed=../agentd");
    println!("cargo:rerun-if-changed=../protocol");

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    build_agentd(&workspace_root, &out_dir);
}

fn build_agentd(workspace_root: &Path, out_dir: &Path) {
    let local = workspace_root.join("build").join(AGENTD_BINARY);
    println!("cargo:rerun-if-changed={}", local.display());

    #[cfg(feature = "prebuilt")]
    {
        let dest = out_dir.join(AGENTD_BINARY);

        // Local development recipes rebuild build/agentd before compiling msb.
        // Prefer it over a cached OUT_DIR copy so the embedded PID 1 binary
        // cannot silently lag behind the freshly built guest agent. A locally
        // built binary has no published digest to compare against, so this
        // branch is explicitly trusted; an externally supplied binary can be
        // pinned with MSB_AGENTD_SHA256, which verifies fail-closed.
        if local.is_file() {
            verify_local_agentd(&local);
            copy_agentd(&local, &dest);
            return;
        }

        if dest.exists() {
            return;
        }

        let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let url = agentd_download_url(PREBUILT_VERSION, &arch);

        // Fail-closed, mirroring the msb bundle verification in
        // sdk/rust/build.rs: a release without published checksums, a
        // checksums file without an agentd entry, or a digest mismatch all
        // fail the build before the binary is embedded.
        let expected = fetch_agentd_digest(&url).unwrap_or_else(|e| {
            panic!("failed to fetch agentd checksums: {e}");
        });
        eprintln!("Downloading {url}");
        let data = download(&url).unwrap_or_else(|e| {
            panic!("failed to download {url}: {e}");
        });
        verify_agentd_digest(&data, &expected);
        write_agentd(&data, &dest);
    }

    #[cfg(not(feature = "prebuilt"))]
    {
        if !local.exists() {
            panic!(
                "{AGENTD_BINARY} binary not found at `{}`.\n\
                 Run `just build-deps` first.",
                local.display()
            );
        }

        // Fail fast if build/agentd is stale relative to the guest source tree.
        // A warning is too easy to miss and leads to confusing runtime behavior
        // when msb embeds an older guest payload than the source implies.
        let agentd_src = workspace_root.join("crates/agentd");
        let protocol_src = workspace_root.join("crates/protocol");
        if let Ok(bin_time) = std::fs::metadata(&local).and_then(|m| m.modified())
            && newest_tree_mtime(&agentd_src)
                .into_iter()
                .chain(newest_tree_mtime(&protocol_src))
                .any(|src_time| src_time > bin_time)
        {
            panic!(
                "build/{AGENTD_BINARY} is older than crates/agentd or crates/protocol source.\n\
                 Run `just build-agentd` to rebuild the guest agent binary."
            );
        }

        let dest = out_dir.join(AGENTD_BINARY);
        copy_agentd(&local, &dest);
    }
}

fn copy_agentd(local: &Path, dest: &Path) {
    std::fs::copy(local, dest).expect("failed to copy agentd to OUT_DIR");
}

#[cfg(not(feature = "prebuilt"))]
fn newest_tree_mtime(root: &Path) -> Option<SystemTime> {
    fn walk(path: &Path, newest: &mut Option<SystemTime>) {
        let entries = match std::fs::read_dir(path) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let entry_path = entry.path();
            let meta = match entry.metadata() {
                Ok(meta) => meta,
                Err(_) => continue,
            };

            if meta.is_dir() {
                walk(&entry_path, newest);
                continue;
            }

            let modified = match meta.modified() {
                Ok(modified) => modified,
                Err(_) => continue,
            };

            match newest {
                Some(current) if *current >= modified => {}
                _ => *newest = Some(modified),
            }
        }
    }

    let mut newest = None;
    walk(root, &mut newest);
    newest
}

#[cfg(feature = "prebuilt")]
fn download(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Read as _;

    let resp = http_client().get(url).call()?;
    let mut buf = Vec::new();
    resp.into_body().into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

/// Fetch the published SHA-256 digest for the agentd binary at `agentd_url`
/// from the release's `checksums.sha256` asset. Fail-closed: an unreachable
/// checksums asset or one without an entry for this binary fails the build.
#[cfg(feature = "prebuilt")]
fn fetch_agentd_digest(agentd_url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let checksums_url = microsandbox_utils::checksums_download_url(PREBUILT_VERSION);
    let checksums = String::from_utf8(download(&checksums_url)?)?;
    let filename = agentd_url.rsplit('/').next().unwrap_or(agentd_url);
    microsandbox_utils::bundle_digest_from_checksums(&checksums, filename)
        .ok_or_else(|| format!("release checksums do not contain an entry for {filename}").into())
}

#[cfg(feature = "prebuilt")]
fn verify_agentd_digest(data: &[u8], expected: &str) {
    use sha2::{Digest as _, Sha256};

    let expected = expected.strip_prefix("sha256:").unwrap_or(expected);
    assert!(
        expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "agentd has an invalid published SHA-256 digest: {expected}"
    );
    let actual = hex::encode(Sha256::digest(data));
    assert!(
        actual.eq_ignore_ascii_case(expected),
        "agentd SHA-256 mismatch: expected {expected}, got {actual}"
    );
}

/// Gate the locally supplied `build/agentd` copy. With `MSB_AGENTD_SHA256`
/// set, the binary must match that digest (fail-closed); without it the copy
/// is trusted as a local build artifact and a warning notes the exemption.
#[cfg(feature = "prebuilt")]
fn verify_local_agentd(local: &Path) {
    println!("cargo:rerun-if-env-changed=MSB_AGENTD_SHA256");
    if let Ok(expected) = std::env::var("MSB_AGENTD_SHA256") {
        let data = std::fs::read(local)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", local.display()));
        verify_agentd_digest(&data, &expected);
    } else {
        println!(
            "cargo:warning=embedding local build/{AGENTD_BINARY} without digest verification; \
             set MSB_AGENTD_SHA256 to pin an externally supplied binary"
        );
    }
}

#[cfg(feature = "prebuilt")]
fn write_agentd(data: &[u8], dest: &Path) {
    let part_path = {
        let mut s = dest.as_os_str().to_os_string();
        s.push(".part");
        PathBuf::from(s)
    };

    std::fs::write(&part_path, data).unwrap_or_else(|e| {
        panic!("failed to write {}: {e}", part_path.display());
    });

    std::fs::rename(&part_path, dest).unwrap_or_else(|e| {
        panic!(
            "failed to rename {} to {}: {e}",
            part_path.display(),
            dest.display()
        );
    });
}
