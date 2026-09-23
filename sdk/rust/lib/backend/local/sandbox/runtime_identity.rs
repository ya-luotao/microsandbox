//! Identity-checked view of the process a run row records.
//!
//! A run row keeps the runtime's PID, but the row can outlive the process (SIGKILL, host
//! crash, container kill) and the kernel can hand that PID to an unrelated process. Every
//! local-backend decision that reads a recorded PID — is this row still live, may this
//! process be signalled — goes through [`RecordedRuntime`], which requires proof that the PID
//! still names the runtime that wrote the row. A positive mismatch is dead for liveness and is
//! never signalled. An unprovable identity stays conservative: the row is treated as live and
//! the process may be signalled, exactly as before identity checks existed.
//!
//! A false `Recycled` is the worst outcome: the row is reconciled away, the name is reused and
//! two VMs share one storage. Every rule below therefore needs positive evidence before it
//! calls a live process a stranger. Evidence, in order of strength:
//!
//! | lifecycle descriptor (fd 99)                  | creation vs `started_at` | executable | verdict                    |
//! |-----------------------------------------------|--------------------------|------------|----------------------------|
//! | same lock file (dev/ino), or same lock name   | any                      | any        | `Live(LifecycleLock)`      |
//! | another sandbox's lifecycle lock              | any                      | any        | `Recycled`                 |
//! | missing, unreadable, or not a lifecycle lock  | `started_at` missing     | any        | `Live(Unverified)`         |
//! | missing, unreadable, or not a lifecycle lock  | created in time          | any        | `Live(StartedBeforeRun)`   |
//! | missing, unreadable, or not a lifecycle lock  | created too late         | msb        | `Live(Unverified)`         |
//! | missing, unreadable, or not a lifecycle lock  | created too late         | other/none | `Recycled`                 |
//!
//! The lock *name* comparison exists for launchers in different mount namespaces (one
//! container generation draining while the next starts): the lock file's inode differs per
//! run-directory overlay, but the path the runtime holds renders with the same
//! `<sha256(name)>.lock` file name. A descriptor on a *different* lifecycle lock is the one
//! definitive recycled-PID signal the time rule cannot give, because the new occupant is then
//! itself an `msb` runtime. Conversely a too-late creation time alone never condemns an `msb`
//! process: the Linux time anchor moves with wall-clock steps, and a legacy runtime without
//! the inherited descriptor has nothing stronger to offer.

use std::path::Path;

use chrono::NaiveDateTime;

#[cfg(unix)]
use super::super::control::identity::ProcessIdentity;
use crate::MicrosandboxResult;

//--------------------------------------------------------------------------------------------------
// Types
//--------------------------------------------------------------------------------------------------

/// What a run row's recorded PID names right now.
#[cfg_attr(windows, allow(dead_code))]
pub(super) enum RecordedRuntime {
    /// No live process holds the PID (missing, or a zombie).
    Dead,

    /// A live process holds the PID but is provably not the recorded runtime.
    Recycled,

    /// A live process that is, or cannot be shown not to be, the recorded runtime.
    Live(RuntimeProcess),
}

/// How a live process was tied to its run row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(windows, allow(dead_code))]
pub(super) enum Proof {
    /// The process holds this sandbox's lifecycle lock on the runtime's inherited descriptor.
    LifecycleLock,

    /// The process was created no later than the row's `started_at`, within timer slack.
    StartedBeforeRun,

    /// Neither proof was available; the process is treated as the runtime.
    Unverified,
}

/// Signals the lifecycle paths deliver to a verified runtime process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RuntimeSignal {
    /// Request graceful termination (SIGTERM).
    Terminate,

    /// Force termination (SIGKILL).
    Kill,

    /// Trigger the legacy drain path (SIGUSR1).
    #[cfg(unix)]
    Drain,
}

/// A live process bound to the run row that recorded it, with a handle that follows the
/// process instance rather than its PID wherever the platform offers one.
pub(super) struct RuntimeProcess {
    pid: i32,
    proof: Proof,
    /// `None` only when the process is live but its kernel identity is unreadable.
    #[cfg(unix)]
    identity: Option<ProcessIdentity>,
}

/// What the process's inherited lifecycle descriptor points at.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LockEvidence {
    /// This sandbox's lifecycle lock, by inode or by lock file name.
    Matches,

    /// A lifecycle lock that belongs to a different sandbox name.
    OtherSandbox,

    /// No descriptor, an unreadable one, or one that is not a lifecycle lock at all.
    None,
}

/// How the process's creation time relates to the row's `started_at`.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimeEvidence {
    /// No `started_at`, or the creation time could not be read.
    Unknown,

    /// Created no later than `started_at` plus slack: consistent with being the runtime.
    Plausible,

    /// Created after `started_at` plus slack: inconsistent with being the runtime.
    TooLate,
}

/// Whether the process runs the `msb` binary.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExeEvidence {
    Msb,
    Other,
    Unknown,
}

//--------------------------------------------------------------------------------------------------
// Methods
//--------------------------------------------------------------------------------------------------

impl RecordedRuntime {
    /// Decide what `pid` names for the run row that recorded it.
    ///
    /// `started_at` is the row's own timestamp; `lifecycle` the sandbox's lifecycle lock file
    /// as this launcher would create it. See the module documentation for the decision table.
    /// Observation failures never yield [`RecordedRuntime::Recycled`] and never fail the call.
    pub(super) fn inspect(
        pid: Option<i32>,
        started_at: Option<NaiveDateTime>,
        lifecycle: Option<&Path>,
    ) -> Self {
        let Some(pid) = pid.filter(|pid| *pid > 0) else {
            return Self::Dead;
        };
        #[cfg(unix)]
        {
            Self::inspect_unix(pid, started_at, lifecycle)
        }
        #[cfg(windows)]
        {
            // Windows keeps its existing rule here; `crate::sandbox::reap` owns identity-checked
            // termination of leaked runtimes there.
            let _ = (started_at, lifecycle);
            if super::LocalBackend::pid_is_alive(pid) {
                Self::Live(RuntimeProcess {
                    pid,
                    proof: Proof::Unverified,
                })
            } else {
                Self::Dead
            }
        }
    }

    #[cfg(unix)]
    fn inspect_unix(pid: i32, started_at: Option<NaiveDateTime>, lifecycle: Option<&Path>) -> Self {
        use microsandbox_control_client::ControlClientError;

        let identity = match ProcessIdentity::capture(pid) {
            Ok(identity) => identity,
            Err(error) => {
                // Gone or a zombie is dead. A live process whose kernel record this user cannot
                // read (hidden procfs, another user's process) is unprovable, not dead.
                if !super::LocalBackend::pid_is_alive(pid) {
                    return Self::Dead;
                }
                if !matches!(error, ControlClientError::RuntimeChanged) {
                    tracing::debug!(
                        pid,
                        error = %error,
                        "recorded runtime PID is live but its identity is unreadable; treating it as the runtime"
                    );
                }
                return Self::Live(RuntimeProcess {
                    pid,
                    proof: Proof::Unverified,
                    identity: None,
                });
            }
        };

        let lock = lifecycle.map_or(LockEvidence::None, |lifecycle| {
            lock_evidence(pid, lifecycle)
        });
        let time = match started_at {
            None => TimeEvidence::Unknown,
            Some(started_at) => match identity.created_unix_micros() {
                Ok(created) => {
                    if crate::runtime::reap::creation_may_belong_to_run(
                        created,
                        started_at.and_utc().timestamp_micros(),
                    ) {
                        TimeEvidence::Plausible
                    } else {
                        TimeEvidence::TooLate
                    }
                }
                Err(error) => {
                    tracing::debug!(
                        pid,
                        error = %error,
                        "cannot read the recorded runtime's creation time; treating it as the runtime"
                    );
                    TimeEvidence::Unknown
                }
            },
        };
        // The executable only matters when the time rule would otherwise condemn the process.
        let exe = if time == TimeEvidence::TooLate && lock == LockEvidence::None {
            exe_evidence(pid)
        } else {
            ExeEvidence::Unknown
        };

        // The reads above are attributed to the pinned instance only while it is still alive;
        // a process that exited meanwhile may have handed its PID to a new occupant.
        match identity.has_exited() {
            Ok(false) => {}
            Ok(true) => return Self::Dead,
            Err(error) => {
                tracing::debug!(pid, error = %error, "cannot re-verify the recorded runtime process");
            }
        }
        match verdict(lock, time, exe) {
            Some(proof) => Self::Live(RuntimeProcess {
                pid,
                proof,
                identity: Some(identity),
            }),
            None => Self::Recycled,
        }
    }

    /// The live process, if any.
    pub(super) fn live(self) -> Option<RuntimeProcess> {
        match self {
            Self::Live(process) => Some(process),
            Self::Dead | Self::Recycled => None,
        }
    }

    /// Whether the row still has a runtime behind it.
    pub(super) fn is_live(&self) -> bool {
        matches!(self, Self::Live(_))
    }
}

impl RuntimeProcess {
    pub(super) fn pid(&self) -> i32 {
        self.pid
    }

    pub(super) fn proof(&self) -> Proof {
        self.proof
    }

    /// Deliver `signal` to this process instance; a process that already exited is not an error.
    pub(super) fn signal(&self, signal: RuntimeSignal) -> MicrosandboxResult<()> {
        tracing::debug!(
            pid = self.pid(),
            proof = ?self.proof(),
            ?signal,
            "signalling the recorded runtime process"
        );
        #[cfg(unix)]
        {
            let signal = match signal {
                RuntimeSignal::Terminate => libc::SIGTERM,
                RuntimeSignal::Kill => libc::SIGKILL,
                RuntimeSignal::Drain => libc::SIGUSR1,
            };
            match &self.identity {
                Some(identity) => identity.signal(signal)?,
                // No readable identity: plain kill(2) is all this platform offers here.
                None if unsafe { libc::kill(self.pid, signal) } == 0 => {}
                None => {
                    let error = std::io::Error::last_os_error();
                    if error.raw_os_error() != Some(libc::ESRCH) {
                        return Err(error.into());
                    }
                }
            }
            Ok(())
        }
        #[cfg(windows)]
        {
            // Windows has no graceful signal; both requests terminate the process.
            match signal {
                RuntimeSignal::Terminate | RuntimeSignal::Kill => {
                    super::LocalBackend::terminate_pid(self.pid)
                }
            }
        }
    }

    /// Whether this process instance has exited, without consuming its wait status.
    ///
    /// An identity that can no longer be observed counts as exited: after a signal, a failed
    /// re-verification means the PID changed hands or this user's access to it ended, never a
    /// live runtime still owned by the row. The bare-PID fallback applies only to a process
    /// whose identity was unreadable from the start.
    pub(super) fn has_exited(&self) -> bool {
        #[cfg(unix)]
        if let Some(identity) = &self.identity {
            return match identity.has_exited() {
                Ok(exited) => exited,
                Err(error) => {
                    tracing::debug!(
                        pid = self.pid,
                        error = %error,
                        "recorded runtime process can no longer be observed; treating it as exited"
                    );
                    true
                }
            };
        }
        super::LocalBackend::pid_has_exited(self.pid)
    }
}

//--------------------------------------------------------------------------------------------------
// Functions
//--------------------------------------------------------------------------------------------------

/// The decision table from the module documentation. `None` is `Recycled`.
#[cfg(unix)]
fn verdict(lock: LockEvidence, time: TimeEvidence, exe: ExeEvidence) -> Option<Proof> {
    match (lock, time, exe) {
        (LockEvidence::Matches, _, _) => Some(Proof::LifecycleLock),
        (LockEvidence::OtherSandbox, _, _) => None,
        (LockEvidence::None, TimeEvidence::Unknown, _) => Some(Proof::Unverified),
        (LockEvidence::None, TimeEvidence::Plausible, _) => Some(Proof::StartedBeforeRun),
        (LockEvidence::None, TimeEvidence::TooLate, ExeEvidence::Msb) => Some(Proof::Unverified),
        (LockEvidence::None, TimeEvidence::TooLate, ExeEvidence::Other | ExeEvidence::Unknown) => {
            None
        }
    }
}

/// Compare the process's inherited lifecycle descriptor with this launcher's `lifecycle` path.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn lock_evidence(pid: i32, lifecycle: &Path) -> LockEvidence {
    match super::process_exit::lifecycle_matches(pid, lifecycle) {
        Ok(true) => return LockEvidence::Matches,
        Ok(false) => {}
        Err(error) => {
            tracing::debug!(
                pid,
                lifecycle = %lifecycle.display(),
                error = %error,
                "cannot compare the recorded runtime's lifecycle descriptor by inode"
            );
        }
    }
    match super::process_exit::lifecycle_link(pid) {
        Ok(Some(link)) => classify_lock_link(&link, lifecycle),
        Ok(None) => LockEvidence::None,
        Err(error) => {
            tracing::debug!(
                pid,
                error = %error,
                "cannot read the recorded runtime's lifecycle descriptor path"
            );
            LockEvidence::None
        }
    }
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn lock_evidence(_pid: i32, _lifecycle: &Path) -> LockEvidence {
    LockEvidence::None
}

/// Classify the path a lifecycle descriptor points at, as the holder's namespace renders it.
///
/// A launcher in another mount namespace sees a different inode but the same stable
/// `<sha256(name)>.lock` file name under `locks/`; a different lifecycle lock name there can
/// only belong to another sandbox's runtime.
#[cfg(unix)]
fn classify_lock_link(link: &Path, lifecycle: &Path) -> LockEvidence {
    let Some(name) = lock_file_name(link) else {
        return LockEvidence::None;
    };
    if lifecycle
        .file_name()
        .is_some_and(|expected| expected == name)
    {
        return LockEvidence::Matches;
    }
    let in_locks_dir = link
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|dir| dir == "locks");
    if in_locks_dir && is_lifecycle_lock_name(name) {
        LockEvidence::OtherSandbox
    } else {
        LockEvidence::None
    }
}

/// File name of a descriptor link, without the ` (deleted)` marker procfs appends to an unlinked target.
#[cfg(unix)]
fn lock_file_name(link: &Path) -> Option<&str> {
    let name = link.file_name()?.to_str()?;
    Some(name.strip_suffix(" (deleted)").unwrap_or(name))
}

/// Whether `name` has the shape [`microsandbox_runtime::ipc::lifecycle_lock_path`] produces.
#[cfg(unix)]
fn is_lifecycle_lock_name(name: &str) -> bool {
    let Some(hash) = name.strip_suffix(".lock") else {
        return false;
    };
    hash.len() == microsandbox_runtime::ipc::LEGACY_SOCKET_HASH_BYTES * 2
        && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Whether the process runs the `msb` binary, from its executable path and its command name.
///
/// Either name suffices: the executable link resolves symlinks and survives an upgrade that
/// replaced the binary (procfs marks it ` (deleted)`), while the command name keeps the name
/// the runtime was launched under.
#[cfg(target_os = "linux")]
fn exe_evidence(pid: i32) -> ExeEvidence {
    let exe = std::fs::read_link(format!("/proc/{pid}/exe"))
        .ok()
        .and_then(|path| lock_file_name(&path).map(str::to_owned));
    let comm = std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .ok()
        .map(|comm| comm.trim().to_owned());
    exe_evidence_from_names([exe, comm])
}

/// Whether the process runs the `msb` binary, from libproc's executable path and command name.
#[cfg(target_os = "macos")]
fn exe_evidence(pid: i32) -> ExeEvidence {
    let mut buffer = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    let length = unsafe {
        libc::proc_pidpath(
            pid,
            buffer.as_mut_ptr().cast(),
            libc::PROC_PIDPATHINFO_MAXSIZE as u32,
        )
    };
    let exe = usize::try_from(length)
        .ok()
        .filter(|length| *length > 0)
        .and_then(|length| {
            Path::new(std::str::from_utf8(&buffer[..length]).ok()?)
                .file_name()?
                .to_str()
                .map(str::to_owned)
        });
    let mut info = std::mem::MaybeUninit::<libc::proc_bsdinfo>::zeroed();
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as i32;
    let read = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTBSDINFO,
            0,
            info.as_mut_ptr().cast(),
            size,
        )
    };
    let comm = (read == size).then(|| {
        let info = unsafe { info.assume_init() };
        let bytes: Vec<u8> = info
            .pbi_comm
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    });
    exe_evidence_from_names([exe, comm])
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn exe_evidence(_pid: i32) -> ExeEvidence {
    ExeEvidence::Unknown
}

#[cfg(unix)]
fn exe_evidence_from_names(names: [Option<String>; 2]) -> ExeEvidence {
    let mut seen = false;
    for name in names.into_iter().flatten() {
        seen = true;
        if crate::runtime::reap::image_basename_is_msb(&name) {
            return ExeEvidence::Msb;
        }
    }
    if seen {
        ExeEvidence::Other
    } else {
        ExeEvidence::Unknown
    }
}

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(all(test, unix))]
mod tests {
    use std::path::PathBuf;
    use std::process::{Child, Command};
    use std::time::{Duration, Instant};

    use chrono::{TimeDelta, Utc};

    use super::*;

    //----------------------------------------------------------------------------------------------
    // Types
    //----------------------------------------------------------------------------------------------

    /// An unrelated long-lived child standing in for whatever now occupies a recorded PID.
    struct Bystander(Child);

    //----------------------------------------------------------------------------------------------
    // Methods
    //----------------------------------------------------------------------------------------------

    impl Bystander {
        fn spawn() -> Self {
            Self(Command::new("sleep").arg("30").spawn().unwrap())
        }

        /// Spawn with `lock_fd` inherited on the runtime's lifecycle descriptor number.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        fn spawn_holding(lock_fd: i32) -> Self {
            use std::os::unix::process::CommandExt;

            let mut command = Command::new("sleep");
            command.arg("30");
            unsafe {
                command.pre_exec(move || {
                    // A spare copy avoids a no-op dup2 should the source already sit at the
                    // target number; dup2 then clears CLOEXEC so the child keeps the inherited
                    // lock descriptor exactly as a spawned runtime does.
                    let spare = libc::fcntl(lock_fd, libc::F_DUPFD_CLOEXEC, 200);
                    if spare < 0
                        || libc::dup2(spare, microsandbox_runtime::vm::LIFECYCLE_LOCK_FD) < 0
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            Self(command.spawn().unwrap())
        }

        fn pid(&self) -> i32 {
            self.0.id() as i32
        }

        fn is_alive(&mut self) -> bool {
            self.0.try_wait().unwrap().is_none()
        }
    }

    impl RuntimeProcess {
        /// A live process whose kernel identity could not be read.
        fn unverified_without_identity(pid: i32) -> Self {
            Self {
                pid,
                proof: Proof::Unverified,
                identity: None,
            }
        }
    }

    //----------------------------------------------------------------------------------------------
    // Trait Implementations
    //----------------------------------------------------------------------------------------------

    impl Drop for Bystander {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    //----------------------------------------------------------------------------------------------
    // Functions
    //----------------------------------------------------------------------------------------------

    fn now() -> NaiveDateTime {
        Utc::now().naive_utc()
    }

    fn an_hour_ago() -> NaiveDateTime {
        (Utc::now() - TimeDelta::hours(1)).naive_utc()
    }

    fn wait_until(mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            assert!(Instant::now() < deadline, "condition not met within 5s");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn lock_path(dir: &str, name: &str) -> PathBuf {
        PathBuf::from(dir)
            .join("locks")
            .join(format!("{name}.lock"))
    }

    //----------------------------------------------------------------------------------------------
    // Tests
    //----------------------------------------------------------------------------------------------

    #[test]
    fn decision_table() {
        use ExeEvidence as E;
        use LockEvidence as L;
        use TimeEvidence as T;

        for time in [T::Unknown, T::Plausible, T::TooLate] {
            for exe in [E::Msb, E::Other, E::Unknown] {
                assert_eq!(verdict(L::Matches, time, exe), Some(Proof::LifecycleLock));
                assert_eq!(verdict(L::OtherSandbox, time, exe), None);
                assert_eq!(verdict(L::None, T::Unknown, exe), Some(Proof::Unverified));
                assert_eq!(
                    verdict(L::None, T::Plausible, exe),
                    Some(Proof::StartedBeforeRun)
                );
            }
        }
        // A too-late creation time condemns a stranger, never a process that is itself msb.
        assert_eq!(
            verdict(L::None, T::TooLate, E::Msb),
            Some(Proof::Unverified)
        );
        assert_eq!(verdict(L::None, T::TooLate, E::Other), None);
        assert_eq!(verdict(L::None, T::TooLate, E::Unknown), None);
    }

    #[test]
    fn lock_link_classification() {
        let hash_a = "a".repeat(32);
        let hash_b = "b".repeat(32);
        let expected = lock_path("/opt/microsandbox/run", &hash_a);

        // Same name in another launcher's run directory: the cross-namespace drainer.
        assert_eq!(
            classify_lock_link(&lock_path("/other/run", &hash_a), &expected),
            LockEvidence::Matches
        );
        // The holder's run directory was torn down under it; the descriptor is still the lock.
        assert_eq!(
            classify_lock_link(
                &PathBuf::from(format!(
                    "/opt/microsandbox/run/locks/{hash_a}.lock (deleted)"
                )),
                &expected
            ),
            LockEvidence::Matches
        );
        // Another sandbox's lifecycle lock: this PID now runs a different VM.
        assert_eq!(
            classify_lock_link(&lock_path("/opt/microsandbox/run", &hash_b), &expected),
            LockEvidence::OtherSandbox
        );
        // Sibling lock kinds and arbitrary files on fd 99 are no evidence either way.
        assert_eq!(
            classify_lock_link(
                &PathBuf::from(format!(
                    "/opt/microsandbox/run/locks/{hash_b}.snapshot-lineage.lock"
                )),
                &expected
            ),
            LockEvidence::None
        );
        assert_eq!(
            classify_lock_link(&PathBuf::from("/tmp/x.lock"), &expected),
            LockEvidence::None
        );
        assert_eq!(
            classify_lock_link(&PathBuf::from("/dev/null"), &expected),
            LockEvidence::None
        );
    }

    #[test]
    fn executable_names_identify_msb() {
        assert_eq!(
            exe_evidence_from_names([Some("/usr/local/bin/msb".into()), Some("msb".into())]),
            ExeEvidence::Msb
        );
        // Upgraded binary: procfs marks the unlinked executable, the command name still tells.
        assert_eq!(
            exe_evidence_from_names([Some("msb (deleted)".into()), None]),
            ExeEvidence::Other
        );
        assert_eq!(
            exe_evidence_from_names([
                Some(
                    lock_file_name(Path::new("/opt/bin/msb (deleted)"))
                        .unwrap()
                        .into()
                ),
                None
            ]),
            ExeEvidence::Msb
        );
        assert_eq!(
            exe_evidence_from_names([Some("/usr/bin/sleep".into()), Some("sleep".into())]),
            ExeEvidence::Other
        );
        assert_eq!(exe_evidence_from_names([None, None]), ExeEvidence::Unknown);
    }

    #[test]
    fn missing_and_dead_pids_are_dead() {
        assert!(matches!(
            RecordedRuntime::inspect(None, Some(now()), None),
            RecordedRuntime::Dead
        ));
        assert!(matches!(
            RecordedRuntime::inspect(Some(-1), Some(now()), None),
            RecordedRuntime::Dead
        ));
        let mut child = Command::new("sh").arg("-c").arg("exit 0").spawn().unwrap();
        let pid = child.id() as i32;
        // The zombie still owns its PID; the runtime it stood for has nonetheless exited.
        wait_until(|| !super::super::LocalBackend::pid_is_alive(pid));
        assert!(matches!(
            RecordedRuntime::inspect(Some(pid), Some(an_hour_ago()), None),
            RecordedRuntime::Dead
        ));
        child.wait().unwrap();
        assert!(matches!(
            RecordedRuntime::inspect(Some(pid), None, None),
            RecordedRuntime::Dead
        ));
    }

    #[test]
    fn stranger_created_after_the_run_started_is_recycled_and_spared() {
        let mut bystander = Bystander::spawn();
        // `sleep` is positively not msb, so the time rule may condemn it.
        assert_eq!(exe_evidence(bystander.pid()), ExeEvidence::Other);
        let verdict = RecordedRuntime::inspect(Some(bystander.pid()), Some(an_hour_ago()), None);
        assert!(matches!(verdict, RecordedRuntime::Recycled));
        assert!(verdict.live().is_none());
        assert!(bystander.is_alive());
    }

    #[test]
    fn process_created_before_the_run_started_is_live_and_signalled() {
        let mut bystander = Bystander::spawn();
        let started_at = now();
        let process = RecordedRuntime::inspect(Some(bystander.pid()), Some(started_at), None)
            .live()
            .expect("a process older than its run row is the runtime");
        assert_eq!(process.proof(), Proof::StartedBeforeRun);
        assert_eq!(process.pid(), bystander.pid());
        assert!(!process.has_exited());

        process.signal(RuntimeSignal::Kill).unwrap();
        wait_until(|| process.has_exited());
        assert!(!bystander.0.wait().unwrap().success());
        // Delivery after exit reports nothing to do rather than following the PID.
        process.signal(RuntimeSignal::Kill).unwrap();
    }

    #[test]
    fn missing_started_at_is_unverified_but_live() {
        let bystander = Bystander::spawn();
        let process = RecordedRuntime::inspect(Some(bystander.pid()), None, None)
            .live()
            .expect("an unprovable identity stays live");
        assert_eq!(process.proof(), Proof::Unverified);
    }

    #[test]
    fn started_at_within_slack_after_creation_is_still_live() {
        let bystander = Bystander::spawn();
        let slack = TimeDelta::microseconds(crate::runtime::reap::IDENTITY_CREATION_SLACK_MICROS);
        // The runtime inserts its row moments after it starts; timer rounding must not
        // turn that ordinary gap into a mismatch.
        let started_at = (Utc::now() - slack + TimeDelta::seconds(1)).naive_utc();
        assert!(RecordedRuntime::inspect(Some(bystander.pid()), Some(started_at), None).is_live());
    }

    #[test]
    fn missing_lifecycle_lock_file_degrades_to_the_time_rule() {
        let bystander = Bystander::spawn();
        // The inode comparison fails with ENOENT on the expected side; that is no evidence.
        let missing = lock_path("/nonexistent/run", &"c".repeat(32));
        let process = RecordedRuntime::inspect(Some(bystander.pid()), Some(now()), Some(&missing))
            .live()
            .expect("an unreadable lock comparison must not condemn the process");
        assert_eq!(process.proof(), Proof::StartedBeforeRun);
    }

    #[test]
    fn process_without_readable_identity_is_still_signalled_and_observed() {
        let mut bystander = Bystander::spawn();
        let process = RuntimeProcess::unverified_without_identity(bystander.pid());
        assert!(!process.has_exited());
        process.signal(RuntimeSignal::Kill).unwrap();
        wait_until(|| process.has_exited());
        assert!(!bystander.0.wait().unwrap().success());
        process.signal(RuntimeSignal::Kill).unwrap();
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn lifecycle_lock_holder_is_the_runtime_regardless_of_started_at() {
        let home = tempfile::tempdir().unwrap();
        let guard =
            microsandbox_runtime::ipc::acquire_lifecycle_guard(home.path(), "identity").unwrap();
        let lifecycle = microsandbox_runtime::ipc::lifecycle_lock_path(home.path(), "identity");
        let mut runtime = Bystander::spawn_holding(guard.as_raw_fd());

        // Same inode: the strongest proof, immune to a stale `started_at`.
        let process =
            RecordedRuntime::inspect(Some(runtime.pid()), Some(an_hour_ago()), Some(&lifecycle))
                .live()
                .expect("the lock holder is the runtime even with a stale started_at");
        assert_eq!(process.proof(), Proof::LifecycleLock);

        // Another launcher's run directory (a second container generation): the inode differs,
        // the lock file may not even exist yet, but the name is the same sandbox's.
        let other_home = tempfile::tempdir().unwrap();
        let foreign = microsandbox_runtime::ipc::lifecycle_lock_path(other_home.path(), "identity");
        assert!(!foreign.exists());
        let process =
            RecordedRuntime::inspect(Some(runtime.pid()), Some(an_hour_ago()), Some(&foreign))
                .live()
                .expect("the same lock name in another namespace is still this runtime");
        assert_eq!(process.proof(), Proof::LifecycleLock);
        let _other_guard =
            microsandbox_runtime::ipc::acquire_lifecycle_guard(other_home.path(), "identity")
                .unwrap();
        assert!(
            RecordedRuntime::inspect(Some(runtime.pid()), Some(an_hour_ago()), Some(&foreign))
                .is_live()
        );

        // A descriptor on a different sandbox's lifecycle lock is a different VM on this PID.
        let other_sandbox = microsandbox_runtime::ipc::lifecycle_lock_path(home.path(), "other");
        assert!(matches!(
            RecordedRuntime::inspect(Some(runtime.pid()), Some(now()), Some(&other_sandbox)),
            RecordedRuntime::Recycled
        ));
        assert!(runtime.is_alive());

        process.signal(RuntimeSignal::Terminate).unwrap();
        wait_until(|| process.has_exited());
        assert!(!runtime.0.wait().unwrap().success());
        drop(guard);
    }
}
