//! Identity-checked view of the process a run row records.
//!
//! A run row keeps the runtime's PID, but the row can outlive the process (SIGKILL, host
//! crash, container kill) and the kernel can hand that PID to an unrelated process. Every
//! local-backend decision that reads a recorded PID — is this row still live, may this
//! process be signalled — goes through [`RecordedRuntime`], which requires proof that the PID
//! still names the runtime that wrote the row. A positive mismatch is dead for liveness and is
//! never signalled. An unprovable identity stays conservative: the row is treated as live and
//! the process may be signalled, exactly as before identity checks existed.

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

//--------------------------------------------------------------------------------------------------
// Methods
//--------------------------------------------------------------------------------------------------

impl RecordedRuntime {
    /// Decide what `pid` names for the run row that recorded it.
    ///
    /// `started_at` is the row's own timestamp; `lifecycle` the sandbox's lifecycle lock file.
    /// Proof strength is lifecycle lock (definitive), then creation time against
    /// `started_at`, then nothing. Only a failed creation-time comparison yields
    /// [`RecordedRuntime::Recycled`]; observation failures never do, and never fail the call.
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
                    tracing::warn!(
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

        let proof = if lifecycle.is_some_and(|lifecycle| lifecycle_matches(pid, lifecycle)) {
            Some(Proof::LifecycleLock)
        } else {
            match started_at.map(|started_at| {
                identity
                    .created_unix_micros()
                    .map(|created| (created, started_at.and_utc().timestamp_micros()))
            }) {
                None => Some(Proof::Unverified),
                Some(Ok((created, started))) => {
                    crate::runtime::reap::creation_may_belong_to_run(created, started)
                        .then_some(Proof::StartedBeforeRun)
                }
                Some(Err(error)) => {
                    tracing::warn!(
                        pid,
                        error = %error,
                        "cannot read the recorded runtime's creation time; treating it as the runtime"
                    );
                    Some(Proof::Unverified)
                }
            }
        };

        // The reads above are attributed to the pinned instance only while it is still alive;
        // a process that exited meanwhile may have handed its PID to a new occupant.
        match identity.has_exited() {
            Ok(false) => {}
            Ok(true) => return Self::Dead,
            Err(error) => {
                tracing::warn!(pid, error = %error, "cannot re-verify the recorded runtime process");
            }
        }
        match proof {
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
    pub(super) fn has_exited(&self) -> MicrosandboxResult<bool> {
        #[cfg(unix)]
        if let Some(identity) = &self.identity {
            return Ok(identity.has_exited()?);
        }
        Ok(super::LocalBackend::pid_has_exited(self.pid))
    }
}

//--------------------------------------------------------------------------------------------------
// Functions
//--------------------------------------------------------------------------------------------------

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn lifecycle_matches(pid: i32, lifecycle: &Path) -> bool {
    match super::process_exit::lifecycle_matches(pid, lifecycle) {
        Ok(matches) => matches,
        Err(error) => {
            tracing::debug!(
                pid,
                lifecycle = %lifecycle.display(),
                error = %error,
                "cannot inspect the recorded runtime's lifecycle descriptor"
            );
            false
        }
    }
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn lifecycle_matches(_pid: i32, _lifecycle: &Path) -> bool {
    false
}

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(all(test, unix))]
mod tests {
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

        fn pid(&self) -> i32 {
            self.0.id() as i32
        }

        fn is_alive(&mut self) -> bool {
            self.0.try_wait().unwrap().is_none()
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

    //----------------------------------------------------------------------------------------------
    // Tests
    //----------------------------------------------------------------------------------------------

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
    fn process_created_after_the_run_started_is_recycled_and_spared() {
        let mut bystander = Bystander::spawn();
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
        assert!(!process.has_exited().unwrap());

        process.signal(RuntimeSignal::Kill).unwrap();
        wait_until(|| process.has_exited().unwrap());
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

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn lifecycle_lock_holder_is_the_runtime_regardless_of_started_at() {
        use std::os::unix::process::CommandExt;

        let home = tempfile::tempdir().unwrap();
        let guard =
            microsandbox_runtime::ipc::acquire_lifecycle_guard(home.path(), "identity").unwrap();
        let lifecycle = microsandbox_runtime::ipc::lifecycle_lock_path(home.path(), "identity");
        let lock_fd = guard.as_raw_fd();
        let mut command = Command::new("sleep");
        command.arg("30");
        unsafe {
            command.pre_exec(move || {
                // A spare copy avoids a no-op dup2 should the source already sit at the
                // target number; dup2 then clears CLOEXEC so the child keeps the inherited
                // lock descriptor exactly as a spawned runtime does.
                let spare = libc::fcntl(lock_fd, libc::F_DUPFD_CLOEXEC, 200);
                if spare < 0 || libc::dup2(spare, microsandbox_runtime::vm::LIFECYCLE_LOCK_FD) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut runtime = Bystander(command.spawn().unwrap());

        let process =
            RecordedRuntime::inspect(Some(runtime.pid()), Some(an_hour_ago()), Some(&lifecycle))
                .live()
                .expect("the lock holder is the runtime even with a stale started_at");
        assert_eq!(process.proof(), Proof::LifecycleLock);

        // Without the lifecycle proof the same row falls back to the creation-time rule.
        let unrelated = home.path().join("unrelated.lock");
        std::fs::write(&unrelated, b"").unwrap();
        assert!(matches!(
            RecordedRuntime::inspect(Some(runtime.pid()), Some(an_hour_ago()), Some(&unrelated)),
            RecordedRuntime::Recycled
        ));
        assert!(runtime.is_alive());

        process.signal(RuntimeSignal::Terminate).unwrap();
        wait_until(|| process.has_exited().unwrap());
        assert!(!runtime.0.wait().unwrap().success());
        drop(guard);
    }
}
