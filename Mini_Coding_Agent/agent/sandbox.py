"""Fail-closed Linux test runner: Landlock filesystem + seccomp syscall policy.

This is a teaching example, not a substitute for a reviewed container sandbox.
Only Python unittest is supported. Run the agent itself as an unprivileged user.
"""
import ctypes
import ctypes.util
import errno
import os
from pathlib import Path
import resource
import sys
import unittest

libc = ctypes.CDLL(None, use_errno=True)
SYS_CREATE, SYS_ADD, SYS_RESTRICT = 444, 445, 446
VERSION = 1
READ = (1 << 0) | (1 << 2) | (1 << 3)
ALL = (1 << 15) - 1  # Linux Landlock ABI 3+ filesystem rights


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def check(ret, operation):
    if ret < 0:
        code = ctypes.get_errno()
        raise OSError(code, f"{operation}: {os.strerror(code)}")
    return ret


def restrict_filesystem(workspace: Path, scratch: Path):
    abi = check(libc.syscall(SYS_CREATE, 0, 0, VERSION), "landlock ABI")
    if abi < 3:
        raise RuntimeError(f"Landlock ABI >= 3 required, found {abi}")
    attr = Ruleset(ALL)
    fd = check(libc.syscall(SYS_CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0), "create ruleset")
    try:
        for path, rights in [(Path('/usr'), READ), (Path('/lib'), READ), (Path('/lib64'), READ),
                             (Path('/etc/ld.so.cache'), 1 << 2), (workspace, READ), (scratch, ALL),
                             (Path('/dev/null'), (1 << 1) | (1 << 2))]:
            if not path.exists():
                continue
            target = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, target)
                check(libc.syscall(SYS_ADD, fd, 1, ctypes.byref(rule), 0), f"allow {path}")
            finally:
                os.close(target)
        check(libc.prctl(38, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")
        check(libc.syscall(SYS_RESTRICT, fd, 0), "restrict self")
    finally:
        os.close(fd)


def restrict_syscalls():
    sec = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_rule_add.restype = ctypes.c_int
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_load.restype = ctypes.c_int
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = sec.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    names = ('socket socketpair connect bind listen accept accept4 sendto recvfrom sendmsg recvmsg '
             'sendmmsg recvmmsg io_uring_setup ptrace process_vm_readv process_vm_writev pidfd_getfd '
             'kill tkill tgkill bpf mount umount2 unshare setns keyctl perf_event_open open_by_handle_at '
             'execve execveat fork vfork clone clone3').split()
    try:
        for item in names:
            num = sec.seccomp_syscall_resolve_name(item.encode())
            if num >= 0 and sec.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, num, 0) != 0:
                raise RuntimeError(f'could not restrict {item}')
        if sec.seccomp_load(ctx) != 0:
            raise RuntimeError('seccomp_load failed')
    finally:
        sec.seccomp_release(ctx)


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: sandbox.py WORKSPACE SCRATCH')
    workspace, scratch = (Path(p).resolve(strict=True) for p in sys.argv[1:])
    os.chdir(workspace)
    os.environ.clear()
    os.environ.update({'HOME': str(scratch), 'TMPDIR': str(scratch), 'PATH': '/usr/bin:/bin',
                       'PYTHONDONTWRITEBYTECODE': '1', 'LANG': 'C.UTF-8'})
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    try:
        restrict_filesystem(workspace, scratch)
        restrict_syscalls()
    except Exception as exc:
        print(f'SANDBOX_SETUP_ERROR: {exc}', file=sys.stderr, flush=True)
        return 125
    if not (workspace / 'tests').is_dir():
        print('No tests/ directory; verification unavailable.', file=sys.stderr)
        return 2
    suite = unittest.defaultTestLoader.discover(str(workspace / 'tests'), pattern='test*.py')
    if suite.countTestCases() == 0:
        print('No tests collected; verification unavailable.', file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
