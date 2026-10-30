"""Stream child output through Python so Colab displays it immediately."""
from collections import deque
import os
import signal
import subprocess
import threading


_GROUP_MARKER = '_FOOOCUS_COLAB_PROCESS_GROUP'


def _interrupt_on_termination(signum, frame):
    raise KeyboardInterrupt


def _stop(process, owns_group):
    def terminate(force=False):
        try:
            if owns_group:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif force:
                process.kill()
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    terminate()
    try:
        # Nested helpers finish reaping before the owning group enforces its deadline.
        process.wait(timeout=5 if owns_group else 3)
    except subprocess.TimeoutExpired:
        terminate(force=True)
        process.wait()
    finally:
        # The direct child can exit while one of its descendants ignores SIGTERM.
        if owns_group:
            terminate(force=True)


def run_live(command, **kwargs):
    """Stream merged output, raise with an error tail, and clean up on interruption.

    Printing through Python preserves ipykernel routing. ``cwd``, ``env`` and
    other Popen options are supported. On POSIX, nested helpers share a managed
    process group so cancelling the notebook also stops the actual application.
    ``CalledProcessError.output`` retains the final 40 lines (max 8,000 chars).
    """
    options = dict(kwargs)
    options.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    owns_group = os.name == 'posix' and os.environ.get(_GROUP_MARKER) != '1'
    if os.name == 'posix':
        environment = options.get('env')
        options['env'] = dict(os.environ if environment is None else environment)
        options['env'][_GROUP_MARKER] = '1'
        options['start_new_session'] = owns_group
    handle_termination = threading.current_thread() is threading.main_thread()
    previous_handler = signal.getsignal(signal.SIGTERM) if handle_termination else None
    process = None
    tail = deque(maxlen=40)
    try:
        if handle_termination:
            signal.signal(signal.SIGTERM, _interrupt_on_termination)
        process = subprocess.Popen(command, **options)
        for line in process.stdout:
            print(line, end='', flush=True)
            tail.append(line[-8000:])
        returncode = process.wait()
    except BaseException:
        if handle_termination:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None:
            _stop(process, owns_group)
        raise
    finally:
        if process is not None:
            process.stdout.close()
        if handle_termination:
            signal.signal(signal.SIGTERM, previous_handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command, output=''.join(tail)[-8000:])
    return subprocess.CompletedProcess(command, returncode)
