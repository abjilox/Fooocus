"""Notebook subprocess output must stay live, useful on failures, and cancellable."""
import contextlib
import io
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import unittest

from modules.colab_process import run_live


class LiveProcessTests(unittest.TestCase):
    def test_stdout_and_stderr_reach_python_stdout(self):
        previous_handler = signal.getsignal(signal.SIGTERM)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run_live([sys.executable, '-u', '-c',
                               "import sys; print('normal'); print('error details', file=sys.stderr)"])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)
        self.assertEqual(output.getvalue(), 'normal\nerror details\n')

    @unittest.skipUnless(os.name == 'posix', 'Pipe readiness check uses POSIX select')
    def test_output_is_visible_while_child_waits(self):
        with tempfile.TemporaryDirectory() as folder:
            release = Path(folder, 'release')
            child = "import pathlib,time; print('ready', flush=True); " + f"p=pathlib.Path({str(release)!r}); " + "\nwhile not p.exists(): time.sleep(.01)\nprint('finished')"
            wrapper = f"from modules.colab_process import run_live; run_live({[sys.executable, '-u', '-c', child]!r})"
            process = subprocess.Popen([sys.executable, '-u', '-c', wrapper], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                self.assertTrue(select.select([process.stdout], [], [], 5)[0], 'No live child output')
                self.assertEqual(process.stdout.readline(), 'ready\n')
                self.assertIsNone(process.poll())
                release.touch()
                output, _ = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, output)
                self.assertEqual(output, 'finished\n')
            finally:
                release.touch()
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_failure_preserves_exit_code_and_bounded_traceback_tail(self):
        output = io.StringIO()
        code = "import sys; [print('old line') for _ in range(60)]; print('inner traceback', file=sys.stderr); sys.exit(7)"
        with contextlib.redirect_stdout(output):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                run_live([sys.executable, '-u', '-c', code])
        self.assertEqual(raised.exception.returncode, 7)
        self.assertIn('inner traceback', raised.exception.output)
        self.assertLess(len(raised.exception.output.splitlines()), 60)
        self.assertEqual(len(output.getvalue().splitlines()), 61)
        self.assertIn('inner traceback', output.getvalue())

    @unittest.skipUnless(os.name == 'posix', 'Colab uses POSIX interruption')
    def test_keyboard_interrupt_terminates_and_reaps_child(self):
        with tempfile.TemporaryDirectory() as folder:
            pidfile = Path(folder, 'pid')
            child = f"import os,pathlib,time; pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); print('ready', flush=True); time.sleep(60)"
            wrapper = f'''import os
from modules.colab_process import run_live
try:
    run_live({[sys.executable, '-u', '-c', child]!r})
except KeyboardInterrupt:
    pid = int(open({str(pidfile)!r}).read())
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        print('reaped', flush=True)
    else:
        raise RuntimeError('Child was not reaped')
else:
    raise RuntimeError('Interruption was swallowed')
'''
            process = subprocess.Popen([sys.executable, '-u', '-c', wrapper], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                self.assertTrue(select.select([process.stdout], [], [], 5)[0])
                self.assertEqual(process.stdout.readline(), 'ready\n')
                process.send_signal(signal.SIGINT)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, output)
                self.assertEqual(output, 'reaped\n')
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    @unittest.skipUnless(os.name == 'posix', 'Colab process tree uses POSIX signals')
    def test_notebook_interrupt_stops_nested_application_without_orphan(self):
        with tempfile.TemporaryDirectory() as folder:
            pidfile = Path(folder, 'application-pid')
            application = f"import os,pathlib,time,signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); print('application-ready', flush=True); time.sleep(60)"
            bootstrap = f"from modules.colab_process import run_live; run_live({[sys.executable, '-u', '-c', application]!r})"
            notebook = f'''import os,signal
from modules.colab_process import run_live
previous_handler = signal.getsignal(signal.SIGTERM)
try:
    run_live({[sys.executable, '-u', '-c', bootstrap]!r})
except KeyboardInterrupt:
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    application_pid = int(open({str(pidfile)!r}).read())
    try:
        os.kill(application_pid, 0)
    except ProcessLookupError:
        print('application-gone', flush=True)
    else:
        raise RuntimeError('Application orphaned after notebook interruption')
else:
    raise RuntimeError('Interruption was swallowed')
'''
            # This fixture models a new notebook kernel, outside the builder's tree.
            notebook_environment = dict(os.environ)
            notebook_environment.pop('_FOOOCUS_COLAB_PROCESS_GROUP', None)
            process = subprocess.Popen([sys.executable, '-u', '-c', notebook], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, env=notebook_environment)
            try:
                self.assertTrue(select.select([process.stdout], [], [], 5)[0])
                self.assertEqual(process.stdout.readline(), 'application-ready\n')
                process.send_signal(signal.SIGINT)
                output, _ = process.communicate(timeout=12)
                self.assertEqual(process.returncode, 0, output)
                self.assertTrue(output.endswith('application-gone\n'), output)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()
                if pidfile.exists():
                    try:
                        os.kill(int(pidfile.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


class NotebookCommandTests(unittest.TestCase):
    def test_prepare_and_start_forward_drive_and_state_without_local_state(self):
        import json
        repository = Path(__file__).resolve().parents[1]
        for filename in ('colab.ipynb', 'fooocus_colab.ipynb'):
            notebook = json.loads((repository / filename).read_text())
            cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
            with self.subTest(notebook=filename), tempfile.TemporaryDirectory() as folder:
                state = str(Path(folder, 'state.json'))
                environment = {
                    'BOOTSTRAP_COMMAND': [sys.executable, '-u', '-c',
                                          'import json,sys; print(json.dumps(sys.argv[1:]))'],
                    'STATE_FILE': state, 'DRIVE_ARGUMENTS': ['--drive-root', '/drive/Fooocus'],
                    'LOCAL_REPO': Path(folder), 'run_live': run_live,
                }
                for cell_id, mode in [('prepare-cache', 'prepare'), ('start', 'start')]:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        exec(compile(cells[cell_id], filename, 'exec'), environment)
                    arguments = json.loads(output.getvalue())
                    self.assertEqual(arguments[0], mode)
                    self.assertEqual(arguments[arguments.index('--state-file') + 1], state)
                    self.assertEqual(arguments[arguments.index('--drive-root') + 1], '/drive/Fooocus')
                    if mode == 'start':
                        self.assertEqual(arguments[arguments.index('--preset') + 1], 'default')
                    self.assertFalse(Path(state).exists())


if __name__ == '__main__':
    unittest.main()
