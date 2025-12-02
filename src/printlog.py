import shutil
import sys
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Optional, Type


# Remplacer `local_log = False` par un context manager "no-op" qui neutralise les appels .print()
class PrintLogNone:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def print(self, *args, **kwargs):
        # no-op: ne fait rien quand le logger est désactivé
        return None


class PrintLog:
    """Class to handle logging to both stdout and a file."""

    def __init__(
        self,
        output_dir: str = "./outputs/",
        log_time: bool = False,
        extra_name: str = "",
        enable: bool = True,
    ):
        """Initialize the PrintLog class."""
        self.time = time.localtime()
        self.time_str = time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = output_dir
        self.output_dir_time = Path(output_dir) / self.time_str
        self.output_dir_time = str(self.output_dir_time) + extra_name
        self.output_dir_last = Path(output_dir) / "last"
        self.output_dir_last = str(self.output_dir_last) + extra_name
        if not Path(self.output_dir_time).exists():
            Path(self.output_dir_time).mkdir(parents=True, exist_ok=True)
        self.filename = Path(self.output_dir_time) / "print.log"
        self.file = open(self.filename, "w")
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        self.enable_stdout_flag = True
        self.enable_fileout_flag = True
        self.enable_flush_flag = True
        self.thread_lock = threading.Lock()
        self.log_time = log_time
        self.override(enable)
        self.with_state = False

    def copy_last(self):  # at explicit destruction, copy to last_* folder
        if Path(self.output_dir_last).exists():
            shutil.rmtree(self.output_dir_last)
        shutil.copytree(self.output_dir_time, self.output_dir_last)
        print(f"Copied log files from {self.output_dir_time} to {self.output_dir_last}")

    def override(self, enable: bool = True) -> None:
        """Override stdout and stderr."""
        sys.stdout = self
        sys.stderr = self
        self.enable_fileout(enable)
        self.enable_flush(enable)

    def override_stdout(self) -> None:
        """Override stdout."""
        sys.stdout = self

    def override_stderr(self) -> None:
        """Override stderr."""
        sys.stderr = self

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.with_state = True
        self.enable_fileout(True)
        self.enable_flush(True)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the runtime context related to this object."""
        self.enable_fileout(False)
        self.enable_flush(False)
        self.with_state = False

    def enable_stdout(self, enable: bool) -> None:
        """Enable or disable stdout."""
        self.enable_stdout_flag = enable

    def enable_fileout(self, enable: bool) -> None:
        """Enable or disable file output."""
        self.enable_fileout_flag = enable

    def enable_flush(self, enable: bool) -> None:
        """Enable or disable flushing."""
        self.enable_flush_flag = enable

    def print(self, message: str) -> None:
        """Print a message."""
        self.write(message + "\n")

    def write(self, text: str) -> None:
        """Write text to stdout and file."""
        with self.thread_lock:
            if self.enable_stdout_flag:
                self.old_stdout.write(text)
                self.old_stdout.flush()

            text = text.rstrip()
            if len(text) == 0:
                return

            if self.enable_fileout_flag:
                if self.log_time:
                    self.file.write(time.strftime("%Y%m%d-%H%M%S : ") + text + "\n")
                else:
                    self.file.write(text + "\n")

            if self.enable_flush_flag:
                self.file.flush()

    def flush(self) -> None:
        """Flush the file."""
        self.file.flush()

    def restore(self) -> None:
        """Restore stdout and stderr."""
        self.flush()


class PrintLogProcess:
    def __init__(
        self,
        output_dir_time: str,
        process_id: int,
        log_time: bool = False,
        enable: bool = True,
    ):
        """Initialize the PrintLog class."""
        self.output_dir_time = output_dir_time
        if not Path(self.output_dir_time).exists():
            Path(self.output_dir_time).mkdir(parents=True, exist_ok=True)
        self.filename = Path(self.output_dir_time) / f"print_{process_id}.log"
        self.file = open(self.filename, "w")
        self.enable_stdout_flag = True
        self.enable_fileout_flag = True
        self.enable_flush_flag = True
        self.log_time = log_time
        self.override(enable)
        self.with_state = False

    def __del__(self):
        """Destructor to restore stdout and close the file."""
        self.file.close()

    def override(self, enable: bool = True) -> None:
        """Override stdout and stderr."""
        self.enable_fileout(enable)

    def enable_fileout(self, enable: bool) -> None:
        """Enable or disable file output."""
        self.enable_fileout_flag = enable

    def print(self, message: str) -> None:
        """Print a message."""
        self.write(message + "\n")

    def write(self, text: str) -> None:
        """Write text to stdout and file."""
        text = text.rstrip()
        if len(text) == 0:
            return

        if self.log_time:
            self.file.write(time.strftime("%Y%m%d-%H%M%S : ") + text + "\n")
        else:
            self.file.write(text + "\n")

        self.file.flush()

    def flush(self) -> None:
        """Flush the file."""
        self.file.flush()

    def restore(self) -> None:
        """Restore stdout and stderr."""
        self.flush()
