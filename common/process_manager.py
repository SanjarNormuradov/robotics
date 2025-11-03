import multiprocessing as mp
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

from common.logging_manager import get_logger


class ProcessManager:
    """Simple process manager - if anything dies, shutdown everything"""

    def __init__(self):
        self.processes: Dict[str, mp.Process] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.thread_stoppers: Dict[str, Callable] = {}

        # Additional synchronization objects for better IPC
        self.running_event = mp.Event()
        self.running_event.set()  # Initially running
        self.recording_event = mp.Event()  # Initially not recording
        self.evaluating_event = mp.Event()  # Initially not evaluating
        self.episode_count = mp.Value("i", 0)  # Shared integer
        self.command_queue = mp.Queue()  # Command queue for thread-safe communication

        # Multiprocessing-safe dictionary for frame info
        self.manager = mp.Manager()
        self.last_frame = self.manager.dict()
        # Initialize with None values
        self.last_frame["top_camera"] = None
        self.last_frame["right_camera"] = None
        # self.last_frame["robot_tcp_pose"] = None
        self.last_frame["robot_tcp_wrench"] = None
        # self.last_frame["timestamp"] = None

        self.log = get_logger(component=self.__class__.__name__)
        self.log.info("Initialized")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.shutdown_all()

    def send_command(self, command: str):
        """Send a command through the queue (thread-safe)"""
        self.command_queue.put(command)

    def get_command(self, timeout: Optional[float] = None):
        """Get a command from the queue (thread-safe)"""
        try:
            if timeout is None:
                return self.command_queue.get_nowait()
            else:
                return self.command_queue.get(timeout=timeout)
        except Exception:
            return None

    def set_running(self, running: bool):
        """Set running state"""
        if running:
            self.running_event.set()
        else:
            self.running_event.clear()

    def is_running(self) -> bool:
        """Check if system is running"""
        return self.running_event.is_set()

    def set_recording(self, recording: bool):
        """Set recording state"""
        if recording:
            self.recording_event.set()
        else:
            self.recording_event.clear()

    def is_recording(self) -> bool:
        """Check if recording"""
        return self.recording_event.is_set()

    def set_evaluating(self, evaluating: bool):
        """Set evaluating state"""
        if evaluating:
            self.evaluating_event.set()
        else:
            self.evaluating_event.clear()

    def is_evaluating(self) -> bool:
        """Check if evaluating"""
        return self.evaluating_event.is_set()

    def set_episode_count(self, count: int):
        """Set episode count"""
        with self.episode_count.get_lock():
            self.episode_count.value = count

    def get_episode_count(self) -> int:
        """Get episode count"""
        with self.episode_count.get_lock():
            return self.episode_count.value

    def update_frame(
        self,
        top_camera=None,
        right_camera=None,
        tcp_pose=None,
        tcp_wrench=None,
        timestamp=None,
    ):
        """Update frame info in multiprocessing-safe dictionary"""
        try:
            if top_camera is not None:
                self.last_frame["top_camera"] = top_camera
            if right_camera is not None:
                self.last_frame["right_camera"] = right_camera
            # if tcp_pose is not None:
            #     self.last_frame["robot_tcp_pose"] = tcp_pose
            if tcp_wrench is not None:
                self.last_frame["robot_tcp_wrench"] = tcp_wrench
            # if timestamp is not None:
            #     self.last_frame["timestamp"] = timestamp
        except Exception as e:
            self.log.warning(f"Failed to update frame info: {e}")

    def get_frame(self) -> Dict:
        """Get current frame info from multiprocessing-safe dictionary"""
        try:
            return dict(self.last_frame)
        except Exception as e:
            self.log.warning(f"Failed to get frame: {e}")
            return {
                "top_camera": None,
                "right_camera": None,
                # "robot_tcp_pose": None,
                "robot_tcp_wrench": None,
                # "timestamp": None,
            }

    def update_from_sensor_msg(self, sensor_msg: Union[Dict[str, Any], Any]):
        """Update frame info from sensor message (SensorMessage object or dict)"""
        try:
            # Handle both SensorMessage objects and dict format
            if isinstance(sensor_msg, dict):
                top_camera = sensor_msg.get("top_wrist_img", None)
                right_camera = sensor_msg.get("right_wrist_img", None)
                tcp_wrench = sensor_msg.get("robot_tcp_wrench", None)
            else:
                # Assume SensorMessage object with attributes
                top_camera = getattr(sensor_msg, "topCameraRGB", None)
                right_camera = getattr(sensor_msg, "rightCameraRGB", None)
                tcp_wrench = getattr(sensor_msg, "robotTCPwrench", None)

            self.update_frame(
                top_camera=top_camera,
                right_camera=right_camera,
                tcp_wrench=tcp_wrench,
            )
        except Exception as e:
            self.log.warning(f"Failed to update from sensor message: {e}")

    def add_process(self, name: str, process: mp.Process):
        """Add a named process"""
        self.processes[name] = process
        self.log.info(f"Added process: {name}")

    def add_thread(
        self, name: str, thread: threading.Thread, stop_func: Optional[Callable] = None
    ):
        """Add a named thread"""
        self.threads[name] = thread
        if stop_func:
            self.thread_stoppers[name] = stop_func
        self.log.info(f"Added thread: {name}")

    def start_processes(
        self, process_names: Optional[Union[str, List[str]]] = None, wait: bool = True
    ):
        """Start specified process(es)"""
        if process_names is None:
            process_names = list(self.processes.keys())
        else:
            assert isinstance(process_names, list) or isinstance(
                process_names, str
            ), "process_names must be a list of string or a string"

        if isinstance(process_names, str):
            process_names = [process_names]

        for name in process_names:
            process = self.processes.get(name)
            if process and hasattr(process, "start"):
                process.start(wait=wait)
                self.log.info(f"Started process: {name} (PID: {process.pid})")
            else:
                self.log.warning(f"Process {name} not found or has no start method")

    def start_threads(self, thread_names: Optional[Union[str, List[str]]] = None):
        """Start specified thread(s)"""
        if thread_names is None:
            thread_names = list(self.threads.keys())
        else:
            assert isinstance(thread_names, list) or isinstance(
                thread_names, str
            ), "thread_names must be a list of string or a string"

        if isinstance(thread_names, str):
            thread_names = [thread_names]

        for name in thread_names:
            thread = self.threads.get(name)
            if thread and hasattr(thread, "start"):
                thread.start()
                self.log.info(f"Started thread: {name}")
            else:
                self.log.warning(f"Thread {name} not found or has no start method")

    def start_all(self, wait: bool = True):
        """Start all processes and threads"""
        self.start_processes(wait=wait)
        self.start_threads()

    def stop_processes(
        self, process_names: Optional[Union[str, List[str]]] = None, wait: bool = True
    ):
        """Signal processes to stop gracefully"""
        if process_names is None:
            process_names = list(self.processes.keys())
        else:
            assert isinstance(process_names, list) or isinstance(
                process_names, str
            ), "process_names must be a list of string or a string"

        if isinstance(process_names, str):
            process_names = [process_names]

        for name in process_names:
            process = self.processes.get(name)
            if process and process.is_alive() and hasattr(process, "stop"):
                self.log.info(f"Signaled {name} process to stop")
                process.stop(wait=wait)
            else:
                self.log.warning(
                    f"Process {name} not found, not alive, or has no stop method"
                )

    def stop_threads(
        self, thread_names: Optional[Union[str, List[str]]] = None, wait: bool = True
    ):
        """Signal threads to stop gracefully"""
        if thread_names is None:
            thread_names = list(self.threads.keys())
        else:
            assert isinstance(thread_names, list) or isinstance(
                thread_names, str
            ), "thread_names must be a list of string or a string"

        if isinstance(thread_names, str):
            thread_names = [thread_names]

        for name in thread_names:
            thread = self.threads.get(name)
            if thread:
                # Try custom stop function first
                if name in self.thread_stoppers:
                    self.thread_stoppers[name]()
                    self.log.info(f"Called custom stop for {name} thread")
                # Try built-in stop method
                elif hasattr(thread, "stop"):
                    thread.stop(wait=wait)
                    self.log.info(f"Signaled {name} thread to stop")
                else:
                    self.log.warning(
                        f"Thread {name} has no stop method - will be abandoned"
                    )

    def stop_all(self, wait: bool = True):
        """Signal all processes and threads to stop gracefully"""
        self.log.info("Initiating graceful shutdown...")
        self.stop_processes(wait=wait)
        self.stop_threads(wait=wait)

    def shutdown_all(self, wait: bool = True):
        """Graceful shutdown with fallback to force"""
        # Check and force terminate processes that didn't stop gracefully
        for name, process in self.processes.items():
            if process.is_alive():
                self.log.warning(
                    f"Force terminating {name} process (PID: {process.pid})"
                )
                process.terminate()
                process.join(timeout=5.0)  # Increased timeout

                if process.is_alive():
                    self.log.error(f"Force killing {name} process (PID: {process.pid})")
                    process.kill()
                    process.join(timeout=2.0)

                    if process.is_alive():
                        self.log.error(f"Process {name} still alive after kill signal")
                else:
                    self.log.info(f"Process {name} terminated gracefully")

        # Handle threads - they can't be force killed, but we can abandon them
        for name, thread in self.threads.items():
            if thread.is_alive():
                self.log.warning(
                    f"Thread {name} is still alive after stop request - will be abandoned"
                )

        self.processes.clear()
        self.threads.clear()
        self.log.info("Shutdown complete")

    def all_ready(self, wait: bool = True, timeout: float = 30.0) -> bool:
        """Check if ALL processes are ready within timeout. Non-blocking approach."""
        if not wait:
            # Quick check without waiting
            for name, process in self.processes.items():
                try:
                    if not process.is_ready:
                        self.log.warning(f"Process {name} not ready yet")
                        return False
                except AttributeError:
                    self.log.warning(f"Process {name} has no is_ready property")
                    return False
                except Exception as e:
                    self.log.warning(f"Process {name} is_ready check failed: {e}")
                    return False
            return True

        # Wait for all processes to be ready with timeout
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms
        last_log_time = 0  # Track when we last logged

        while time.time() - start_time < timeout:
            # Check if system is shutting down - exit early
            if not self.is_running():
                self.log.info("System shutdown detected - stopping readiness check")
                return False

            all_processes_ready = True
            not_ready_processes = []

            # Check all processes in one iteration
            for name, process in self.processes.items():
                try:
                    if not process.is_ready:
                        not_ready_processes.append(name)
                        all_processes_ready = False
                except AttributeError:
                    self.log.warning(f"Process {name} has no is_ready property")
                    not_ready_processes.append(f"{name}(no-is_ready)")
                    all_processes_ready = False
                except Exception as e:
                    # Other errors (process died, communication error, etc.)
                    self.log.warning(f"Process {name} is_ready check failed: {e}")
                    not_ready_processes.append(f"{name}(error)")
                    all_processes_ready = False

            if all_processes_ready:
                self.log.info("All processes are ready!")
                return True

            # Log progress every 5 seconds (but not if shutting down)
            elapsed = time.time() - start_time
            if (
                elapsed - last_log_time >= 5.0 and self.is_running()
            ):  # Every 5 seconds + still running
                self.log.info(
                    f"Waiting for processes to be ready... ({elapsed:.1f}s/{timeout}s) Not ready: {not_ready_processes}"
                )
                last_log_time = elapsed

            time.sleep(check_interval)

        # Timeout reached - but check if we're shutting down
        if not self.is_running():
            self.log.info("System shutdown detected during readiness timeout")
            return False

        final_not_ready = []
        for name, process in self.processes.items():
            try:
                if not process.is_ready:
                    final_not_ready.append(name)
            except AttributeError:
                final_not_ready.append(f"{name}(no-is_ready)")
            except Exception as e:
                final_not_ready.append(f"{name}(error: {e})")

        self.log.error(
            f"Timeout ({timeout}s) waiting for processes to be ready. Not ready: {final_not_ready}"
        )
        return False

    def all_alive(self) -> bool:
        """Check if ALL processes are alive. Return False if ANY died."""
        for name, process in self.processes.items():
            if not process.is_alive():
                self.log.error(f"Process {name} died (exit code: {process.exitcode})")
                self.log.error(
                    "Shutting down everything - system is no longer functional"
                )
                return False
        return True

    def get_status(self) -> str:
        """Get simple status report"""
        alive_processes = sum(1 for p in self.processes.values() if p.is_alive())
        total_processes = len(self.processes)
        recording_status = "🔴 RECORDING" if self.is_recording() else "⚪ IDLE"
        return f"Processes: {alive_processes}/{total_processes} alive | {recording_status} | Episodes: {self.get_episode_count()}"
