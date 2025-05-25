import threading
import time
import logging
import psutil

class SystemMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._log_stats, daemon=True)
        self._started = False

    def _log_stats(self):
        try:
            while not self._stop_event.is_set():
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                logging.info(f"[MONITOR] CPU usage: {cpu}%, Memory usage: {mem}%")
                time.sleep(self.interval)
        except Exception as e:
            logging.error(f"[MONITOR] Error in monitoring thread: {e}")

    def start(self):
        if not self._started and not self.thread.is_alive():
            self.thread.start()
            self._started = True
            logging.info("[MONITOR] Monitoring thread started.")
        else:
            logging.warning("[MONITOR] Monitoring thread is already running or was previously started.")

    def stop(self):
        if self._started:
            self._stop_event.set()
            self.thread.join()
            logging.info("[MONITOR] Monitoring thread stopped.")
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
