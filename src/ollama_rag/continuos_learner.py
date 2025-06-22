import queue
import threading
import os
import time
import logging
import sqlite3

from watchdog.events import (
    FileSystemEventHandler,
    DirModifiedEvent,
    FileModifiedEvent,
)
from watchdog.observers import Observer
from ollama_rag.learner import Learner

logger = logging.getLogger(__name__)


class LearningQueue:
    def __init__(self, learner: Learner, num_workers: int = 3):
        self.learner = learner
        self.num_workers = num_workers
        self.file_queue = queue.Queue()
        self.threads = []
        self._running = False

    def __worker(self):
        while True:
            filepath = self.file_queue.get()
            if filepath is None:
                break
            try:
                logging.info(f"Starting learning {filepath}")
                self.learner.learn([filepath])
            finally:
                self.file_queue.task_done()

    def start(self):
        if self._running:
            return
        self._running = True
        for _ in range(self.num_workers):
            thread = threading.Thread(target=self.__worker, daemon=True)
            thread.start()
            self.threads.append(thread)

    def add(self, filepath: str) -> None:
        if not self._running:
            raise RuntimeError("Queue must be started before adding tasks.")

        logging.debug(f"Added {filepath} to learning queue")
        self.file_queue.put(filepath)

    def wait_for_completion(self) -> None:
        logger.warn("Waiting for learning tasks to complete...")
        self.file_queue.join()
        logger.info("Learning tasks finished")

    def stop(self) -> None:
        logger.warn("Stopping all learner workers...")
        for _ in self.threads:
            self.file_queue.put(None)
        for t in self.threads:
            t.join()
        self._running = False
        self.threads.clear()
        logger.info("All learner workers stopped")


class LearningHandle(FileSystemEventHandler):
    def __init__(self, learning_queue: LearningQueue):
        self.learning_queue = learning_queue

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        if event.is_directory or event.is_synthetic:
            return

        self.__learn(os.path.abspath(event.src_path))
        return

    def __learn(self, filepath: os.PathLike) -> None:
        self.learning_queue.add(filepath)


class ContinousLearner:
    def __init__(self, learner: Learner, index_db_path: os.PathLike):
        self.learner = learner
        self.db = sqlite3.connect(index_db_path)

    # def __first_learn(self, dest_path: os.PathLike):
    #     self.db.execute(
    #         "CREATE TABLE IF NOT EXISTS learning_index(filename, last_update)"
    #     )
    #
    #     indexed_files = self.db.execute("SELECT * FROM learning_index")
    #
    #     for root, _, files in os.walk(dest_path):
    #         if root.startswith("."):
    #             continue
    #         for file in files:
    #             print(file)
    #             print(root)
    #             if file.startswith("."):
    #                 continue
    #             print(os.path.join(os.path.abspath(root), file))
    #
    def run(self, path: os.PathLike) -> None:
        # self.__first_learn(path)

        learning_queue = LearningQueue(self.learner)
        learning_queue.start()

        learning_handle = LearningHandle(learning_queue)
        observer = Observer()
        observer.schedule(learning_handle, path, recursive=True)
        logger.info(f"Starting observer for {path}")
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.Warn("Interrupted by user")
        finally:
            logger.warn("Stopping continous learner...")
            observer.stop()
            observer.join()
            logger.info("Continous learner stopped")

            learning_queue.wait_for_completion()
            learning_queue.stop()
