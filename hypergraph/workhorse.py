import threading
from datetime import datetime
import queue as Queue
from threading import Thread
import traceback
import abc
import sys
import os
from abc import abstractmethod
import weakref
import itertools


class _CloseSignal:
    """
    A special class used to stop a worker
    """
    pass


class Group(abc.ABC):
    """
    A callback invoked by a group of job associated with this.
    """

    def __init__(self):
        self.count = 0  # the number of jobs associated to this group

    @abstractmethod
    def callback(self, job, value=None, ex=None):
        """
        Callback invoked by the worker at the end of the execution of a job. This call must be thread safe
        because it may be invoked by multiple concurrent threads.
        :param job: The job object correspondent to this call
        :param value: The value returned by job's run
        :param ex: The exception thrown by job's run
        :return:
        """
        pass


class Job(abc.ABC):
    """
    A workhorse job, the user should implement the abstract method run which will be executed by the worker.
    """

    def __init__(self, group: Group):
        if not isinstance(group, Group):
            raise ValueError()
        self.group_wref = weakref.ref(group)
        group.count += 1
        self._consumed = False

    @abstractmethod
    def run(self):
        """
        Method executed by the worker. Implementations must specify this method.
        :return: The value to be returned to the associated callback.
        """
        pass

    def _run(self):
        group = self.group_wref()
        try:
            v = self.run()
        except:
            if group is not None:
                group.callback(self, ex=sys.exc_info()[1])
        if group is not None:
            group.callback(self, value=v)


class _LocalWorker(Thread):
    """
    A worker running on a local thread. This class is internal and should't invoked directly.
    """

    def __init__(self, worker_id, queue: Queue):
        Thread.__init__(self)
        self.worker_id = worker_id
        self.queue = queue
        self.close_flag = False

    def _get_job(self):
        queue = self.queue

        try:
            while True:
                job = queue.get(block=True)

                if isinstance(job, _CloseSignal):
                    self.close_flag = True
                    queue.task_done()
                    return None

                if isinstance(job, Job):
                    # the main loop must invoke queue.task_done()
                    return job

                # unknown record type
                queue.task_done()
        except Queue.Empty:
            pass

    def run(self):
        queue = self.queue

        while True:
            if self.close_flag:
                return

            job = self._get_job()
            if job is None:
                continue

            try:
                job._run()
            except:
                # TODO monitor errors
                traceback.print_exc()
                pass
            finally:
                queue.task_done()


class Workhorse:
    """
    A manager for a group of workers.
    """

    def __init__(self):
        self.next_worker_id = itertools.count(start=1)
        self._worker_count = 0
        self.queue = Queue()
        self.max_worker_count = 128

    def start_worker(self):
        """
        Start a new worker thread.
        :return: The worker id of the new thread
        """
        worker_id = next(self.next_worker_id)
        th = _LocalWorker(worker_id=worker_id, queue=self.queue)
        th.setDaemon(True)
        th.start()
        self._worker_count += 1
        return worker_id

    def set_workers(self, count=-1):
        """
        Set the number of active workers. This method can be invoked during the execution of the workers and it
        actively start or stop the threads.
        :param count: The number of requested threads
        :return:
        """
        if not isinstance(count, int):
            raise ValueError()

        if count < 0:
            count = len(os.sched_getaffinity(0))

        if count < 0 or count > self.max_worker_count:
            raise ValueError()

        diff = count - self.worker_count
        if diff > 0:
            for _ in range(diff):
                self.start_worker()
        else:
            for _ in range(-diff):
                self.queue.put(_CloseSignal())

    @property
    def worker_count(self) -> int:
        """
        Return the number of worker running
        :return: An integer
        """
        return self._worker_count

    def enqueue(self, job: Job):
        """
        Enqueue a new job into the workhorse's queue. Note that this method is not thread safe from the point of
        view of the job, that is if the same job in enqueue concurrently multiple times it may be consumed multiple
        times.
        :param job:
        :return:
        """
        if not isinstance(job, Job):
            raise ValueError()
        if job._consumed:
            raise ValueError()
        job._consumed = True
        self.queue.put(job)

    def join(self, close=False):
        """
        Wait until the queue is empty. It is advisable to invoke this method as the last function of an application
        so we are sure the queue has been flushed before the process is destroyed
        :param close: if close is True then a close signal is sent to all workers before the join
        :return:
        """
        if close:
            self.set_workers(0)
        self.queue.join()
