import math
import multiprocessing as mp
import numpy as np
import queue
import time
import torch


class DSAL:

    def __init__(self, images,
                 yml,
                 read_and_transform_function,
                 batch_size=1,
                 epochs=1,
                 num_processes=1,
                 max_queue_size=50,
                 transform=None,
                 mean=None,
                 std=None):

        assert batch_size >= 1, 'The batch size entered is <= 0'
        assert epochs >= 1, 'The epochs entered is <= 0'
        assert num_processes >= 1, 'The number of processes entered is <= 0'

        # storing parameters
        self.images = images

        # check to see if this is a path to labels or label csv
        self.yml = list(yml)

        self.read_and_transform_function = read_and_transform_function
        self.epochs = epochs
        self.transform = transform
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.mean = mean
        self.std = std

        # defining the joinable queues
        self.index_queue = mp.JoinableQueue()
        self.image_label_queue = mp.JoinableQueue(max_queue_size)
        self.command_queue = mp.JoinableQueue()

        # storing indexes to the path array
        self.index_arr = []
        for i in range(len(self.images)):
            self.index_arr.append(i)


        self.index_arr = np.array(self.index_arr)

        self.total_size = self.epochs * self.__len__()

        # defining the processes
        self.read_transform_processes = []

        for _ in range(num_processes):
            proc = mp.Process(target=self.__batch_image_label__,
                              args=(self.read_and_transform_function,
                                    self.images,
                                    yml,
                                    self.index_queue,
                                    self.image_label_queue,
                                    self.command_queue,
                                    self.transform,
                                    mean,
                                    std))
            self.read_transform_processes.append(proc)

        # counter to tell when the processes terminate
        self.accessed = 0

        # variable to use when running training loop
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def __populate_index_queue__(self):
        # Does the first epoch - 1 times

        index_batch = []
        index_counter = 0
        total_counter = 0
        batch_counter = 0

        while True:
            if index_counter == len(self.index_arr):
                index_counter = 0
                shuffler = np.random.permutation(len(self.index_arr))
                self.index_arr = self.index_arr[shuffler]

            if total_counter == self.total_size:
                if len(index_batch) > 0:
                    self.index_queue.put(index_batch)
                break

            index_batch.append(self.index_arr[index_counter])
            index_counter += 1
            total_counter += 1
            batch_counter += 1

            if batch_counter == self.batch_size:
                self.index_queue.put(index_batch)
                index_batch = []
                batch_counter = 0

        for _ in range(self.num_processes):
            self.index_queue.put(None)

    """
    Consumer process of __populate_index_queue__
    Producer process to __getitem__
    """

    @staticmethod
    def __batch_image_label__(read_and_transform_function,
                             images_arr: np.array,
                             yml,
                             index_queue: mp.JoinableQueue,
                             image_label_queue: mp.JoinableQueue,
                             command_queue: mp.JoinableQueue,
                             transform=None,
                             mean=None,
                             std=None):
        while True:
            indexes = index_queue.get()
            index_queue.task_done()

            if indexes is None:
                break

            image_batch = []
            label_batch = []

            for item in indexes:
                index = item

                image, image_name = images_arr[index]
                labels = yml[image_name]

                image, label = read_and_transform_function(image, labels, transform, mean, std)
                image_batch.append(image)
                label_batch.append(label)






            image_batch = torch.stack(image_batch, dim=0)
            label_batch = torch.stack(label_batch, dim=0)

            image_label_queue.put((image_batch, label_batch))
        # Waiting for get_item to be finished with the queue
        while True:
            try:
                sent_val = command_queue.get()
                if sent_val is None:
                    command_queue.task_done()
                    break
            except queue.Empty:
                time.sleep(0.5)
                continue

    """
    Populate queue path and initialize the processes
    """

    def start(self):
        self.__populate_index_queue__()

        for process in self.read_transform_processes:
            process.start()

    """
    Join the processes and terminates them
    """

    def join(self):

        for process in self.read_transform_processes:
            process.join()

        self.image_label_queue.join()

    """
                            SINGLE THREADED BELOW
    ________________________________________________________________________
    """

    # create batch method
    def __len__(self):
        return len(self.images)

    def get_item(self):
        try:
            image, label = self.image_label_queue.get()
            self.image_label_queue.task_done()
            self.accessed += 1

            # if the none counter is the same amount of processes this means that all processes eof is reached
            # deploy the None into command queue to terminate them
            # this is essential in stopping NO FILE found error
            if self.accessed == self.num_batches:
                for j in range(self.num_processes):
                    self.command_queue.put(None)
            return image, label

        except Exception as e:
            print(e, flush=True)

