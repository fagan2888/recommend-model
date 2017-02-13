#coding=utf8

import thread
import time
import threading
import Queue


def print_time(threadname, delay):
    count = 0
    while count < 5:
        time.sleep(delay)
        count += 1
        print '%s: %s' % (threadname, time.ctime(time.time()))


class mythread(threading.Thread):

    def __init__(self, q):
        threading.Thread.__init__(self)
        self.queue = q

    def run(self):
        n = 0
        while n < 100000:
            n = n + 1
            self.queue.put(n)
            #print n


if __name__ == '__main__':

    '''
    try:
        thread.start_new_thread(print_time, ("thread-1", 2))

        thread.start_new_thread(print_time, ("thread-2", 4))
    except:
        print 'Error'

    while 1:
        pass
    '''

    queue = Queue.Queue()
    thread1 = mythread(queue)
    thread2 = mythread(queue)

    threads = []
    threads.append(thread1)
    threads.append(thread2)

    thread1.start()
    thread2.start()


    for t in threads:
        t.join()

    for i in range(0, queue.qsize()):
        print queue.get(i)
