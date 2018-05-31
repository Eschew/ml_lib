import os, sys, numpy as np, ast
import tensorflow as tf

import multiprocessing as mp
import threading
import ctypes

import time
import collections

ctypes_to_check = [ctypes.c_float, ctypes.c_double,
                   ctypes.c_bool, ctypes.c_int,
                   ctypes.c_long]

dtype_ctype_mapping = {np.dtype(k): k for k in ctypes_to_check}

"""
Run like the following
def test_fn(prng=prng):
  t0 = time.time()
  while t0 + 5 > time.time():
    pass
  return prng.rand(20, 224, 224, 3).astype(np.float32), prng.randint(10, size=(20, 1)).astype(np.int32)

cr = CustomRunner(4, 5, test_fn)
sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

cr.start_p_threads(sess)

out = cr.get_inputs(5)
t0 = time.time()
for i in range(200):
  sess.run(out)
  
print time.time()-t0
  
# Clean up context
  
cr.stop()
sess.close()


"""

class CustomRunner(object):
  """
  This class manages the the background threads needed to fill
    a queue full of data.
  """

  def __init__(self, num_processes, num_t_per_process,
         data_generating_fn):
    """
    num_processes: Number of NumpyDataBatchingProcess to spawn
    num_t_per_process: The num_threads argument when initializing NumpyDataBatchingProcess
    data_generating_fn: A data generating fn that should be able to start returning
      numpy data before tf Session has been constructed.
      
      If multiple numpy arrays are being returned, need to ensure that the batching
      dimension is the same for all numpy arrays. If different batch size inputs
      are required, pad an extra return tensor1[None, ...], tensor2[None, ...]
      
      Afterwards call cr.get_inputs(1) and remove the first dimension.
      
      Because np.random isn't thread-safe, fn should accept 
        ``prng=np.random.RandomState()`` and all np.random calls
          replaced by prng.


      Refer to NumpyDataBatchingProcess.test_batch for example

    """
    # data_generating_fn should be able to start returning samples before
    # tf session has been constructed, without any arguments
    self.num_processes = num_processes
    self.num_t_per_process = num_t_per_process
    self.data_fn = data_generating_fn
    
    self.data = self.data_fn(prng=np.random.RandomState())
    
    shapes = []
    dtypes = []
    self.placeholder_shapes = []
    self.inps = []
    
    bs = self.data[0].shape[0]
    for d in self.data:
      assert d.shape[0] == bs
      shapes.append(d.shape[1:])
      dtypes.append(d.dtype)
      self.placeholder_shapes.append([None] + list(d.shape[1:]))
    
      self.inps.append(
        tf.placeholder(dtype=d.dtype, shape=[None] + list(d.shape[1:])))
      
    self.queue = tf.FIFOQueue(shapes=shapes,
                  dtypes=dtypes,
                  capacity=200)
    
    self.enqueue_op = self.queue.enqueue_many(self.inps)
    self.started = False
    
  def get_inputs(self, batch_size):
    """
    Return's a list of tensor(s) containing a batch of data
    """
    images_batch = self.queue.dequeue_up_to(batch_size)
    if type(images_batch) is not list:
        images_batch = [images_batch]
        
    return [tf.placeholder_with_default(v, self.placeholder_shapes[i])
            for i, v in enumerate(images_batch)]
  
  def thread_main(self, sess, **kwargs):
    """
    Function run on alternate thread. Basically, keep adding data to the queue.
    
    enqueue_op may block if tf queue is full, closing sess in main thread will raise
      an exception to terminate this thread main
    """
    event = kwargs["event"]
    while not event.is_set():
      for i, v in enumerate(self.valids):
        if v.value:
          # entry is valid
          try:
            sess.run(self.enqueue_op, feed_dict=self._write_fd(i))
          except tf.errors.CancelledError as e:
            return
          v.value = False
          
  def stop(self):
    [e.set() for e in self.p_events]
    [e.set() for e in self.t_events]
    
  def is_alive(self, return_obj=False):
    # returns true if any child process
    # if return_item, returns the first thread/process encountered alive
    if not self.started:
      return False
    
    found_obj = None
    found_alive = False

    for p in self.processes:
      if found_alive:
        break
        
      if p.is_alive():
        found_alive = True
        found_obj = p
        break
        
      for t in p.ts:
        if t.is_alive():
          found_alive = True
          found_obj = t
          break
            
    for t in self.threads:
      if found_alive:
        break
      if t.is_alive():
        found_alive = True
        found_obj = t
        
        
    if return_obj:
      return found_alive, found_obj
    return found_alive
        
  
  def shut_down(self, sess):
    # returns a tensor to close tf queue
    # will only shut down the queue
    self.stop()
    sess.run(self.queue.close(cancel_pending_enqueues=True))
    [t.join() for t in self.threads]
    [p.join() for p in self.processes]
    
  def _write_fd(self, ind):
    return {inp: self.shared_memories[ind][k] for k, inp in enumerate(self.inps)}
    
  def _allocate_share_mem(self, array):
    # allocates and references a shared spot for this array
    ctype = dtype_ctype_mapping[array.dtype]
    shared_array_base = mp.Array(ctype, np.prod(array.shape), lock=False)
    shared_array = np.ctypeslib.as_array(shared_array_base)
    shared_array = shared_array.reshape(array.shape)
    return shared_array
    
    
  def start_p_threads(self, sess):
    # ith index is list of shared memory
    shared_memories = []
    valids = []
    es = []

    processes = []

    for i in range(self.num_processes):
      # Doesn't need to be locked because valid bit should be used to designate valid
      memories = []
      for d in self.data:
        shared_spot = self._allocate_share_mem(d)
        memories.append(shared_spot)
        
      valid = mp.Value(ctypes.c_bool, lock=True)
      e = mp.Event()

      # Each process will create their own np.random.RandomState()
      # for each of their child threads
      p = NumpyDataBatchingProcess(self.data_fn, e, memories, valid,
                                   num_threads=self.num_t_per_process)
      p.daemon=True
      
      processes.append(p)
      
      shared_memories.append(memories)
      es.append(e)
      valids.append(valid)
      
      p.start()
    
    self.p_events = es
    self.valids = valids
    self.shared_memories = shared_memories
    self.processes = processes
    
    threads = []
    es = []
    for i in range(1):
      # should only need 1 thread
      t = StoppableThread(target=self.thread_main, args=(sess,))
      t.daemon = True
      
      es.append(t.get_event())
      threads.append(t)
      
      t.start()
    self.t_events = es
    self.threads = threads
    
    self.started = True
      
class StoppableThread(threading.Thread):

  def __init__(self, group=None, target=None, name=None,
         args=(), kwargs=None, verbose=None):

    """
    Functions exactly like regular threads but expose a terminating
      event which can be accessed in target via kwargs["event"]
    """
    if not kwargs:
      kwargs = {}
    self.event = threading.Event()
    kwargs["event"] = self.event
    super(StoppableThread, self).__init__(group=group, target=target, 
                        name=name, args=args, kwargs=kwargs,
                        verbose=verbose)
  
  def get_event(self):
    return self.event

class NumpyDataBatchingProcess(mp.Process):
  # Valid bit could be implemented with pipe, could be faster
  def __init__(self, data_fn, stop_event, shared_array, valid_bit, maxsize=10,
               num_threads=2):
    """
    data_fn: (fn) A fn that takes an optional argument PRNG that generates batches
    stop_event: (event) A mp.Event which allows the parent process to terminate
      child process
    shared_array: (np-wrapped ctype array) Excluive shared memory between child and parent
    valid_bit: (np-wrapped ctype boolean) parent sets to false once the data has been read,
      child sets to true once new data has been written
    maxsize: (int) maxsize of internal queue for child process
    num_threads: (int) number of data batching threads to run in this process

    Child process simply pops data from its internal queue while the child process' threads
      populate the internal queue.
    """
    super(NumpyDataBatchingProcess, self).__init__()
    
    self.data_fn = data_fn
    self.stop_event = stop_event
    self.shared_array = shared_array
    self.valid_bit = valid_bit
    
    # Adjust for number of threads
    self.data_buffer = collections.deque(maxlen=maxsize + 1)
    self.max_size = maxsize
    # randomize so different from other processes
    np.random.seed()
    
    self.ts = [StoppableThread(target=self.thread_main,
                   kwargs={"prng": np.random.RandomState()}) for i in range(num_threads)]
    for t in self.ts:
      t.daemon=True
    
  def thread_main(self, **kwargs):
    e = kwargs["event"]
    prng = kwargs["prng"]
    while not e.is_set():
      if len(self.data_buffer) < self.max_size:
        # works well for single threads
        self.data_buffer.append(self.data_fn(prng=prng))
    
  def test_batch(self, prng):
    # 5 seconds spinwaiting to simulate long batching times
    # Testing purposes
    t0 = time.time()
    while t0 + 5 > time.time():
      pass
    return [prng.rand(20, 224, 224, 3),
            prng.rand(20, 120, 3)]
  
  def clean_up(self):
    # send close in case thread is blocked temporarily.
    # Daemon should kill thread anyways.
    [t.get_event().set() for t in self.ts]
    [t.join() for t in self.ts]
  
  def run(self):
    [t.start() for t in self.ts]
    while True:
      while len(self.data_buffer):
        if self.stop_event.is_set():
          self.clean_up()
          return
        if not self.valid_bit.value:
          # valid is false
          new_batch = self.data_buffer.pop()
          [np.copyto(self.shared_array[i], new_batch[i]) for i in range(len(new_batch))]
          self.valid_bit.value = True
          break
          