from ml_lib.data_utils import shared_memory_queue_runner

import os, time
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

class TestCase():
    def __init__(self):
        # no gpu needed for this
        self.config = tf.ConfigProto(device_count={'GPU': 0})
        # get warning out
        sess = tf.Session(config=self.config)
        sess.close()
        pass
    
    def executeTests(self):
        testCases = [(names, getattr(self, names)) for names in dir(self) if "test_" == names[:5]]
        print ""
        print "Number of tests: %d" % len(testCases)
        for case in testCases:
            names, fn = case
            print "\t %s" % (names)
        
        for case in testCases:
            # seed 12345
            np.random.seed(12345)
            
            names, fn = case
            print "\n======\n%s Starting" % (names)
            print ""
            time, did_pass = fn()
            print ""
            if did_pass:
                print "%s passed in time: %6f"%(names, time)
            else:
                print "%s failed in time: %6f"%(names, time)
            print "======\n"
                

    def test_StartupShutdown(self):
        t0 = time.time()
        def test_fn(_unused=None, prng=None):
            if _unused:
                raise ValueError("pass prng via prng=prng")
            if prng is None:
                raise ValueError("no prng was passed")
                
            # the probability a thread by chance produces the same
            # thing is .5^100
            return [prng.randint(2, size=(100, )).astype(np.int32)]
        
        cr = shared_memory_queue_runner.CustomRunner(2, 3, test_fn)
        out = cr.get_inputs(100)
        
        sess = tf.Session(config=self.config)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        cr.start_p_threads(sess)
        
        status, obj = cr.is_alive(return_obj=True)
        assert cr.is_alive()
        
        sess.run(out)
        
        # cleanup
        # shut_down will block
        cr.shut_down(sess)
        
        sess.close()
        
        status, obj = cr.is_alive(True)
        print status
        
        
        assert not cr.is_alive()
        t1 = time.time()
        return t1 - t0, True
    
    def test_SingleItemBatchingProcess(self):
        # return 0, True
        t0 = time.time()
        def test_fn(_unused=None, prng=None):
            if _unused:
                raise ValueError("pass prng via prng=prng")
            if prng is None:
                raise ValueError("no prng was passed")
                
            # the probability a thread by chance produces the same
            # thing is .5^100
            return [prng.randint(2, size=(100, )).astype(np.int32)]
        
        cr = shared_memory_queue_runner.CustomRunner(4, 5, test_fn)
        out = cr.get_inputs(100)
        
        sess = tf.Session(config=self.config)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        cr.start_p_threads(sess)
        
        seen_strings = []
        
        for i in range(30):
            data = sess.run(out)
            assert len(data) == 1
            string = str(data[0].tolist())
            assert string not in seen_strings
            seen_strings.append(string)
        
        t1 = time.time()
        # cleanup
        
        cr.shut_down(sess)
        sess.close()
        return t1 - t0, True

    def test_SameBatchSizeMultiItemBatchingProcess(self):
        t0 = time.time()
        def test_fn(_unused=None, prng=None):
            if _unused:
                raise ValueError("pass prng via prng=prng")
            if prng is None:
                raise ValueError("no prng was passed")
                
            # the probability a thread by chance produces the same
            # thing is .5^100
            return [prng.randint(2, size=(100, )).astype(np.int32),
                    prng.randint(10, size=(100, 10, 10)).astype(np.float32)]
        
        cr = shared_memory_queue_runner.CustomRunner(4, 5, test_fn)
        out = cr.get_inputs(100)
        
        sess = tf.Session(config=self.config)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        cr.start_p_threads(sess)
        
        seen_strings = []
        
        for i in range(30):
            data = sess.run(out)
            assert len(data) == 2
            
            data1 = data[0]
            data2 = data[1]
            
            string1 = str(data1.tolist())
            string2 = str(data2.flatten().tolist())
            
            assert data1.shape == (100,)
            assert data2.shape == (100, 10, 10)
            
            assert string1 not in seen_strings
            assert string2 not in seen_strings
            seen_strings.append(string1)
            seen_strings.append(string2)
        
        t1 = time.time()
        # cleanup
        
        cr.shut_down(sess)
        sess.close()
        return t1 - t0, True

    def test_DiffBatchSizeMultiItemBatchingProcess(self):
        t0 = time.time()
        def test_fn(_unused=None, prng=None):
            if _unused:
                raise ValueError("pass prng via prng=prng")
            if prng is None:
                raise ValueError("no prng was passed")
                
            # the probability a thread by chance produces the same
            # thing is .5^100
            return [prng.randint(2, size=(100, )).astype(np.int32)[None, ...],
                    prng.randint(10, size=(10, 10)).astype(np.float32)[None, ...]]
        
        cr = shared_memory_queue_runner.CustomRunner(4, 5, test_fn)
        out = cr.get_inputs(1)
        
        sess = tf.Session(config=self.config)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        cr.start_p_threads(sess)
        
        seen_strings = []
        
        for i in range(30):
            data = sess.run(out)
            assert len(data) == 2
            
            data1 = data[0][0]
            data2 = data[1][0]
            
            string1 = str(data1.tolist())
            string2 = str(data2.flatten().tolist())
            
            assert data1.shape == (100,)
            assert data2.shape == (10, 10)
            
            assert string1 not in seen_strings
            assert string2 not in seen_strings
            seen_strings.append(string1)
            seen_strings.append(string2)
        
        t1 = time.time()
        # cleanup
        
        cr.shut_down(sess)
        sess.close()
        return t1 - t0, True
    
def main():
    tc = TestCase()
    tc.executeTests()
    
    
if __name__ == "__main__":
    main()