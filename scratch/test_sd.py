import sounddevice as sd
import numpy as np
import time

def test():
    print("Testing sounddevice stop callback")
    data = np.random.uniform(-0.1, 0.1, 44100 * 5).astype(np.float32)
    sd.play(data, 44100)
    
    time.sleep(1)
    
    count = [0]
    def cb(indata, frames, time_info, status):
        count[0] += 1

    with sd.InputStream(channels=1, samplerate=16000, callback=cb):
        print("InputStream opened.")
        time.sleep(1)
        print("Count before stop:", count[0])
        print("Calling sd.stop(ignore_errors=True)")
        sd.stop()
        time.sleep(1)
        print("Count after stop:", count[0])

test()
