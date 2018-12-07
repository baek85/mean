import time

class timer(object):
    def __init__(self, args):
        self.args = args
        self.start_time = None
        self.end_time = None
    def start(self):
        self.overall = 0
        self.start_time = time.time()
    def go(self):
        self.start_time = time.time()
    def stop(self):
        self.end_time = time.time()
        self.overall += self.end_time - self.start_time
        #print("--- {} seconds ---".format(self.end_time - self.start_time))
