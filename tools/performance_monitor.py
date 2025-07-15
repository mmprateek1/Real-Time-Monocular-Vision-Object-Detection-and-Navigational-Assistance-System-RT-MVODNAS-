import time
import psutil
import GPUtil
from threading import Thread

class PerformanceMonitor:
    def __init__(self):
        self.cpu_percent = 0
        self.memory_percent = 0
        self.gpu_util = 0
        self.fps = 0
        self.frame_times = []
        self.is_monitoring = False
        
    def start_monitoring(self):
        self.is_monitoring = True
        Thread(target=self._monitor, daemon=True).start()
        
    def stop_monitoring(self):
        self.is_monitoring = False
        
    def update_fps(self, frame_time):
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        if self.frame_times:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            
    def _monitor(self):
        while self.is_monitoring:
            self.cpu_percent = psutil.cpu_percent()
            self.memory_percent = psutil.virtual_memory().percent
            
            if GPUtil.getGPUs():
                gpu = GPUtil.getGPUs()[0]
                self.gpu_util = gpu.load * 100
                
            time.sleep(1)
            
    def get_stats(self):
        return {
            'cpu': self.cpu_percent,
            'memory': self.memory_percent,
            'gpu': self.gpu_util,
            'fps': self.fps
        }