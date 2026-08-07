import subprocess
import time
import csv
from datetime import datetime

class HardwareMonitor:
    def __init__(self, output_file="hardware_trace.csv", poll_interval_seconds=1.0):
        self.output_file = output_file
        self.poll_interval = poll_interval_seconds
        
    def get_gpu_metrics(self):
        """Calls nvidia-smi to get current GPU utilization and Memory usage."""
        try:
            # Querying GPU utilization (%), Memory Used (MB), Power Draw (W)
            cmd = [
                "nvidia-smi", 
                "--query-gpu=utilization.gpu,memory.used,power.draw", 
                "--format=csv,noheader,nounits"
            ]
            output = subprocess.check_output(cmd).decode("utf-8").strip()
            # output is something like: "95, 12000, 250.5"
            metrics = [float(x.strip()) for x in output.split(",")]
            return metrics
        except Exception as e:
            # Fallback for local testing without GPU
            return [0.0, 0.0, 0.0]

    def start_monitoring(self):
        print(f"Starting hardware monitor. Logging to {self.output_file} every {self.poll_interval}s...")
        with open(self.output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_absolute", "gpu_util_perc", "gpu_mem_used_mb", "power_draw_w"])
            
            try:
                while True:
                    metrics = self.get_gpu_metrics()
                    # We log the standard unix time so we can align it with the workload script later
                    current_time = time.time() 
                    writer.writerow([current_time] + metrics)
                    f.flush()
                    time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                print("Monitoring stopped.")

if __name__ == "__main__":
    monitor = HardwareMonitor()
    monitor.start_monitoring()
