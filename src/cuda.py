import os


def auto_select_gpu(threshold_mb=500):
    try:
        import pynvml

        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        min_used = float("inf")
        best_gpu = 0
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = meminfo.used // 1024 // 1024
            if used_mb < min_used and used_mb < threshold_mb:
                min_used = used_mb
                best_gpu = i
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        print(f"Using GPU {best_gpu} with {min_used} MB used")
        pynvml.nvmlShutdown()
    except Exception as e:
        print("Could not auto-select GPU:", e)
