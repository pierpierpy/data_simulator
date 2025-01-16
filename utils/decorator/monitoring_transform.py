import time
from memory_profiler import memory_usage
import psutil
import os
import json
import portalocker


def tranform_monitor_resources(func):
    """
    A decorator that wraps a function to monitor and log its resource usage.

    Args:
        func: The function to wrap.

    Returns:
        The wrapper function that logs the resource usage of the wrapped function.
    """

    def wrapper(*args, **kwargs):
        config = args[0].__dict__
        timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        start_time = time.time()
        pid = os.getpid()
        # peak_memory_before = memory_usage(-1, interval=0.1, timeout=1)
        results = func(*args, **kwargs)
        timestamp_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        end_time = time.time()

        execution_time = end_time - start_time
        for result in results:
            if result["detail"] == "already present":
                return result

            additional_info = {
                "Function Name": func.__name__,
                "Timestamp_start": timestamp_start,
                "Timestamp_end": timestamp_end,
                "pid": pid,
            }

            metrics = {
                "Execution Time (seconds)": execution_time,
                # "Peak Memory Usage (MB)": peak_memory_usage,
                # "CPU Usage (%)": cpu_usage,
                **additional_info,
            }
            dump = {**additional_info, **metrics, **args[1], **result}

            er = config["transform_report"]

            with portalocker.Lock(
                os.path.join(er, f"transform_dataset.json"), "a+", timeout=2
            ) as file:
                file.seek(0)
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []

                existing_data.append(dump)

                file.seek(0)
                file.truncate()
                json.dump(existing_data, file, indent=4, ensure_ascii=False)
            file.close()
        return results

    return wrapper
