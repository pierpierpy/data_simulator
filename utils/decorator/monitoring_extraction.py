import time
from memory_profiler import memory_usage
import psutil
import os
import json
from src.utils.dataset.utils import GetExtraction
import portalocker


def extract_monitor_resources(func):
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
        # peak_memory_before = memory_usage(
        #     -1, interval=0.1, timeout=1, multiprocess=True
        # )
        result = func(*args, **kwargs)
        timestamp_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # peak_memory_after = memory_usage(-1, interval=0.1, timeout=1, multiprocess=True)
        end_time = time.time()
        # cpu_usage = psutil.cpu_percent(interval=1)

        execution_time = end_time - start_time
        # pma = [sum(i) / len(i) for i in peak_memory_after]
        # pmb = [sum(i) / len(i) for i in peak_memory_before]
        # peak_memory_usage = abs(sum(pma) / len(pma) - sum(pmb) / len(pmb))

        additional_info = {
            "Function Name": func.__name__,
            "Timestamp_start": timestamp_start,
            "Timestamp_end": timestamp_end,
            "root_urls": args[1],
            "pid": pid,
        }

        metrics = {
            "Execution Time (seconds)": execution_time,
            # "Peak Memory Usage (MB)": peak_memory_usage,
            # "CPU Usage (%)": cpu_usage,
            **additional_info,
        }
        dataset = GetExtraction(
            **{
                "landing_zone_path": os.path.join(config["metadata_path"]),
                "dataset": "dataset.json",
            }
        )
        extraction = dataset.get_extraction

        dataset_info = {
            "total": len(extraction),
            "number_of_htmls": len(dataset.get_html_dataset),
            "number_of_pdfs": len(dataset.get_pdf_dataset),
            "number_of_OK": len([i for i in extraction if i["status"] == True]),
            "number_of_KO": len([i for i in extraction if i["status"] == False]),
        }
        dump = {
            **dataset_info,
            **metrics,
            **{
                "n_threads": config["n_threads"],
                "depth": config["depth"],
            },
        }
        er = config["extracton_report"]

        with portalocker.Lock(
            os.path.join(er, f"extraction_report.json"), "a+", timeout=2
        ) as file:
            file.seek(0)
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []

            if dump["root_urls"] not in [D["root_urls"] for D in existing_data]:
                existing_data.append(dump)

            file.seek(0)
            file.truncate()
            json.dump(existing_data, file, indent=4, ensure_ascii=False)
        file.close()
        return result

    return wrapper
