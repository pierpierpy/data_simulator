import hashlib
import os
from transformers import pipeline
import psutil


def hash_value(input_value: str) -> str:
    """
    Generates a SHA-256 hash of the input value.

    Args:
        input_value (str): The value to hash.

    Returns:
        A string representing the hexadecimal digest of the hash.
    """
    if not isinstance(input_value, bytes):
        input_value = str(input_value).encode("utf-8")
    hash_object = hashlib.sha256()
    hash_object.update(input_value)
    hashed_value = hash_object.hexdigest()
    return hashed_value


def truncate_url(url, max_length=30):
    """
    Truncates a URL to a specified maximum length, ensuring that the domain and the last part of the path are preserved.

    Args:
        url (str): The URL to truncate.
        max_length (int): The maximum length of the truncated URL.

    Returns:
        The truncated URL as a string.
    """
    if len(url) <= max_length:
        return url
    # Keep the domain and the last part of the path
    parts = url.split("/")
    truncated = "/".join([parts[1], "...", parts[-1]])
    if len(truncated) > max_length:
        return truncated[: max_length - 3] + "..."
    return truncated


def initialize_language_tagger_model():
    if not os.path.exists("src/utils/pipeline_utils/LTM"):
        pipe = pipeline(
            "text-classification", model="papluca/xlm-roberta-base-language-detection"
        )
        pipe.save_pretrained("src/utils/pipeline_utils/LTM")
    else:

        pipe = pipeline("text-classification", model="src/utils/pipeline_utils/LTM")
    return pipe


def kill_subprocesses():
    # Get the current process ID
    current_process = psutil.Process()

    # Get all child processes of the current process
    children = current_process.children(recursive=True)

    # Iterate through each child process and terminate it
    for child in children:
        child.terminate()