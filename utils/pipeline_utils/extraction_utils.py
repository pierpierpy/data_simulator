import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import re
from bs4 import BeautifulSoup
from requests import Response


# TODO rinomina il file python
def requests_retry_session(
    retries: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (400, 401, 403, 500, 502, 503, 504, 505),
    session: requests.Session = None,
) -> requests.Session:
    """
    This function creates a new requests session that automatically retries failed requests.

    Args:
            retries (int): The number of times to retry a failed request. Default is 5.
            backoff_factor (float): The delay factor to apply between retry attempts. Default is 0.5.
            status_forcelist (tuple): A tuple of HTTP status codes that should force a retry.
                    A retry is initiated if the HTTP status code of the response is in this list.
                    Default is a tuple of common server error codes.
            session (requests.Session): An existing requests session to use. If not provided, a new session will be created.

    Returns:
            requests.Session: A requests session configured with retry behavior.
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def clean_link_lang(url):
    """
    Removes language path segments from URLs.

    Args:
        url (str): The original URL to clean.

    Returns:
        A string with the cleaned URL.
    """
    cleaned_url = re.sub(r"/[a-zA-Z]{2}/", "/", url)
    cleaned_url = re.sub(r"/[a-zA-Z]{2}$", "", cleaned_url)
    return cleaned_url


def get_unique_links(links):
    """
    Filters out similar links, keeping only unique ones based on language path considerations.

    Args:
        links (List[str]): The list of URLs to filter.

    Returns:
        A list of unique URLs after filtering.
    """
    checked_links = {links[0].strip(): 1}
    for link1 in links[1:]:
        link1 = link1.strip()
        is_similar = False
        for link2 in checked_links:
            if clean_link_lang(link1) == clean_link_lang(link2):
                is_similar = True
                break
            if is_similar:
                checked_links[link1] = checked_links[link2] = 0
            else:
                checked_links[link1] = 1
    unique_links = [key for key, val in checked_links.items() if val == 1]
    for link in [key for key, val in checked_links.items() if val == 0]:
        if "/it" in link:
            unique_links.append(link)
    return unique_links


def get_string_content(soup: BeautifulSoup, status: bool, type: str) -> str:
    """
    Extracts and returns all text from a BeautifulSoup object as a single string.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML content.
        status (bool): The status indicating if the content fetch was successful.
        type (str): The content type (e.g., 'pdf' or 'webpage').

    Returns:
        A string containing all extracted text, or None if status is False or type is 'pdf'.
    """
    if not status:
        return None
    if type == "pdf":
        return None
    text = soup.get_text(strip=True)
    return text


def get_clean_content(soup: BeautifulSoup, status: bool, type: str) -> str:
    """
    Extracts and cleans text from a BeautifulSoup object, removing extra whitespace.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object.
        status (bool): The status of the content fetch.
        type (str): The type of content.

    Returns:
        Cleaned text as a string, or None if status is False or type is 'pdf'.
    """
    if not status:
        return None
    if type == "pdf":
        return None
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = "\n".join(chunk for chunk in chunks if chunk)
    return cleaned_text




def get_extention(reqs: Response) -> str:
    if "application/pdf" in reqs.headers.get("content-type"):
        return "pdf"
    elif "text/html" in reqs.headers.get("content-type"):
        return "webpage"
    else:
        return None
