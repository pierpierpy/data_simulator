import requests
import json
from tqdm import tqdm
from typing import List, Dict
import os
from urllib3.exceptions import MaxRetryError
from urllib.request import urljoin
from src.utils import misc as misc
from src.utils.pipeline_utils import extraction_utils as e_ut
from src.utils.dataset import utils as d_ut
from src.utils.decorator import monitoring_extraction as mt
import warnings
from bs4 import GuessedAtParserWarning, XMLParsedAsHTMLWarning, BeautifulSoup
import time
from pathos.pools import ProcessPool
import random
import portalocker
from fastapi import status, HTTPException


requests.urllib3.disable_warnings()
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
# TODO suppress matplolib self.self.logger


class Crawler:
    """A class designed to perform web scraping operations, including fetching webpage content, extracting links, and building structure from the links."""

    def __init__(
        self,
        depth: int,
        landing_zone: str = None,
        metadata_path: str = None,
        document_folder_path: str = None,
        html_folder_path: str = None,
        content_path: str = None,
        n_threads: int = None,
        continue_from_before: bool = False,
        extracton_report: str = None,
        logger=None,
    ) -> None:
        """
        Initializes the Crawler object with specified configurations for web scraping.

        Parameters:
        - depth (int): Maximum depth for recursive link searching.
        - landing_zone (str): Base directory to save scrape results.
        - metadata_path (str): Directory to save metadata related to the scraping. # TODO andrebbe cambiato il nome di questo in dataset_path, ovunque
        - document_folder_path (str): Directory to save downloaded documents.
        - html_folder_path (str): Directory to save raw HTML files.
        - content_path (str): Directory to save cleaned content from HTMLs in a json format (OPTIONAL). the folder will be created only if present
        - n_threads (int): Number of threads to use for parallel processing.
        - continue_from_before (bool): Whether to continue from previously saved progress.

        Throws:
        - Exception: If required parameters are not provided or incorrect.
        """
        self.logger = logger
        self.extracton_report = extracton_report
        self.depth = depth
        self.n_threads = n_threads
        self.continue_from_before = continue_from_before
        if not n_threads:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="per favore, fornisci il numero di threads",
            )
        self.landing_zone = landing_zone
        self.metadata_path = os.path.join(self.landing_zone, metadata_path)
        if not landing_zone:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="per favore, fornisci il path della landing zone",
            )
        os.makedirs(self.landing_zone, exist_ok=True)

        self.document_folder_path = os.path.join(
            self.landing_zone, document_folder_path
        )
        if not document_folder_path:
            raise Exception("per favore, fornisci il path per i documenti")
        os.makedirs(self.document_folder_path, exist_ok=True)

        self.html_folder_path = os.path.join(self.landing_zone, html_folder_path)
        if not html_folder_path:
            raise Exception("per favore, fornisci il path per gli html")
        os.makedirs(self.html_folder_path, exist_ok=True)

        if content_path:
            self.content_path = os.path.join(self.landing_zone, content_path)
            os.makedirs(self.content_path, exist_ok=True)
        else:
            self.content_path = None
        self.filters = [
            "twitter",
            "facebook",
            "linkedin",
            "instagram",
            "youtube",
            "just-eat.co",
            "mailto",
            "slack",
            "subscription",
            "donate",
            "meetup",
        ]  # TODO update this filters please
        if metadata_path:
            os.makedirs(self.metadata_path, exist_ok=True)

    def get_content(self, url: str) -> Dict:
        """
        Fetches and processes the content of the given URL, handling different content types.

        Parameters:
        - url (str): The URL to fetch.

        Returns:
        - A dictionary with the keys 'content', 'status', and 'type', describing the fetched content.
        """
        self.logger.info(f"processing {url} at {time.time()}")
        session = requests.Session()

        try:
            reqs = e_ut.requests_retry_session(
                retries=5, backoff_factor=0.2, session=session
            ).get(url, verify=False)
            reqs.raise_for_status()
            extention = e_ut.get_extention(reqs)
            if extention == "pdf":
                with open(
                    os.path.join(
                        self.document_folder_path, f"{misc.hash_value(url)}.pdf"
                    ),
                    "wb",
                ) as document_type_file:
                    document_type_file.write(reqs.content)
                return {"content": None, "status": True, "type": "pdf"}
            elif extention == "webpage":
                soup = BeautifulSoup(reqs.content, "html.parser")
                with open(
                    os.path.join(self.html_folder_path, f"{misc.hash_value(url)}.html"),
                    "wb",
                ) as document_type_file:
                    document_type_file.write(reqs.content)
            else:
                return {"content": None, "status": False, "type": None}
            return {"content": soup, "status": True, "type": "webpage"}
        except requests.HTTPError as e:
            self.logger.error(f"\nHTTP error: {e}\n")
        except MaxRetryError as e:
            self.logger.error(f"\nMax retries exceeded: {e}\n")
        except Exception as e:
            self.logger.error(f"\nunespected error: {e}")
        return {"content": None, "status": False, "type": None}

    def get_links(self, url: str, request_data: Dict) -> List[str]:
        """
        Extracts and returns all the hyperlinks from the fetched content of a webpage.

        Parameters:
        - url (str): The URL from which links are being extracted.
        - request_data (Dict): The content data from which links should be extracted.

        Returns:
        - A dictionary containing the type, status, and the list of extracted URLs.
        """
        urls = []
        if request_data["type"] == "pdf":
            return {"type": "pdf", "status": True, "urls": urls}
        if not request_data["status"]:
            return {"type": None, "status": False, "urls": urls}
        for link in request_data["content"].find_all("a", href=True):
            href = link["href"]
            if not href.startswith(
                "http"
            ):  # TODO qui possiamo mettere anche un altro check, se il link senza estensione ha un hashtag è solo un paragrafo dell'intero sito
                joined_link = urljoin(url, href)
                urls.append(joined_link)
            else:
                urls.append(href)
        return {"type": "webpage", "status": True, "urls": urls}

    def fetch_links(self, url: str, request_data: Dict) -> List[str]:
        """
        Filters and returns relevant hyperlinks from the fetched content, excluding unwanted URLs.

        Parameters:
        - url (str): The URL of the webpage being processed.
        - request_data (Dict): The content data from which links are being fetched and filtered.

        Returns:
        - A dictionary with filtered URLs, their type, and status.
        """
        links_request_data = self.get_links(url, request_data)
        if links_request_data["type"] == "webpage":
            filtered_urls = [
                url
                for url in links_request_data["urls"]
                if not any(social_network in url for social_network in self.filters)
            ]

            return {"type": "webpage", "status": True, "urls": filtered_urls}
        return links_request_data

    def save_content_cleaned(self, parent_url: str, url: str, content: str):
        """
        Saves the cleaned content of a webpage to a JSON file, organizing it by URLs.

        Parameters:
        - parent_url (str): The URL of the parent page.
        - url (str): The current page URL.
        - content (str): The cleaned content to be saved.
        """
        if not content:
            return
        json_path = os.path.join(self.content_path, f"{misc.hash_value(url)}.json")
        html_path = os.path.join(self.html_folder_path, f"{misc.hash_value(url)}.html")
        structure = {
            "hash_parent_url": misc.hash_value(parent_url),
            "parent_url": parent_url,
            "hash_url": misc.hash_value(url),
            "current_url": url,
            "content": content,
            "path": html_path,
        }

        with open(json_path, "w") as data:
            json.dump(structure, data, indent=2, ensure_ascii=False)

    def save_meta(self, url: str, status: str, type: str):
        """
        Saves metadata for a fetched URL to a JSON file, including its status and content type.

        Parameters:
        - url (str): The URL being processed.
        - status (str): The status of the URL fetch.
        - type (str): The content type of the fetched URL.
        """

        url_hash = misc.hash_value(url)
        meta_file_path = os.path.join(self.metadata_path, "dataset.json")

        new_meta = {
            "hash_url": url_hash,
            "url": url,
            "status": status,
            "type": type,
            "path": (
                None
                if not status
                else os.path.join(
                    (
                        self.document_folder_path
                        if type == "pdf"
                        else self.html_folder_path
                    ),
                    f"{url_hash}.pdf" if type == "pdf" else f"{url_hash}.html",
                )
            ),
        }

        with portalocker.Lock(meta_file_path, "a+", timeout=2) as file:
            file.seek(0)
            try:  # TODO questo try è necessario?
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []

            existing_data.append(new_meta)
            unique_existing_data = [
                dict(t) for t in {tuple(d.items()) for d in existing_data}
            ]

            file.seek(0)
            file.truncate()
            json.dump(unique_existing_data, file, indent=4, ensure_ascii=False)
        file.close()

    @mt.extract_monitor_resources
    def crawl_website(self, root_url: str):
        """
        Recursively crawls websites starting from a root URL to the specified depth, saving content and metadata.

        Parameters:
        - root_url (str): The starting point URL for the crawl.

        Returns:
        - The process ID of the crawl operation.
        """
        pid = os.getpid()

        queue = [(root_url, 0)]
        visited = set()

        if (
            os.path.exists(os.path.join(self.metadata_path, "dataset.json"))
            and self.continue_from_before
        ):
            # use this only if you are SURE you want to skip already visited ROOT websites.
            dataset = d_ut.GetExtraction(
                self.metadata_path, "dataset.json"
            ).get_extraction
            visited = set([url_data["url"] for url_data in dataset]) - set([root_url])

        colour = random.choice(["red", "green", "blue", "yellow", "white"])
        short_url = misc.truncate_url(root_url)
        pbar = tqdm(
            total=len(queue),
            desc=f"pid: {pid}: Crwlng Url {short_url} ",
            colour=colour,
        )
        while queue:
            current_url, current_depth = queue.pop(0)  # FIFO queue

            if current_url in visited or current_depth > self.depth:
                pbar.update(1)
                self.logger.debug(
                    f"{current_url} already visited, or depth out of reach"
                )

                continue
            visited.add(current_url)
            request_data = self.get_content(current_url)
            if self.content_path:
                request_string_data = e_ut.get_string_content(
                    request_data["content"],
                    request_data["status"],
                    request_data["type"],
                )
                self.save_content_cleaned("", current_url, request_string_data)
            self.save_meta(current_url, request_data["status"], request_data["type"])

            result = self.fetch_links(current_url, request_data)
            if not result["status"] or result["type"] == "pdf":
                pbar.update(1)
            if result["status"] and result["type"] == "webpage":
                pbar.update(1)
                new_links = [
                    link
                    for link in result["urls"]
                    if link not in visited and current_depth < self.depth
                ]  # questa lista si svuota se tutti gli url dentro il parent sono stati già visitati, il resto è lasciato li dove sono <3
                for link in new_links:
                    queue.append((link, current_depth + 1))
                    pbar.total += 1

        pbar.close()
        return pid

    def crawl_websites_pool(self, root_urls: List[str]) -> ProcessPool:
        """
        Manages a pool of processes for crawling multiple websites in parallel, based on the configured number of threads.

        Parameters:
        - root_urls (List[str]): A list of root URLs to start crawling from.
        """

        pool = ProcessPool(ncpus=self.n_threads, id="INIT")
        pool.map(self.crawl_website, root_urls)

        return pool
