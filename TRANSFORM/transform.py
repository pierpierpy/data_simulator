from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
import json
import os
from typing import List, Dict
import tiktoken
from pathos.pools import ProcessPool
from tqdm import tqdm
import random

from src.utils.pipeline_utils import transform_utils as t_ut
from src.utils.decorator import monitoring_transform as mt
from src.utils import misc as misc
import warnings
from transformers import Pipeline
from fastapi import APIRouter, Request, HTTPException, status
import re

warnings.filterwarnings("ignore", category=Warning)


class ParsePdf:
    def __init__(
        self,
        transform_landing_zone: str = None,
        pdf_landing_zone: str = None,
        full_pdf_landing_zone: str = None,
        n_threads: int = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = ".",
        pipeline: Pipeline = None,
        transform_report: str = None,
        accepted_languages: List[str] = None,
        logger=None,
        total_lenght: int = None,
    ) -> None:
        self.logger = logger
        self.transform_report = transform_report
        self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.splitter = CharacterTextSplitter.from_tiktoken_encoder(
            "cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
        )
        self.accepted_languages = accepted_languages
        self.pipeline = pipeline

        self.transform_landing_zone = transform_landing_zone
        self.total_lenght = total_lenght
        if not transform_landing_zone:
            raise Exception("per favore, fornisci il path per i documenti")

        os.makedirs(self.transform_landing_zone, exist_ok=True)
        self.pdf_landing_zone = pdf_landing_zone
        if not pdf_landing_zone:
            raise Exception("per favore fornisci il path per i documenti")
        self.transform_landing_zone_pdf = os.path.join(
            self.transform_landing_zone, self.pdf_landing_zone
        )
        os.makedirs(self.transform_landing_zone_pdf, exist_ok=True)
        self.full_pdf_landing_zone = full_pdf_landing_zone
        if not full_pdf_landing_zone:
            raise Exception("per favore fornisci il path per i documenti")
        self.transform_landing_zone_full_pdf = os.path.join(
            self.transform_landing_zone, self.full_pdf_landing_zone
        )
        os.makedirs(self.transform_landing_zone_full_pdf, exist_ok=True)
        # self.visited = list(set([hash_.split("c_")[0] for hash_ in os.listdir(self.transform_landing_zone_pdf)]))
        tr_path = os.path.join(self.transform_report, "transform_dataset.json")
        if os.path.exists(tr_path):
            with open(tr_path, "r") as f:
                visited_data = json.load(f)
            self.visited = [v["hash_url"] for v in visited_data]
        else:
            self.visited = []

    @mt.tranform_monitor_resources
    def load_pdf(self, pdf_data: Dict) -> List[Dict]:
        pdf_data_hash = pdf_data["hash_url"]
        if pdf_data_hash in self.visited:
            print(f"skipping hash: {pdf_data_hash}")
            return [{"status_transform": False, "detail": "already present"}]
        path = pdf_data["path"]
        loader = PyPDFLoader(path)
        loaded_pdf = [l.page_content for l in loader.load_and_split() if l]
        if not loaded_pdf:
            return [{"status_transform": False, "detail": "empty doc"}]
        sample = t_ut.sample_docs(text=loaded_pdf, k=10)
        language, confidence = t_ut.democratic_language_tagger_detect_lang(
            sample, pipeline=self.pipeline
        )
        if not language in self.accepted_languages:
            return [
                {
                    "status_transform": False,
                    "detail": "language not in scope",
                    "language": language,
                    "confidence": confidence,
                }
            ]

        colour = random.choice(["red", "green", "blue", "yellow", "white"])
        pid = os.getpid()
        short_path = misc.truncate_url(pdf_data["path"])

        content = " ".join([page for page in loaded_pdf])
        splitted_doc = self.splitter.split_text(content)
        pbar = tqdm(
            total=len(splitted_doc),
            desc=f"pid: {pid}: Splitting Url {short_path} ",
            colour=colour,
        )

        full_unique_code = pdf_data["hash_url"] + "_full"
        full_path = self.save_pdf_content(
            {
                **pdf_data,
                "unique_code": full_unique_code,
                # "token_gpt": token,
                "language": language,
                "confidence": confidence,
                "content": content,
            }
        )

        results = []
        for n, chunk in enumerate(splitted_doc):

            unique_code = pdf_data["hash_url"] + f"c_{n}"
            token = len(self.tiktoken_encoder.encode(chunk))
            path = self.save_pdf_chunk(
                {
                    **pdf_data,
                    "chunk": chunk,
                    "chunk_num": n,
                    "unique_code": unique_code,
                    "token_gpt": token,
                    "language": language,
                    "confidence": confidence,
                    "full_document_path": full_path,
                }
            )
            results.extend(
                [
                    {
                        "status_transform": True,
                        "detail": "fine",
                        "language": language,
                        "confidence": confidence,
                        "path": path,
                        "chunk_num": n,
                        "unique_code": unique_code,
                    }
                ]
            )
            pbar.update(1)

        pbar.close()
        return results

    def save_pdf_chunk(self, pdf_data: Dict):
        path = (
            os.path.join(self.transform_landing_zone_pdf, pdf_data["unique_code"])
            + ".json"
        )
        with open(
            path,
            "w",
        ) as data:
            json.dump(pdf_data, data, indent=2, ensure_ascii=False)
        return path

    def save_pdf_content(self, pdf_data: Dict):
        path = (
            os.path.join(self.transform_landing_zone_full_pdf, pdf_data["unique_code"])
            + ".json"
        )
        with open(
            path,
            "w",
        ) as data:
            json.dump(pdf_data, data, indent=2, ensure_ascii=False)
        return path

    def load_pdf_pool(self, pdf_file_paths) -> ProcessPool:
        pool = ProcessPool(ncpus=self.n_threads, id="INIT")
        pool.map(self.load_pdf, pdf_file_paths)

        return pool


class ParseHtml:
    def __init__(
        self,
        transform_landing_zone: str = None,
        html_landing_zone: str = None,
        full_html_landing_zone: str = None,
        n_threads: int = None,
        pipeline: Pipeline = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = ".",
        transform_report: str = None,
        logger=None,
        accepted_languages: List[str] = None,
        total_lenght: int = None,
    ) -> None:
        self.logger = logger
        self.transform_report = transform_report
        self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.splitter = CharacterTextSplitter.from_tiktoken_encoder(
            "cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
        )
        self.pipeline = pipeline
        self.accepted_languages = accepted_languages
        self.transform_landing_zone = transform_landing_zone
        self.total_lenght = total_lenght
        if not transform_landing_zone:
            raise Exception("per favore, fornisci il path per i documenti")

        self.html_landing_zone = html_landing_zone
        if not html_landing_zone:
            raise Exception("per favore fornisci il path per i documenti")
        self.transform_landing_zone_html = os.path.join(
            self.transform_landing_zone, self.html_landing_zone
        )
        os.makedirs(self.transform_landing_zone_html, exist_ok=True)
        # self.visited = list(set([hash_.split("_c")[0] for hash_ in os.listdir(self.transform_landing_zone_html)]))

        self.full_html_landing_zone = full_html_landing_zone
        if not full_html_landing_zone:
            raise Exception("per favore fornisci il path per i documenti")
        self.transform_landing_zone_full_html = os.path.join(
            self.transform_landing_zone, self.full_html_landing_zone
        )
        os.makedirs(self.transform_landing_zone_full_html, exist_ok=True)

        tr_path = os.path.join(self.transform_report, "transform_dataset.json")
        if os.path.exists(tr_path):
            with open(tr_path, "r") as f:
                visited_data = json.load(f)
            self.visited = [v["hash_url"] for v in visited_data]
        else:
            self.visited = []

    @mt.tranform_monitor_resources
    def load_html(self, html_data: Dict) -> List[Document]:
        html_data_hash = html_data["hash_url"]
        if html_data_hash in self.visited:
            print(f"skipping hash: {html_data_hash}")
            return [{"status_transform": False, "detail": "already present"}]
        path = html_data["path"]
        try:
            loader = BSHTMLLoader(path, open_encoding="utf8")
            loaded_html = [
                re.sub(r"\s+", " ", re.sub(r"\n\s*\n", " ", l.page_content))
                for l in loader.load_and_split()
                if l
            ]
            # re.sub(r"(\n\s*)+\n+", "\n\n", sourceFileContents)

        except:
            return [{"status_transform": False, "detail": "error in reading"}]
        cut = 5
        if len(loaded_html) < cut:
            if len(" ".join([l for l in loaded_html])) < 100:
                return [{"status_transform": False, "detail": "too short html"}]
        try:
            sample = t_ut.sample_docs(text=loaded_html, n_phrases=10, k=cut)
            language, confidence = t_ut.democratic_language_tagger_detect_lang(
                text=sample, pipeline=self.pipeline
            )
        except:
            return [
                {
                    "status_transform": False,
                    "detail": "uknown error in lang detetion",
                }
            ]
        if not language in self.accepted_languages:
            return [
                {
                    "status_transform": False,
                    "detail": "language not in scope",
                    "language": language,
                    "confidence": confidence,
                }
            ]
        colour = random.choice(["red", "green", "blue", "yellow", "white"])
        pid = os.getpid()
        short_url = misc.truncate_url(html_data["path"])

        content = " ".join(loaded_html)
        loaded_html_splitted = self.splitter.split_text(content)
        pbar = tqdm(
            total=len(loaded_html_splitted),
            desc=f"pid: {pid}: Splitting Url {short_url} ",
            colour=colour,
        )

        full_unique_code = html_data["hash_url"]
        full_path = self.save_html_content(
            {
                **html_data,
                "unique_code": full_unique_code,
                # "token_gpt": token,
                "language": language,
                "confidence": confidence,
                "content": content,
            }
        )

        results = []
        for n, chunk in enumerate(loaded_html_splitted):
            unique_code = html_data["hash_url"] + f"c_{n}"

            path = self.save_html_chunk(
                {
                    **html_data,
                    "chunk": chunk,
                    "unique_code": unique_code,
                    "token_gpt": len(self.tiktoken_encoder.encode(chunk)),
                    "language": language,
                    "confidence": confidence,
                    "full_document_path": full_path,
                }
            )
            results.extend(
                [
                    {
                        "status_transform": True,
                        "detail": "fine",
                        "language": language,
                        "confidence": confidence,
                        "path": path,
                        "chunk_num": n,
                        "unique_code": unique_code,
                    }
                ]
            )
            pbar.update(1)

        pbar.close()
        return results

    def save_html_chunk(self, html_data: Dict):
        path = (
            os.path.join(self.transform_landing_zone_html, html_data["unique_code"])
            + ".json"
        )
        with open(
            path,
            "w",
        ) as data:
            json.dump(html_data, data, indent=2, ensure_ascii=False)
        return path

    def save_html_content(self, html_data: Dict):
        path = (
            os.path.join(
                self.transform_landing_zone_full_html, html_data["unique_code"]
            )
            + ".json"
        )
        with open(
            path,
            "w",
        ) as data:
            json.dump(html_data, data, indent=2, ensure_ascii=False)
        return path

    def load_html_pool(self, html_file_data) -> ProcessPool:
        pool = ProcessPool(ncpus=self.n_threads, id="INIT")
        pool.map(self.load_html, html_file_data)
        return pool
