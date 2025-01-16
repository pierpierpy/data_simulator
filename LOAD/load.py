import src.utils.pipeline_utils.load_utils as l_ut
import os
import json
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from distilabel.llms.azure import AzureOpenAILLM
import portalocker
from fastapi import HTTPException, status
from pathos.pools import ProcessPool
from typing import List
from tqdm import tqdm
from typing import Dict
import src.utils.misc as msc
from langchain.text_splitter import CharacterTextSplitter


class Loader:
    def __init__(
        self,
        root: str = None,
        landing_zone: str = None,
        client: AzureChatOpenAI = None,
        client_3_5: AzureChatOpenAI = None,
        mutation_client: AzureOpenAILLM = None,
        accepted_languages: List[str] = None,
        token_threshold: int = None,
        num_questions_per_chunk: int = None,
        embeddings: AzureOpenAIEmbeddings = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separator: str = None,
    ) -> None:
        self.num_questions_per_chunk = num_questions_per_chunk
        self.client_3_5 = client_3_5
        self.accepted_languages = accepted_languages
        self.token_threshold = token_threshold
        self.root = root
        self.landing_zone = landing_zone
        self.client = client
        self.mutation_client = mutation_client
        self.visited = []
        self.splitter = CharacterTextSplitter.from_tiktoken_encoder(
            "cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
        )
        self.embeddings = embeddings
        if not os.path.exists(root):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="path not found"
            )
        if not landing_zone:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="please provide a landing zone",
            )
        os.makedirs(landing_zone, exist_ok=True)

        if os.path.exists(os.path.join(self.landing_zone, "meta.json")):
            with open(os.path.join(self.landing_zone, "meta.json")) as file:
                self.visited = [
                    chunk["content_hash"]
                    for chunk in json.load(file)
                    if "content_hash" in chunk.keys()
                ]

    def add_chunk(self, chunk_data: Dict):
        if "language" not in chunk_data.keys():
            new_meta = {
                **chunk_data,
                **{
                    "status": False,
                    "reason": "LMKO",
                },
            }
            self.add_new_metadata(new_meta)
            return
        if chunk_data["language"] not in self.accepted_languages:
            new_meta = {
                **chunk_data,
                **{
                    "status": False,
                    "reason": "LKO",
                },
            }
            self.add_new_metadata(new_meta)
            return
        with open(chunk_data["path"]) as ptc_data:
            chunk = json.load(ptc_data)
        content_hash = msc.hash_value(chunk["chunk"])
        if content_hash in self.visited:
            print(f"skip hash {content_hash}")
            return

        if chunk["token_gpt"] < self.token_threshold:
            new_meta = {
                **chunk,
                **{
                    "status": False,
                    "reason": "TKO",
                    "content_hash": content_hash,
                },
            }
            self.add_new_metadata(new_meta)
            return
        generated_questions = l_ut.add_chunk_to_dataset(
            chunk,
            number_of_questions=self.num_questions_per_chunk,
            client=self.client,
            client_3_5=self.client_3_5,
            mutation_client=self.mutation_client,
            splitter=self.splitter,
            embeddings=self.embeddings,
        )
        if "status" in generated_questions[0]:
            if generated_questions[0]["status"] == "<NOQUESTION>":
                new_meta = {
                    **{
                        key: value
                        for key, value in zip(chunk.keys(), chunk.values())
                        if key not in ["chunk"]
                    },
                    **{
                        "status": False,
                        "reason": "no question",
                        "content_hash": content_hash,
                    },
                }
                self.add_new_metadata(new_meta)
                return
            if generated_questions[0]["status"] == "too many tokens":
                new_meta = {
                    **{
                        key: value
                        for key, value in zip(chunk.keys(), chunk.values())
                        if key not in ["chunk"]
                    },
                    **{
                        "status": False,
                        "reason": "too many tokens",
                        "content_hash": content_hash,
                    },
                }
                self.add_new_metadata(new_meta)
                return
        with portalocker.Lock(
            os.path.join(self.landing_zone, "dataset.json"),
            "a+",
            timeout=2,
            encoding="utf-8",
        ) as file:
            file.seek(0)
            try:
                qac = json.load(file)
            except json.JSONDecodeError:
                qac = []

            qac.extend(generated_questions)

            file.seek(0)
            file.truncate()
            json.dump(qac, file, indent=4, ensure_ascii=False)
        file.close()

        new_meta = {
            **{
                key: value
                for key, value in zip(chunk.keys(), chunk.values())
                if key not in ["chunk"]
            },
            **{
                "status": True,
                "reason": None,
                "content_hash": content_hash,
            },
        }
        self.add_new_metadata(new_meta)
        return True

    def add_new_metadata(self, new_meta: Dict):
        with portalocker.Lock(
            os.path.join(self.landing_zone, "meta.json"), "a+", timeout=2
        ) as file:
            file.seek(0)
            try:
                meta = json.load(file)
            except json.JSONDecodeError:
                meta = []

            meta.append(new_meta)  # TODO rimuovere chunk

            file.seek(0)
            file.truncate()
            json.dump(meta, file, indent=4, ensure_ascii=False)
        file.close()
        return True

    def add_chunks(self, chunk_datas: List[dict]):
        for chunk_data in tqdm(chunk_datas, colour="blue"):
            try:
                self.add_chunk(chunk_data)
            except ValueError:
                new_meta = {**chunk_data, **{"status": "KO-Azure-error"}}
                self.add_new_metadata(new_meta)
                continue
        return True
