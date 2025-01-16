import os
import json
import statistics
from datasets import Dataset

from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from fastapi import HTTPException, status
from datetime import datetime

# TODO includere dati report da load


class Evaluate:
    def __init__(
        self,
        root: str = None,
        landing_zone: str = None,
        client: AzureChatOpenAI = None,
        embedding: AzureOpenAIEmbeddings = None,
        do_evaluate: bool = None,
        version: str = None,
    ) -> None:
        self.root = root
        self.landing_zone = landing_zone
        self.client = client
        self.embedding = embedding
        self.do_evaluate = do_evaluate
        self.version = version
        if not os.path.exists(root):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="path not found"
            )
        with open(os.path.join(self.root, "dataset.json"), "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
            f.close()
        with open(os.path.join(self.root, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
            f.close()

        if not landing_zone:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="please provide a landing zone",
            )
        os.makedirs(landing_zone, exist_ok=True)

    @staticmethod
    def get_type_statistics(meta):
        type_statistics_dict = {}

        for m in meta:
            if m["type"] not in type_statistics_dict.keys():
                type_statistics_dict[m["type"]] = 1
            else:
                type_statistics_dict[m["type"]] += 1

        return type_statistics_dict

    @staticmethod
    def get_token_statistics(meta):
        token_statistics_dict = {}

        n_token = [m["token_gpt"] for m in meta if "token_gpt" in m.keys()]
        token_statistics_dict = {
            "mean": statistics.mean(n_token),
            "max": max(n_token),
            "min": min(n_token),
            "median": statistics.median(n_token),
        }

        return token_statistics_dict

    @staticmethod
    def get_language_statistics(meta):
        language_statistics_dict = {}

        for m in meta:
            if not "language" in m.keys():
                return language_statistics_dict

            if m["language"] not in language_statistics_dict.keys():
                language_statistics_dict[m["language"]] = 1
            else:
                language_statistics_dict[m["language"]] += 1

        return language_statistics_dict

    def validate_dataset(self):

        date = datetime.now().strftime("%d-%m-%y")
        data = [
            {
                "chunk": d["context"],
                "question": d["question"],
                "answer": d["answer"],
            }
            for d in self.dataset
        ]

        metadata = {
            "dataset_size": len(self.dataset),  # numero record totali
            "type": self.get_type_statistics(self.meta),  # tipo{pdf-count, html-count}
            "chunk": self.get_token_statistics(self.meta),  # chunk{media, max, min}
            "language": self.get_language_statistics(self.meta),  # lingua{en-num_en}
            # "meta": self.meta,
            # "version": self.version,
            # "date": date,
        }
        dataset = {
            "version": self.version,
            "date": date,
            "data": data,
            "metadata": metadata,
        }
        v = "".join(self.version.split("."))
        with open(os.path.join(self.landing_zone, f"dataset_{v}.json"), "w") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

        return {
            "version": self.version,
            "date": date,
            "data": data,
            "metadata": metadata,
        }

    def evaluate_dataset(self):

        generated_data = {
            "question": [d["question"] for d in self.dataset],
            "answer": [d["answer"] for d in self.dataset],
            "contexts": [[d["chunk"]] for d in self.dataset],
        }

        generated_data_dict = Dataset.from_dict(generated_data)
        score = evaluate(
            generated_data_dict,
            metrics=[faithfulness, answer_relevancy, harmfulness],
            llm=self.client,
            embeddings=self.embedding,
        )
        score.to_pandas()

        # with open(os.path.join(self.landing_zone, "score.json"), "w") as f:
        #    json.dump(score, f, indent=4)

        return score
