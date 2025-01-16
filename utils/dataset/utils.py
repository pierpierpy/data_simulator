import os
import json
from typing import List, Dict
from fastapi import status, HTTPException

# TODO rinominare il file come dataset_extraction o qualcosa di simile


class GetExtraction:  # TODO non usare get, mettere un nome più parlante
    """
    Provides functionalities to extract and manage data from a dataset.

    Methods are available to retrieve the full dataset, PDF paths, HTML paths, and subsets of the dataset based on content type.
    """

    def __init__(self, landing_zone_path: str, dataset: str) -> None:
        """
        Initializes the object with the path to the landing zone and the dataset file name.
        """
        self.landing_zone_path = landing_zone_path

        self.dataset = dataset
        self.html_dataset = []
        self.pdf_dataset = []
        if not os.path.exists(os.path.join(self.landing_zone_path, self.dataset)):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="path to dataset not found",
            )

    @property
    def get_extraction(self) -> List[Dict]:
        """
        Retrieves the full dataset as a list of dictionaries.
        """
        with open(
            os.path.join(self.landing_zone_path, self.dataset), "r"
        ) as extraction_dataset_data:
            extraction_dataset = json.load(extraction_dataset_data)
        extraction_dataset_data.close()

        return extraction_dataset

    @property
    def get_pdf_paths(self):
        """
        Retrieves paths to all PDF files in the dataset.
        """
        if not self.pdf_dataset:
            self.get_pdf_dataset
        return [pdf_data["path"] for pdf_data in self.pdf_dataset]

    @property
    def get_html_paths(self):
        """
        Retrieves paths to all HTML files in the dataset.
        """

        if not self.html_dataset:
            self.get_html_dataset
        return [html_data["path"] for html_data in self.html_dataset]
    @property
    def get_paths(self):
        """
        Retrieves paths to all files in the dataset.
        """

        return [data["path"] for data in self.get_extraction]
    @property
    def get_pdf_dataset(self) -> List[Dict]:
        """
        Retrieves a subset of the dataset containing only PDF files.
        """
        self.pdf_dataset = [_ for _ in self.get_extraction if _["type"] == "pdf"]
        return self.pdf_dataset

    @property
    def get_html_dataset(self) -> List[Dict]:
        """
        Retrieves a subset of the dataset containing only HTML/webpage files.
        """
        self.html_dataset = [_ for _ in self.get_extraction if _["type"] == "webpage"]
        return self.html_dataset
