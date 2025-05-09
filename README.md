# Extrinsic Evaluation of Text Summarization Techniques for Resume Analysis Using a Classification Task

**Author:** Mohammed Maree
**Affiliation:** Information Technology Department, Faculty of Information Technology, The Arab American University, Jenin, Palestine
**Email:** mohammad.maree@aaup.edu
**ORCID:** [0000-0002-6114-4687](https://orcid.org/0000-0002-6114-4687)

---

This repository contains the Python code, experimental setup, and resources for the paper titled "Extrinsic Evaluation of Text Summarization Techniques for Resume Analysis Using a Classification Task".

**Published Paper (Link will be added upon publication):**
*   [Link to Published Paper in ...]
*   **DOI:** [DOI once available]

## Abstract

Efficiently processing the large volume of resumes submitted to online recruitment platforms remains a significant challenge. Automatic text summarization offers a potential solution by condensing resume content, potentially aiding downstream tasks like candidate screening. This paper presents a comparative evaluation framework for assessing text summarization techniques applied to resumes, focusing on extrinsic utility due to the lack of standard reference summaries. We investigate prominent extractive methods (Luhn, LSA, LexRank, TextRank, KL, Reduction, Random) and abstractive models (BART, PEGASUS, T5, T5-small, Flan-T5-base) using implementations based on sumy and Hugging Face's transformers. Extrinsically, we evaluate how well summaries preserve category-distinguishing information by measuring the performance (Accuracy, F1-Score) of a downstream classifier predicting the resume's job category based solely on the summary. Efficiency metrics (processing time) are also reported. Our findings highlight the trade-offs between classification utility and computational cost for various summarization algorithms in the context of e-recruitment.


## Setup and Installation

### Prerequisites
*   Python 3.10 or higher
*   pip or Conda for package management
*   Access to a GPU is recommended for running abstractive summarization models efficiently, though extractive methods and classification can run on a CPU.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YourGitHubUsername]/resume-summarization-extrinsic-evaluation.git
    cd resume-summarization-extrinsic-evaluation
    ```

2.  **Create a virtual environment (recommended):**
    *   Using `venv`:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    *   Using `conda`:
        ```bash
        conda env create -f environment.yml
        conda activate resume-summ-env # Or the name specified in environment.yml
        ```
        (If `environment.yml` is not provided, create an environment and install from `requirements.txt`)
        ```bash
        conda create -n resume-summ-env python=3.10
        conda activate resume-summ-env
        ```

3.  **Install dependencies:**
    *   If using `pip` (and `venv` or a manually created Conda env):
        ```bash
        pip install -r requirements.txt
        ```
    *   Ensure NLTK data (e.g., 'punkt' for sentence tokenization) is downloaded:
        ```python
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords') # If used for TF-IDF
        ```
        You can run this in a Python interpreter after activating your environment.

## Data

This study utilizes the "Resume Dataset" publicly available on Kaggle.
*   **Source:** [Resume Dataset on Kaggle by Snehaan Bhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
*   **File Used:** The primary input for this study was the CSV file (`Resume.csv` or similar, containing pre-extracted text) from this dataset to ensure consistency.

**Instructions:**
1.  Download the dataset from the Kaggle link above.
2.  Locate the CSV file containing the resume text and categories.
3.  Place the CSV file (e.g., `Resume.csv`) into the `data/` directory of this repository.
    *   Alternatively, modify the data loading path in `src/main_pipeline.py` or configuration files to point to the location of your dataset.

Please refer to `data/README_DATA.md` for more specific details if provided.

## Running the Experiments

The main experimental pipeline can be executed using the `main_pipeline.py` script.

```bash
python src/main_pipeline.py
