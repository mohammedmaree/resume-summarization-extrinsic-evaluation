from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer 
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.random import RandomSummarizer

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except ImportError:
    print("Transformers library not found. Please install it: pip install transformers sentencepiece torch")
    exit()
import torch
import traceback
import nltk


try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, nltk.downloader.DownloadError):
    print("NLTK 'punkt' data not found. Downloading...")
    try:
         nltk.download('punkt', quiet=True)
         print("'punkt' downloaded.")
    except Exception as e:
        print(f"Failed to download NLTK 'punkt' data: {e}")
        print("Please ensure you have an internet connection or download it manually.")

class SummarizationModels:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() and device=='cuda' else 'cpu'
        print(f"Using device: {self.device}")

        self.extractive_models = {
            'luhn': LuhnSummarizer(),
            'lsa': LsaSummarizer(),
            'lexrank': LexRankSummarizer(),
            'textrank': TextRankSummarizer(),
            'kl': KLSummarizer(),        
            'reduction': ReductionSummarizer(), 
            'random': RandomSummarizer(),    
        }

        self.abstractive_models = {} #

        models_to_load = {

            'bart': "facebook/bart-large-cnn",
            'pegasus': "google/pegasus-xsum",
            't5': "t5-base", 
            't5_small': "t5-small",
            'flan_t5_base': "google/flan-t5-base",
            't5_large': "t5-large",
        }

        for model_key, model_id in models_to_load.items():
            print(f"\nAttempting to load {model_key.upper()} model ({model_id})...")
            try:
                self.abstractive_models[model_key] = {
                    'model': AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device),
                    'tokenizer': AutoTokenizer.from_pretrained(model_id)
                }
                print(f"{model_key.upper()} model loaded successfully.")
            except Exception as e:
                print(f"Could not load {model_key.upper()} model ({model_id}): {e}")
                if model_key in self.abstractive_models:
                    del self.abstractive_models[model_key]

    def extractive_summarize(self, text, model_name, sentences=5):
        if not text:
             return ""
        if model_name not in self.extractive_models:
            print(f"Model '{model_name}' not found in extractive_models. Available: {list(self.extractive_models.keys())}")
            return ""


        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))

            num_parsed_sentences = len(parser.document.sentences)
            if num_parsed_sentences == 0:
                 return ""

            effective_sentences = min(sentences, num_parsed_sentences)
            if effective_sentences <= 0: 
                return ""

            summarizer = self.extractive_models[model_name]
            summary_sentences = summarizer(parser.document, effective_sentences)

            num_summary_sentences = len(summary_sentences)

            final_summary = ' '.join([str(s) for s in summary_sentences])

            return final_summary

        except Exception as e:
            print(f"--- ERROR (modeling.py: extractive_summarize): Error during extractive summarization ({model_name}) ---")
            print(traceback.format_exc()) 
            print(f"--- END ERROR ({model_name}) ---\n")
            return "" 

    def abstractive_summarize(self, text, model_name, input_max_length=1024, summary_max_length=150):
        if not text:
            print("Input text is empty.")
            return ""
        if model_name not in self.abstractive_models:
            print(f"Model {model_name} not available or failed to load. Available: {list(self.abstractive_models.keys())}")
            return ""

        try:
            config = self.abstractive_models[model_name]


            input_text_for_model = text
            if model_name.startswith('t5') and 'flan' not in model_name:
                input_text_for_model = "summarize: " + text

            inputs = config['tokenizer'](
                input_text_for_model,
                max_length=input_max_length, 
                truncation=True,
                return_tensors="pt"
            ).to(self.device) 
            if inputs.input_ids.shape[1] == 0:
                 print(f"--- WARNING (modeling.py: abstractive_summarize): Input became empty after tokenization for model {model_name}. Returning empty summary. ---")
                 return ""

            summary_ids = config['model'].generate(
                inputs["input_ids"],
                max_length=summary_max_length, 
                min_length=max(10, int(summary_max_length * 0.1)), 
                num_beams=4, 
                early_stopping=True,
            )

            decoded_summary = config['tokenizer'].decode(
                summary_ids[0],
                skip_special_tokens=True
            )

            return decoded_summary.strip() 

        except Exception as e:
            print(f"--- ERROR (modeling.py: abstractive_summarize): Error during abstractive summarization ({model_name}) ---")
            print(traceback.format_exc()) 
            if 'CUDA out of memory' in str(e) or 'cuda runtime error' in str(e).lower():
                 print("--- INFO: Attempting to clear CUDA cache due to OOM error ---")
                 torch.cuda.empty_cache()
            print(f"--- END ERROR ({model_name}) ---\n")
            return "" 

    def hybrid_summarize(self, text):
        print("Hybrid summarization not implemented yet.")
        pass 

    def get_available_models(self):
        """Returns a list of successfully loaded/available model names."""
        return sorted(list(self.extractive_models.keys()) + list(self.abstractive_models.keys()))
