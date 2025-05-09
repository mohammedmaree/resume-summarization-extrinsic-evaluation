import yaml
import pandas as pd
from tqdm import tqdm
from pathlib import Path 
import os
import time 


from preprocessing import ResumePreprocessor
from modeling_v2 import SummarizationModels


class Pipeline:
    def __init__(self, config_path='config/params.yaml'):
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            print("Using default configuration.")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            exit() 

        self.raw_dir = Path(self.config['raw_data_dir'])
        self.output_dir = Path(self.config['output_dir'])
        
        self.results_dir = Path(self.config.get('results_output_dir', 'evaluation_results')) 
        self.models_to_run = self.config.get('models_to_run', ['luhn'])
        self.preserve_structure = self.config.get('preserve_subfolder_structure', True)

        self.preprocessor = ResumePreprocessor()
        self.models = SummarizationModels(device=self.config.get('device', 'cuda'))

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Raw data directory: {self.raw_dir.resolve()}")
        print(f"Output directory: {self.output_dir.resolve()}")
        print(f"Results directory: {self.results_dir.resolve()}")


    def find_resume_files(self):
        """
        Finds supported resume files (.pdf, .docx, .txt) recursively
        within the raw data directory.
        """
        supported_extensions = ['.pdf', '.docx', '.txt']
        files = []
        if not self.raw_dir.is_dir():
            print(f"Error: Raw data directory '{self.raw_dir}' not found or is not a directory.")
            return files

        print(f"Scanning for {supported_extensions} files recursively in {self.raw_dir}...")
        for ext in supported_extensions:
            files.extend(self.raw_dir.glob(f'**/*{ext}'))
        files = list(set(files))

        print(f"Found {len(files)} resume files.")
        if files:
             print("Example files found:")
             for f in files[:min(5, len(files))]:
                 print(f" - {f}")
        return files

    def run(self):
        """Processes each resume file found recursively with the specified models."""
        resume_files = self.find_resume_files()
        if not resume_files:
            print("No resume files found to process.")
            return

        available_system_models = self.models.get_available_models()
        models_to_process = [m for m in self.models_to_run if m in available_system_models]

        if not models_to_process:
            print("No specified models are available or loaded.")
            return

        print(f"Will process files using models: {models_to_process}")

        timing_results = {model: [] for model in models_to_process}
        processed_files_count = {model: 0 for model in models_to_process}

        for model_name in tqdm(models_to_process, desc="Overall Model Progress"):
            model_base_output_dir = self.output_dir / model_name
            print(f"\nProcessing with model: {model_name}")

            for file_path in tqdm(resume_files, desc=f"  Files for {model_name}", leave=False):
                raw_text = self.preprocessor.convert_to_text(str(file_path))
                if not raw_text: continue

                input_text_for_summary = raw_text
                if not input_text_for_summary: continue

                relative_path = file_path.relative_to(self.raw_dir)
                if self.preserve_structure:
                    output_sub_dir = model_base_output_dir / relative_path.parent
                    summary_filename = f"{relative_path.stem}_summary.txt"
                else:
                    output_sub_dir = model_base_output_dir
                    flat_filename_prefix = str(relative_path.parent).replace(os.sep, '_')
                    summary_filename = f"{flat_filename_prefix}_{relative_path.stem}_summary.txt" if flat_filename_prefix and flat_filename_prefix != '.' else f"{relative_path.stem}_summary.txt"
                summary_output_path = output_sub_dir / summary_filename
                output_sub_dir.mkdir(parents=True, exist_ok=True)
                summary = ""
                start_time = time.time()
                try:
                    if model_name in self.models.extractive_models:
                        summary = self.models.extractive_summarize(input_text_for_summary, model_name, sentences=self.config.get('extractive_sentences', 3))
                    elif model_name in self.models.abstractive_models:
                        summary = self.models.abstractive_summarize(input_text_for_summary, model_name, input_max_length=self.config.get('abstractive_input_max_length', 1024), summary_max_length=self.config.get('abstractive_max_length', 150))
                    else:
                         print(f"      Critical Error: Model '{model_name}' not found.")
                         continue

                    end_time = time.time()
                    duration = end_time - start_time
                    timing_results[model_name].append(duration)
                    processed_files_count[model_name] += 1

                except Exception as e:
                    print(f"      Caught Exception during summarization for {file_path.name} with {model_name}. Skipping timing.")
                    continue


                try:
                    with open(summary_output_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(summary)
                except Exception as e:
                    print(f"      Error saving summary for {file_path.name} to {summary_output_path}: {e}")

        print("\nPipeline finished.")
        print(f"Generated summaries saved under: {self.output_dir.resolve()}")

        print("\n--- Efficiency Analysis (Summarization Step Timing) ---")
        efficiency_data = []
        for model_name, durations in timing_results.items():
            count = processed_files_count[model_name]
            model_type = "Abstractive (GPU/CPU)" if model_name in self.models.abstractive_models else "Extractive (CPU)"
            if count > 0:
                total_time = sum(durations)
                avg_time = total_time / count
                min_time = min(durations)
                max_time = max(durations)
                efficiency_data.append({
                    "Model": model_name, "Type": model_type, "Files Processed": count,
                    "Total Time (s)": total_time, "Avg Time/File (s)": avg_time,
                    "Min Time (s)": min_time, "Max Time (s)": max_time,
                })
            else:
                 efficiency_data.append({
                    "Model": model_name, "Type": model_type, "Files Processed": 0,
                    "Total Time (s)": 0, "Avg Time/File (s)": float('nan'),
                    "Min Time (s)": float('nan'), "Max Time (s)": float('nan'),
                })

        if efficiency_data:
            efficiency_df = pd.DataFrame(efficiency_data)
            efficiency_df.sort_values(by="Avg Time/File (s)", ascending=True, inplace=True)
            print("Summary generation performance per model (lower average time is faster):")
            print(efficiency_df.to_markdown(index=False, floatfmt=".3f"))


            try:
                efficiency_file_path = self.results_dir / "efficiency_summary.csv"
                efficiency_df.to_csv(efficiency_file_path, index=False, float_format='%.3f')
                print(f"\nEfficiency results saved to: {efficiency_file_path.resolve()}")
            except Exception as e:
                print(f"Error saving efficiency results to {efficiency_file_path}: {e}")

            print("\nNotes:")
            print("- Times represent wall-clock duration for the summarization function call only.")
            print("- Excludes text loading, preprocessing, saving, model loading.")
            print("- Abstractive models likely used GPU; Extractive models used CPU.")
        else:
            print("No timing data collected.")



if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
