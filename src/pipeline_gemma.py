import logging
import os
import re
import time

import ollama
import pandas as pd
from google import genai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def normalize_label(label):
    """Helper to normalize labels for comparison (strip, lowercase, remove quotes and punctuation)"""
    if not isinstance(label, str):
        return ""
    label = label.strip().strip('"').strip("'")
    label = re.sub(r"[^\w\s-]", "", label)
    return label.lower()


def compute_metrics(y_true, y_pred):
    """
    Calcola le metriche di classificazione standard tra due serie/array:
    - accuracy
    - precision (weighted)
    - recall (weighted)
    - f1 (weighted)
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class OllamaGemmaPredictor:
    def __init__(
            self,
            model_name="gemma3:12b",
            temperature=0.0,
            num_predict=4
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict

    @staticmethod
    def build_prompt(content, candidate_labels=None, system_message_internal=None):
        system = f"{system_message_internal}\n" if system_message_internal else ""
        labels = f"Possible categories (choose only one, answer only with the category name, nothing else): {', '.join(candidate_labels)}."
        prompt = (
            f"""{system}
            {labels}\n
            Given the following content, classify it strictly into one of the above categories.
            Answer only with the category name, exactly as given, and nothing more.\n
            Content:\n{content}\n
            Category:"""
        )
        return prompt

    def predict(self, content, candidate_labels=None, system_message_internal=None):
        prompt = self.build_prompt(content, candidate_labels, system_message_internal)
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": self.num_predict,
                "temperature": self.temperature,
                "stop": ["\n"]
            }
        )
        output = response['response'].strip()
        lines = [line for line in output.split("\n") if line.strip()]
        if not lines:
            return ""
        first = lines[0]
        if first.startswith("[") and first.endswith("]"):
            return ""
        first_clean = normalize_label(first)
        normalized_labels = [normalize_label(l) for l in candidate_labels]
        for norm_label, orig_label in zip(normalized_labels, candidate_labels):
            if first_clean == norm_label:
                return orig_label
        for norm_label, orig_label in zip(normalized_labels, candidate_labels):
            if norm_label in first_clean:
                return orig_label
        return ""

    def predict_frame(
            self,
            frame,
            content_column,
            category_column,
            system_message_internal=None,
            output_column="predicted_category"
    ):
        candidate_labels = sorted(frame[category_column].dropna().unique())
        preds = []
        total = len(frame)

        logging.info(f"Starting predictions for {total} samples with categories: {candidate_labels}")
        tqdm_bar = tqdm(total=total, desc="Ollama Predicting", ncols=80)
        for idx, (content, true_label) in enumerate(zip(frame[content_column], frame[category_column])):
            if content is not None and content != [] and content != "" and true_label is not None and true_label not in [
                "", "nan"] and true_label != "" and true_label in candidate_labels and len(content) > 0:
                pred = self.predict(
                    content,
                    candidate_labels=candidate_labels,
                    system_message_internal=system_message_internal
                )
                preds.append(pred)
                correct_so_far = sum(
                    normalize_label(p) == normalize_label(t)
                    for p, t in zip(preds, frame[category_column][:idx + 1])
                )
                accuracy_so_far = correct_so_far / (idx + 1)
                tqdm_bar.set_postfix({"accuracy": f"{accuracy_so_far:.4f}"})
                tqdm_bar.update(1)
                logging.info(
                    f"[Ollama {idx + 1}/{total}] True: '{true_label}' | Predicted: '{pred}' | Accuracy: {accuracy_so_far:.4f}")
            else:
                preds.append("")
                tqdm_bar.update(1)
        tqdm_bar.close()

        frame = frame.copy()
        frame[output_column] = preds

        # --- Compute overall metrics after predictions ---
        y_true = frame[category_column].astype(str).str.strip().str.lower()
        y_pred = frame[output_column].astype(str).str.strip().str.lower()

        metrics = compute_metrics(y_true, y_pred)

        logging.info(f"Ollama final accuracy:  {metrics['accuracy']:.4f}")
        logging.info(f"Ollama final precision: {metrics['precision']:.4f}")
        logging.info(f"Ollama final recall:    {metrics['recall']:.4f}")
        logging.info(f"Ollama final f1_score:  {metrics['f1']:.4f}")
        print(f"\nOllama final metrics (from predict_frame):")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}\n")

        return frame


class GeminiPredictor:
    def __init__(self, temperature=0.0, model="models/gemini-2.5-flash", max_retries=5, initial_wait=10):
        self.temperature = temperature
        self.model = model
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise ValueError("GEMINI_KEY or GEMINI_API_KEY not set in environment variables.")

        self.client = genai.Client(api_key=self.key)
        self.max_retries = max_retries
        self.initial_wait = initial_wait

        # For rate limiting: keep timestamps of last requests
        self.request_times = []

    @staticmethod
    def build_prompt(content, candidate_labels=None, system_message_internal=None):
        system = f"{system_message_internal}\n" if system_message_internal else ""
        labels = f"Possible categories (choose only one, answer only with the category name, nothing else): {', '.join(candidate_labels)}."
        prompt = (f"""{system}
            {labels}\n
            Given the following content, classify it strictly into one of the above categories. 
            Answer only with the category name, exactly as given, and nothing more.\n
            Content:\n{content}\n
            Category:"""
                  )
        return prompt

    def _rate_limit(self, max_calls=15, period=60):
        now = time.time()
        # Remove timestamps older than 'period' seconds
        self.request_times = [t for t in self.request_times if now - t < period]
        if len(self.request_times) >= max_calls:
            sleep_time = period - (now - self.request_times[0])
            logging.info(f"[Gemini] Rate limit hit. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            # After sleeping, remove entries now outside the window
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < period]
        self.request_times.append(time.time())

    def safe_generate_content(self, prompt):
        retries = 0
        wait_time = self.initial_wait
        while retries < self.max_retries:
            try:
                self._rate_limit(max_calls=15, period=60)
                logging.info(f"[Gemini] Calling API with prompt: {prompt[:120]}...")  # Log the first part of the prompt
                model = self.client.models
                response = model.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                output_text = response.text.strip() if hasattr(response, "text") else str(response)
                logging.info(f"[Gemini] API call succeeded. Response: {output_text[:100]}...")
                return output_text
            except Exception as e:
                logging.warning(
                    f"Gemini server overloaded (attempt {retries + 1}/{self.max_retries}): {e}. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2  # Exponential backoff
        raise Exception("Max retries exceeded for generate_content (Gemini)")

    def predict(self, content, candidate_labels=None, system_message_internal=None):
        prompt = self.build_prompt(content, candidate_labels, system_message_internal)
        output = self.safe_generate_content(prompt)
        lines = [line for line in output.split("\n") if line.strip()]
        if not lines:
            return ""
        first = lines[0]
        if first.startswith("[") and first.endswith("]"):
            return ""
        first_clean = normalize_label(first)
        normalized_labels = [normalize_label(l) for l in candidate_labels]
        for norm_label, orig_label in zip(normalized_labels, candidate_labels):
            if first_clean == norm_label:
                return orig_label
        for norm_label, orig_label in zip(normalized_labels, candidate_labels):
            if norm_label in first_clean:
                return orig_label
        return ""

    def predict_frame(
            self,
            frame,
            content_column,
            category_column,
            system_message_internal=None,
            output_column="gemini_predicted_category"
    ):
        candidate_labels = sorted(frame[category_column].dropna().unique())
        preds = []
        total = len(frame)

        logging.info(f"Starting Gemini predictions for {total} samples with categories: {candidate_labels}")
        tqdm_bar = tqdm(total=total, desc="Gemini Predicting", ncols=80)
        for idx, (content, true_label) in enumerate(zip(frame[content_column], frame[category_column])):
            if content is not None and content != [] and content != "" and true_label is not None and true_label not in [
                "", "nan"] and true_label != "" and true_label in candidate_labels and len(content) > 0:
                pred = self.predict(
                    content,
                    candidate_labels=candidate_labels,
                    system_message_internal=system_message_internal
                )
                preds.append(pred)
                correct_so_far = sum(
                    normalize_label(p) == normalize_label(t)
                    for p, t in zip(preds, frame[category_column][:idx + 1])
                )
                accuracy_so_far = correct_so_far / (idx + 1)
                tqdm_bar.set_postfix({"accuracy": f"{accuracy_so_far:.4f}"})
                tqdm_bar.update(1)
                logging.info(
                    f"[Gemini {idx + 1}/{total}] True: '{true_label}' | Predicted: '{pred}' | Accuracy: {accuracy_so_far:.4f}")
            else:
                preds.append("")
                tqdm_bar.update(1)
        tqdm_bar.close()

        frame = frame.copy()
        frame[output_column] = preds

        # --- Compute overall metrics after predictions ---
        y_true = frame[category_column].astype(str).str.strip().str.lower()
        y_pred = frame[output_column].astype(str).str.strip().str.lower()

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        logging.info(f"Gemini final accuracy:  {acc:.4f}")
        logging.info(f"Gemini final precision: {precision:.4f}")
        logging.info(f"Gemini final recall:    {recall:.4f}")
        logging.info(f"Gemini final f1_score:  {f1:.4f}")

        return frame


if __name__ == "__main__":
    df = pd.read_json("../data/processed/combined.json")
    system_message = "You are an expert classifier for Linked Open Data. Only answer with one of the given category. Use your reasoning and your knowledge. Do not make up categories."

    # Ollama/Gemma
    predictor = OllamaGemmaPredictor(
        model_name="mistral",  # gemma3:12b
        temperature=0.2,
        num_predict=4
    )
    df_with_preds = predictor.predict_frame(
        df,
        content_column="voc",
        category_column="category",
        system_message_internal=system_message
    )

    # Gemini 2 Flash
    # gemini_predictor = GeminiPredictor(
    #     temperature=0.2,
    #     model="models/gemini-2.0-flash"
    # )
    # df_with_gemini_preds = gemini_predictor.predict_frame(
    #     df,
    #     content_column="comments",
    #     category_column="category",
    #     system_message_internal=system_message
    # )
    # print(df_with_gemini_preds)
