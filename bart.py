import warnings
from tqdm.notebook import tqdm
import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast
from datasets import load_dataset
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

warnings.filterwarnings('ignore')

# Check PyTorch version and MPS availability
print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')

def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def compute_scores(reference, summary):
    """Computes ROUGE and BLEU scores for a batch."""
    rouge = Rouge()
    rouge_scores = rouge.get_scores(summary, reference)
    bleu_score = sentence_bleu([reference.split()], summary.split())
    return bleu_score, rouge_scores[0]

def execute(test_data, dataset_name, batch_size=1):
    generated_summaries_list = []
    reference_summaries_list = []

    article_column = dataset_column_names[dataset_name]["article"]
    summary_column = dataset_column_names[dataset_name]["summary"]

    for i in tqdm(range(0, len(test_data), batch_size), desc="Processing"):
        batch = test_data.select(range(i, i+batch_size))
        articles = batch[article_column]
        highlights = batch[summary_column]

        generated_summaries = [generate_summary(article) for article in articles]
        generated_summaries_list.extend(generated_summaries)
        reference_summaries_list.extend(highlights)

    return generated_summaries_list, reference_summaries_list

def compute_all_scores(generated_summaries, reference_summaries):
    total_scores = {"rouge-1": {"f": 0, "p": 0, "r": 0},
                    "rouge-2": {"f": 0, "p": 0, "r": 0},
                    "rouge-l": {"f": 0, "p": 0, "r": 0}}
    total_bleu = 0

    for summary, reference in zip(generated_summaries, reference_summaries):
        bleu, rouge_scores = compute_scores(reference, summary)
        total_bleu += bleu

        for key in total_scores:
            total_scores[key]["f"] += rouge_scores[key]["f"]
            total_scores[key]["p"] += rouge_scores[key]["p"]
            total_scores[key]["r"] += rouge_scores[key]["r"]

    num_samples = len(generated_summaries)
    average_bleu = total_bleu / num_samples
    for key in total_scores:
        total_scores[key]["f"] /= num_samples
        total_scores[key]["p"] /= num_samples
        total_scores[key]["r"] /= num_samples

    return average_bleu, total_scores

datasets_list = ["samsum", "cnn_dailymail", "xsum", "gigaword", "multi_news"]
dataset_versions = {
    "cnn_dailymail": "3.0.0",
    "xsum": "1.2.0",
    "gigaword": "1.2.0",
    "multi_news": "1.0.0",
    "samsum": "1.0.0"
}
dataset_column_names = {
    "cnn_dailymail": {"article": "article", "summary": "highlights"},
    "xsum": {"article": "document", "summary": "summary"},
    "gigaword": {"article": "document", "summary": "summary"},
    "multi_news": {"article": "document", "summary": "summary"},
    "samsum": {"article": "dialogue", "summary": "summary"}
}
records_to_select = {
   "samsum": 819,
    "cnn_dailymail": 11490,
    "xsum": 11334,
    "gigaword": 1951,
    "multi_news": 5622
}

loaded_datasets = {}
for dataset_name in tqdm(datasets_list, desc="Loading Datasets", leave=True):
    if dataset_name in ["samsum"]:
        dataset = load_dataset(dataset_name, split="test").select(range(records_to_select[dataset_name]))
    else:
        dataset = load_dataset(dataset_name, dataset_versions[dataset_name], split="test").select(range(records_to_select[dataset_name]))
    num_records = records_to_select.get(dataset_name, len(dataset))
    dataset = dataset.select(range(min(len(dataset), num_records)))
    loaded_datasets[dataset_name] = dataset

results = []
summaries = []

for dataset_name in tqdm(datasets_list, desc="Processing Datasets", leave=True):
    dataset = loaded_datasets[dataset_name]
    generated_summaries, reference_summaries = execute(dataset, dataset_name)
    bleu, scores = compute_all_scores(generated_summaries, reference_summaries)
    results.append([dataset_name, bleu] + [scores[key]["f"] for key in scores])

    for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
        summaries.append([dataset_name] + [gen_summary, ref_summary])

df = pd.DataFrame(results, columns=["Dataset", "BLEU", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"])
display(df)

df1 = pd.DataFrame(summaries, columns=["Dataset", "Generated Summary", "Reference Summary"])
display(df1)

df1.to_csv('summaries_bart.csv', index=False)