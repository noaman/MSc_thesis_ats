import warnings
from datasets import load_dataset
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')

LANGUAGE = "english"

def classical_summary(text, method="lsa", num_sentences=10):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    if method == "lsa":
        summarizer = LsaSummarizer(stemmer)
    elif method == "textrank":
        summarizer = TextRankSummarizer(stemmer)
    elif method == "sumbasic":
        summarizer = SumBasicSummarizer(stemmer)
    else:
        raise ValueError("Invalid summarization method")

    summarizer.stop_words = get_stop_words(LANGUAGE)
    try:
        return " ".join([str(sentence) for sentence in summarizer(parser.document, num_sentences)])
    except KeyError:
        return " ".join(text.split(".")[:num_sentences]) + "."

def compute_scores(reference, summary):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(summary, reference)
    bleu_score = sentence_bleu([reference.split()], summary.split())
    return bleu_score, rouge_scores[0]

def execute(test_data, dataset_name, method="lsa", batch_size=1):
    results = []
    summaries = []
    total_scores = {"rouge-1": {"f": 0, "p": 0, "r": 0},
                    "rouge-2": {"f": 0, "p": 0, "r": 0},
                    "rouge-l": {"f": 0, "p": 0, "r": 0}}
    total_bleu = 0
    article_column = dataset_column_names[dataset_name]["article"]
    summary_column = dataset_column_names[dataset_name]["summary"]
    count_summaries = 10
    count_summaries_ctr = 0
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        articles = batch[article_column]
        highlights = batch[summary_column]
        generated_summaries = [classical_summary(article, method) for article in articles]
        if count_summaries_ctr < count_summaries:
            summaries.extend(zip([dataset_name]*batch_size, generated_summaries, highlights))
            count_summaries_ctr += batch_size

        for j, (reference, summary) in enumerate(zip(highlights, generated_summaries)):
            if not summary.strip():
                continue
            bleu, rouge_scores = compute_scores(reference, summary)
            total_bleu += bleu
            for key in total_scores:
                total_scores[key]["f"] += rouge_scores[key]["f"]
                total_scores[key]["p"] += rouge_scores[key]["p"]
                total_scores[key]["r"] += rouge_scores[key]["r"]

    num_samples = len(test_data)
    average_bleu = total_bleu / num_samples
    for key in total_scores:
        total_scores[key]["f"] /= num_samples
        total_scores[key]["p"] /= num_samples
        total_scores[key]["r"] /= num_samples

    results.append([dataset_name, method, average_bleu] + [total_scores[key]["f"] for key in total_scores])
    return results, summaries

datasets_list = ["samsum","cnn_dailymail", "xsum", "gigaword", "multi_news"]
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




methods = ["lsa", "textrank", "sumbasic"]

loaded_datasets = {}

# Load the datasets first, outside of the methods loop
for dataset_name in tqdm(datasets_list, desc="Loading Datasets", leave=True):
    if dataset_name in ["samsum"]:
        dataset = load_dataset(dataset_name, split="test").select(range(records_to_select[dataset_name]))
    else:
        dataset = load_dataset(dataset_name, dataset_versions[dataset_name], split="test").select(range(records_to_select[dataset_name]))


    num_records = records_to_select.get(dataset_name, len(dataset))
    dataset = dataset.select(range(min(len(dataset), num_records)))

    loaded_datasets[dataset_name] = dataset

for method in tqdm(methods, desc="Processing Methods", leave=True):
    all_datasets_results = []
    all_datasets_summaries = []

    for dataset_name in tqdm(datasets_list, desc="Processing Datasets for method: " + method, leave=True, position=0):

        dataset = loaded_datasets[dataset_name]  # Get the pre-loaded dataset

        print(dataset_name, len(dataset))
        results, summaries = execute(dataset, dataset_name, method=method)
        all_datasets_results.extend(results)
        all_datasets_summaries.extend(summaries)

    df_all = pd.DataFrame(all_datasets_results, columns=["Dataset", "Method", "BLEU", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"])
    df_summaries_all = pd.DataFrame(all_datasets_summaries, columns=["Dataset", "Generated Summary", "Reference Summary"])
    # df_all.to_csv(f'{method}_all_datasets_results_classical.csv', index=False)
    df_summaries_all.to_csv(f'{method}_all_datasets_summaries_classical.csv', index=False)
    print(method)
    print(df_all)

    print("**************************")