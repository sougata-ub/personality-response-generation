import argparse
import pickle
from datasets import load_metric
import numpy as np
import pandas as pd
import shutil
import time
from datetime import datetime
import nltk
from nltk import sent_tokenize
from sklearn.metrics import f1_score, accuracy_score
from bleurt import score
import os
import os.path
import torch

nltk.download('punkt')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_v3.csv"
    # consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_all.csv"
    consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_all_v2.csv"
    # log_folder = "/home/sougatas/wikibot_naacl_2022/results/"

    parser.add_argument("--fname", type=str, help="Log file")
    argv = parser.parse_args()
    fname = argv.fname
    print("Got fname:", fname)
    log = pickle.load(open(fname, "rb"))
    experiment_num = int(fname.split("experiment_")[-1].split(".")[0])
    encoder, decoder = log["config"]["encoder"], log["config"]["decoder"]
    print("experiment_num:", experiment_num)

    test_1_auto_metrics = [k for k in log["test"].keys() if "test_1_auto_metrics" in k]
    test_2_auto_metrics = [k for k in log["test"].keys() if "test_2_auto_metrics" in k]

    assert len(test_1_auto_metrics) == len(test_2_auto_metrics) == 1
    test_1_auto_metrics = test_1_auto_metrics[0]
    test_2_auto_metrics = test_2_auto_metrics[0]

    print("Calculating bert scores")
    bertscore = load_metric("bertscore")
    bert_score1 = bertscore.compute(predictions=log["test"][test_1_auto_metrics]["hyp"],
                                    references=log["test"][test_1_auto_metrics]["cor"], lang="en")
    torch.cuda.empty_cache()
    bert_score2 = bertscore.compute(predictions=log["test"][test_2_auto_metrics]["hyp"],
                                    references=log["test"][test_2_auto_metrics]["cor"], lang="en")
    torch.cuda.empty_cache()
    bert_score11 = bertscore.compute(predictions=log["test"][test_1_auto_metrics]["non_golden_dict"]["hyp"],
                                     references=log["test"][test_1_auto_metrics]["non_golden_dict"]["cor"], lang="en")
    torch.cuda.empty_cache()
    bert_score22 = bertscore.compute(predictions=log["test"][test_2_auto_metrics]["non_golden_dict"]["hyp"],
                                     references=log["test"][test_2_auto_metrics]["non_golden_dict"]["cor"], lang="en")
    torch.cuda.empty_cache()

    print("Calculating bleurt score")
    bleurt = score.BleurtScorer("../BLEURT-20")
    bleurt_score1 = bleurt.score(candidates=log["test"][test_1_auto_metrics]["hyp"],
                                references=log["test"][test_1_auto_metrics]["cor"], batch_size=8)
    bleurt_score2 = bleurt.score(candidates=log["test"][test_2_auto_metrics]["hyp"],
                                 references=log["test"][test_2_auto_metrics]["cor"], batch_size=8)
    bleurt_score11 = bleurt.score(candidates=log["test"][test_1_auto_metrics]["non_golden_dict"]["hyp"],
                                references=log["test"][test_1_auto_metrics]["non_golden_dict"]["cor"], batch_size=8)
    bleurt_score22 = bleurt.score(candidates=log["test"][test_2_auto_metrics]["non_golden_dict"]["hyp"],
                                 references=log["test"][test_2_auto_metrics]["non_golden_dict"]["cor"], batch_size=8)
    torch.cuda.empty_cache()
    ds = "topical_chat" if "topical_chat" in test_1_auto_metrics else "wizard_of_wikipedia"
    metrics = [[experiment_num, encoder, decoder, "gold", round(log["test"][ds + "_test_1"][7], 5),
                round(log["test"][test_1_auto_metrics]["bleu"], 5),
                round(log["test"][test_1_auto_metrics]["bleu4"], 5),
                round(log["test"][test_1_auto_metrics]["rougeL"], 5),
                round(np.mean(bert_score1["f1"]), 5), round(np.percentile(bert_score1["f1"], 50), 5),
                round(np.mean(bleurt_score1), 5), round(np.percentile(bleurt_score1, 50), 5),
                round(log["test"][ds + "_test_2"][7], 5),
                round(log["test"][test_2_auto_metrics]["bleu"], 5),
                round(log["test"][test_2_auto_metrics]["bleu4"], 5),
                round(log["test"][test_2_auto_metrics]["rougeL"], 5),
                round(np.mean(bert_score2["f1"]), 5), round(np.percentile(bert_score2["f1"], 50), 5),
                round(np.mean(bleurt_score2), 5), round(np.percentile(bleurt_score2, 50), 5)],

               [experiment_num, encoder, decoder, "non-gold", round(log["test"][ds + "_test_1"][7], 5),
                round(log["test"][test_1_auto_metrics]["non_golden_dict"]["bleu"], 5),
                round(log["test"][test_1_auto_metrics]["non_golden_dict"]["bleu4"], 5),
                round(log["test"][test_1_auto_metrics]["non_golden_dict"]["rougeL"], 5),
                round(np.mean(bert_score11["f1"]), 5), round(np.percentile(bert_score11["f1"], 50), 5),
                round(np.mean(bleurt_score11), 5), round(np.percentile(bleurt_score11, 50), 5),
                round(log["test"][ds + "_test_2"][7], 5),
                round(log["test"][test_2_auto_metrics]["non_golden_dict"]["bleu"], 5),
                round(log["test"][test_2_auto_metrics]["non_golden_dict"]["bleu4"], 5),
                round(log["test"][test_2_auto_metrics]["non_golden_dict"]["rougeL"], 5),
                round(np.mean(bert_score22["f1"]), 5), round(np.percentile(bert_score22["f1"], 50), 5),
                round(np.mean(bleurt_score22), 5), round(np.percentile(bleurt_score22, 50), 5)]]
    # metrics = [[experiment_num, "test_1", round(np.mean(bert_score["f1"]), 4), round(np.percentile(bert_score["f1"], 50), 4),
    #             round(np.percentile(bert_score["f1"], 90), 4), round(np.mean(bleurt_score), 4),
    #             round(np.percentile(bleurt_score, 50), 4),
    #             round(np.percentile(bleurt_score, 90), 4),
    #             log["test"][test_1_auto_metrics]["bleu"], log["test"][test_1_auto_metrics]["bleu4"],
    #             log["test"][test_1_auto_metrics]["rougeL"], log["test"][ds+"_test_1"][7]],
    #            [experiment_num, "test_2", round(np.mean(bert_score2["f1"]), 4), round(np.percentile(bert_score2["f1"], 50), 4),
    #             round(np.percentile(bert_score2["f1"], 90), 4), round(np.mean(bleurt_score2), 4),
    #             round(np.percentile(bleurt_score2, 50), 4),
    #             round(np.percentile(bleurt_score2, 90), 4),
    #             log["test"][test_2_auto_metrics]["bleu"], log["test"][test_2_auto_metrics]["bleu4"],
    #             log["test"][test_2_auto_metrics]["rougeL"], log["test"][ds+"_test_2"][7]]
    #            ]

    # all_metrics = pd.DataFrame(metrics, columns=["split", "bertscore_mean", "bertscore_median",
    #                                              "bertscore_90pct", "bleurt_mean", "bleurt_median",
    #                                              "bleurt_90pct", "id", "bleu_score", "bleu_4", "rougeL", "PPL"])
    all_metrics = pd.DataFrame(metrics, columns=["id", "encoder", "decoder", "split", "T1_PPL", "T1_BLEU", "T1_BLEU4",
                                                 "T1_RougeL", "T1_bertscore_mean", "T1_bertscore_median",
                                                 "T1_bleurt_mean", "T1_bleurt_median",
                                                 "T2_PPL", "T2_BLEU", "T2_BLEU4",
                                                 "T2_RougeL", "T2_bertscore_mean", "T2_bertscore_median",
                                                 "T2_bleurt_mean", "T2_bleurt_median"])
    all_metrics["created_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if os.path.isfile(consolidated_results):
        consolidated_results_df = pd.read_csv(consolidated_results)
        consolidated_results_df = pd.concat([consolidated_results_df, all_metrics])
    else:
        consolidated_results_df = all_metrics

    consolidated_results_df.to_csv(consolidated_results, index=False)
    print(consolidated_results_df)


if __name__ == '__main__':
    main()
