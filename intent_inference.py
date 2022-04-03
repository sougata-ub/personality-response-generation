import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from torch.cuda.amp import autocast
from transformers import BertTokenizer, BertModel, BertConfig
import argparse
import os
import time, datetime
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
from datasets import Dataset
from scipy.stats import pearsonr
import pickle
# from IntentPredictor import IPredictor
from sklearn.metrics import f1_score, accuracy_score
import nltk
from nltk import sent_tokenize

nltk.download('punkt')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


class SequenceClassifier(torch.nn.Module):
    def __init__(self, transformer_model, config, n_layers=4, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.transformer = transformer_model
        self.out = torch.nn.Linear(config.hidden_size * 2, self.num_classes)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask, classification_labels=None):

        # Batch max length
        max_length = (attention_mask != 0).max(0)[0].nonzero()[-1].item() + 1
        if max_length < input_ids.shape[1]:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]

        segment_id = torch.zeros_like(attention_mask)
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                  token_type_ids=segment_id)

        token_hidden = hidden[2][-self.n_layers:]
        token_hidden = torch.mean(torch.sum(torch.stack(token_hidden), dim=0),
                                  dim=1)

        classifier_hidden = hidden[1]
        hidden_cat = torch.cat([token_hidden, classifier_hidden], dim=1)

        classification_logits = self.out(self.dropout(hidden_cat))
        outputs = [classification_logits]
        if classification_labels is not None:
            loss_fct_classification = torch.nn.CrossEntropyLoss()

            loss_classification = loss_fct_classification(classification_logits.view(-1, self.num_classes),
                                                          classification_labels.view(-1))

            outputs += [loss_classification]
        return outputs


class IPredictor(object):
    def __init__(self, model_path, device_num):
        self.num_classes = 12
        self.n_layers = 4
        self.DEVICE = torch.device("cpu") if device_num is None else torch.device("cuda:{}".format(device_num))
        print("Device:",self.DEVICE)
        # self.DEVICE = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else torch.device('cpu')
        self.model, self.tokenizer = self._load_model(model_path)
        self.max_length = 100
        self.batch_size = 64
        self.INTENT_MAP = {0: 'Acknowledgment', 1: 'Clarification', 2: 'General chat', 3: 'Other',
                           4: 'Rejection', 5: 'Request Knowledge Fact', 6: 'Request Personal Fact',
                           7: 'Request opinion', 8: 'State Knowledge Fact', 9: 'State Personal Fact',
                           10: 'State opinion', 11: 'Topic suggestion/Topic Switch'}

    def _load_model(self, model_path):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        config = BertConfig.from_pretrained("bert-base-uncased")
        transformer_model = BertModel.from_pretrained("bert-base-uncased", config=config)
        config.output_hidden_states = True

        model = SequenceClassifier(transformer_model, config, self.n_layers, self.num_classes).to(self.DEVICE)

        model.load_state_dict(torch.load('{model_path}'.format(model_path=model_path)))
        model.eval()
        print("\nIntent Model Loaded\n")
        return model, tokenizer

    def _process_input(self, text_list):
        enc = self.tokenizer(text_list, padding=True, max_length=self.max_length,
                             truncation=True)
        ids = torch.tensor(enc["input_ids"])
        attn = torch.tensor(enc["attention_mask"])
        print("Inputs processed, ids: {}, attn: {}".format(ids.shape, attn.shape))

        t_data = TensorDataset(ids, attn)
        sampler = SequentialSampler(t_data)
        data_loader = DataLoader(t_data, sampler=sampler, batch_size=min(self.batch_size, ids.shape[0]))
        return data_loader

    def predict(self, text_list):
        data_loader = self._process_input(text_list)
        class_assigned_list = []

        for batch in tqdm(data_loader):
            batch = tuple(t.to(self.DEVICE) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, b_input_mask)

            classification_logits = F.softmax(outputs[0].detach().cpu(), -1).numpy()
            class_assigned, confidence = np.argmax(classification_logits, 1).tolist(), np.max(classification_logits,
                                                                                              1).tolist()
            class_assigned_list.extend(class_assigned)
        print("{} Intent predictions made".format(len(class_assigned_list)))
        return [self.INTENT_MAP[i] for i in class_assigned_list]


def intent2strategy(intent_list):
    sh_exp, sh_kg, se_exp, se_kg = 0, 0, 0, 0
    if "State Personal Fact" in intent_list or "State opinion" in intent_list:
        sh_exp = 1
    if "State Knowledge Fact" in intent_list:
        sh_kg = 1
    if "Request Personal Fact" in intent_list or "Request opinion" in intent_list:
        se_exp = 1
    if "Request Knowledge Fact" in intent_list:
        se_kg = 1
    return [sh_exp, sh_kg, se_exp, se_kg]


def get_strategy(text_list, intent_predictor):
    st = time.time()
    examples = []
    for ix, text in enumerate(text_list):
        sents = sent_tokenize(text)
        if len(sents) == 0:
            print(ix)
            sents = [""]
        examples.extend(list(zip(sents, [ix * len(sents)], [text] * len(sents))))
    text_df_tokenized = pd.DataFrame(examples, columns=["sentence", "idx", "text"])

    intent_pred = intent_predictor.predict(list(text_df_tokenized["sentence"]))
    text_df_tokenized["intent"] = intent_pred
    text_df_tokenized.groupby(["idx", "text"]).agg({"sentence": list, "intent": list}).reset_index()
    text_df_tokenized["strategy"] = text_df_tokenized["intent"].apply(intent2strategy)
    print("Intent Predictor Took ", str(time.time() - st), "ms to execute")
    return np.asarray(list(text_df_tokenized["strategy"]))


def main():
    intent_model_path = "/home/sougatas/wikibot_naacl_2022/models/additional_models/intent_classifier_bert"
    # consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_intents.csv"
    # consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_intents_all.csv"
    consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_intents_all_v2.csv"
    device_num = None

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_pickle_file", type=str, help="The input file is direct output of training process",
                        default="")
    parser.add_argument("--device_num", type=int, help="CUDA device number",
                        default=device_num)
    argv = parser.parse_args()
    input_pickle_file = argv.input_pickle_file
    device_num = argv.device_num

    print("Processing for file:", input_pickle_file)

    log = pickle.load(open(input_pickle_file, "rb"))

    experiment_num = int(input_pickle_file.split("experiment_")[-1].split(".")[0])
    encoder, decoder = log["config"]["encoder"], log["config"]["decoder"]

    test_1_auto_metrics = [k for k in log["test"].keys() if "test_1_auto_metrics" in k]
    test_2_auto_metrics = [k for k in log["test"].keys() if "test_2_auto_metrics" in k]
    assert len(test_1_auto_metrics) == len(test_2_auto_metrics) == 1
    test_1_auto_metrics = test_1_auto_metrics[0]
    test_2_auto_metrics = test_2_auto_metrics[0]

    ds = "topical_chat" if "topical_chat" in test_1_auto_metrics else "wizard_of_wikipedia"

    intent_predictor = IPredictor(intent_model_path, device_num)

    test_1_hyp = get_strategy(log["test"][test_1_auto_metrics]["hyp"], intent_predictor)
    test_1_cor = get_strategy(log["test"][test_1_auto_metrics]["cor"], intent_predictor)
    test_2_hyp = get_strategy(log["test"][test_2_auto_metrics]["hyp"], intent_predictor)
    test_2_cor = get_strategy(log["test"][test_2_auto_metrics]["cor"], intent_predictor)

    print("\nStrategy Array Shapes: ", test_1_hyp.shape, test_1_cor.shape, test_2_hyp.shape, test_2_cor.shape)

    share_experience_test_1_f1, share_experience_test_1_acc = f1_score(test_1_cor[:, 0], test_1_hyp[:, 0]), \
                                                              accuracy_score(test_1_cor[:, 0], test_1_hyp[:, 0])
    share_knowledge_test_1_f1, share_knowledge_test_1_acc = f1_score(test_1_cor[:, 1], test_1_hyp[:, 1]), \
                                                            accuracy_score(test_1_cor[:, 1], test_1_hyp[:, 1])
    seek_experience_test_1_f1, seek_experience_test_1_acc = f1_score(test_1_cor[:, 2], test_1_hyp[:, 2]), \
                                                            accuracy_score(test_1_cor[:, 2], test_1_hyp[:, 2])
    seek_knowledge_test_1_f1, seek_knowledge_test_1_acc = f1_score(test_1_cor[:, 3], test_1_hyp[:, 3]), \
                                                          accuracy_score(test_1_cor[:, 3], test_1_hyp[:, 3])

    share_experience_test_2_f1, share_experience_test_2_acc = f1_score(test_2_cor[:, 0], test_2_hyp[:, 0]), \
                                                              accuracy_score(test_2_cor[:, 0], test_2_hyp[:, 0])
    share_knowledge_test_2_f1, share_knowledge_test_2_acc = f1_score(test_2_cor[:, 1], test_2_hyp[:, 1]), \
                                                            accuracy_score(test_2_cor[:, 1], test_2_hyp[:, 1])
    seek_experience_test_2_f1, seek_experience_test_2_acc = f1_score(test_2_cor[:, 2], test_2_hyp[:, 2]), \
                                                            accuracy_score(test_2_cor[:, 2], test_2_hyp[:, 2])
    seek_knowledge_test_2_f1, seek_knowledge_test_2_acc = f1_score(test_2_cor[:, 3], test_2_hyp[:, 3]), \
                                                          accuracy_score(test_2_cor[:, 3], test_2_hyp[:, 3])

    combined = [experiment_num, encoder, decoder, ds] + \
               [share_experience_test_1_f1, share_knowledge_test_1_f1, seek_experience_test_1_f1,
                seek_knowledge_test_1_f1, share_experience_test_2_f1, share_knowledge_test_2_f1,
                seek_experience_test_2_f1, seek_knowledge_test_2_f1] + \
               [share_experience_test_1_acc, share_knowledge_test_1_acc, seek_experience_test_1_acc,
                seek_knowledge_test_1_acc, share_experience_test_2_acc, share_knowledge_test_2_acc,
                seek_experience_test_2_acc, seek_knowledge_test_2_acc]
    print(combined)
    df = pd.DataFrame([combined], columns=["id", "encoder", "decoder", "dataset", "share_experience_T1_f1",
                                         "share_knowledge_T1_f1", "seek_experience_T1_f1", "seek_knowledge_T1_f1",
                                         "share_experience_T2_f1", "share_knowledge_T2_f1", "seek_experience_T2_f1",
                                         "seek_knowledge_T2_f1", "share_experience_T1_acc", "share_knowledge_T1_acc",
                                         "seek_experience_T1_acc", "seek_knowledge_T1_acc", "share_experience_T2_acc",
                                         "share_knowledge_T2_acc",  "seek_experience_T2_acc", "seek_knowledge_T2_acc"])
    df["created_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if os.path.isfile(consolidated_results):
        consolidated_results_df = pd.read_csv(consolidated_results)
        consolidated_results_df = pd.concat([consolidated_results_df, df])
    else:
        consolidated_results_df = df

    consolidated_results_df.to_csv(consolidated_results, index=False)
    print(consolidated_results_df)


if __name__ == '__main__':
    main()
