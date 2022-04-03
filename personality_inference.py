import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
from torch.cuda.amp import autocast
import argparse
import os
import time, datetime
from datetime import datetime
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from datasets import Dataset
from scipy.stats import pearsonr
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


class Model(nn.Module):
    def __init__(self, transformer, d_model, n_class=5):
        super().__init__()
        self.base_encoder = transformer
        self.out = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 512), nn.Tanh(),
                                                nn.Linear(512, 128), nn.Tanh(),
                                                nn.Linear(128, 1), nn.Tanh()) for _ in range(n_class)])

    def forward(self, batch):
        op = self.base_encoder(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"])
        op = torch.stack(op["hidden_states"])[-4:].mean(0).mean(1)  # batch, h_dim

        preds = []
        for encoder in self.out:
            preds.append(encoder(op))
        return torch.cat(preds, -1)


class PersonalityPredictor(object):
    def __init__(self, n_classes, model_path, device_num=None):
        self.DEVICE = torch.device("cpu") if device_num is None else torch.device("cuda:{}".format(device_num))
        print("Model device selcted:", self.DEVICE)
        self.MODEL_PATH = model_path
        self.MAX_LEN = 300
        self.tokenizer, self.model = self._load_model(n_classes)
        self.batch_size = 8 if device_num is None else 128

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _load_model(self, n_class):
        mname = "roberta-base"
        # n_class = 5
        config = RobertaConfig.from_pretrained(mname)
        config.output_hidden_states = True
        base = RobertaModel.from_pretrained(mname, config=config)
        model = Model(base, config.hidden_size, n_class).to(self.DEVICE)
        for name, param in model.named_parameters():
            if "pooler" in name:
                param.requires_grad = False
        tokenizer = RobertaTokenizerFast.from_pretrained(mname)
        model.load_state_dict(torch.load(self.MODEL_PATH))
        model.eval()
        print(f'The model has {self.count_parameters(model):,} trainable parameters')
        return tokenizer, model

    def _process_input(self, inpt):
        if type(inpt) != list:
            inpt = [inpt]
        test_dict = {"x": inpt}
        test_dataset = Dataset.from_dict(test_dict)
        test_dataset_mapped = test_dataset.map(lambda e: self.tokenizer(e['x'], truncation=True, padding=True,
                                                                        max_length=300), batched=True,
                                               batch_size=self.batch_size)
        test_dataset_mapped.set_format(type='torch', columns=['attention_mask', 'input_ids'])
        test_dataloader = DataLoader(dataset=test_dataset_mapped, sampler=SequentialSampler(test_dataset_mapped),
                                     batch_size=self.batch_size)
        return test_dataloader

    def predict(self, inpt):
        dataloader = self._process_input(inpt)

        predicted = []
        for ix, batch in tqdm(enumerate(dataloader)):
            batch = {k: v.to(self.DEVICE) for k, v in batch.items()}
            with autocast():
                with torch.no_grad():
                    pred = self.model(batch)
            predicted.extend(pred.tolist())
        return np.asarray(predicted)


def process_context(txt, ref):
    all_ctx = [i.replace("<agent_2>", "").strip() for i in
               txt.replace("<agent_1>", "#<agent_1>").replace("<agent_2>", "#<agent_2>").split("#") if
               len(i) > 0 and i.startswith("<agent_2>")]
    return " ".join(all_ctx) + ref


def calculate_correlation(actual, pred):
    agreeableness = pearsonr(actual[:, 0], pred[:, 0])[0]
    openness = pearsonr(actual[:, 1], pred[:, 1])[0]
    conscientiousness = pearsonr(actual[:, 2], pred[:, 2])[0]
    extraversion = pearsonr(actual[:, 3], pred[:, 3])[0]
    neuroticism = pearsonr(actual[:, 4], pred[:, 4])[0]
    vals = [round(agreeableness, 5), round(openness, 5), round(conscientiousness, 5), round(extraversion, 5),
            round(neuroticism, 5)]
    return vals + [round(np.mean(vals), 5)]


def main():
    st = time.time()
    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')#datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    # consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_personality_traits.csv"
    # consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_personality_traits_all.csv"
    consolidated_results = "/home/sougatas/wikibot_naacl_2022/results/consolidated_results_personality_traits_all_v2.csv"
    op_file = "personality_pred_output_" + dt + ".tsv"
    device_num, model_type = None, None
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--text", type=str,
                        help="Single text on which Personality Pred is to be performed", default="")
    parser.add_argument("--input_pickle_file", type=str, help="The input file is direct output of training process",
                        default="")
    parser.add_argument("--output_file", type=str, help="Results will be written to output file if provided.",
                        default="")
    # parser.add_argument("--model_type", type=str, help="Type of personality model",
    #                     default=model_type)
    parser.add_argument("--device_num", type=int, help="CUDA device number",
                        default=device_num)

    argv = parser.parse_args()
    text = argv.text
    input_pickle_file = argv.input_pickle_file
    output_file = argv.output_file
    # model_type = argv.model_type
    device_num = argv.device_num

    if input_pickle_file != "":
        print("Processing for file:", input_pickle_file)

        log = pickle.load(open(input_pickle_file, "rb"))

        experiment_num = int(input_pickle_file.split("experiment_")[-1].split(".")[0])
        encoder, decoder = log["config"]["encoder"], log["config"]["decoder"]

        # if "pandora" in encoder.lower() or "pandora" in decoder.lower():
        #     model_path = "/home/sougatas/wikibot_naacl_2022/models/additional_models/pandora_big5_model_roberta-base.pt"
        # else:
        #     model_path = "/home/sougatas/wikibot_naacl_2022/models/additional_models/essays_big5_model_roberta-base.pt"
        model_path = "/home/sougatas/wikibot_naacl_2022/models/additional_models/essays_big5_model_roberta-base.pt"

        print("Model path:",model_path)
        test_1_auto_metrics = [k for k in log["test"].keys() if "test_1_auto_metrics" in k]
        test_2_auto_metrics = [k for k in log["test"].keys() if "test_2_auto_metrics" in k]
        assert len(test_1_auto_metrics) == len(test_2_auto_metrics) == 1
        test_1_auto_metrics = test_1_auto_metrics[0]
        test_2_auto_metrics = test_2_auto_metrics[0]
        ds = "topical_chat" if "topical_chat" in test_1_auto_metrics else "wizard_of_wikipedia"

        test_1_cor_samples = [process_context(i, log["test"][test_1_auto_metrics]["cor"][ix]) for ix, i in
                              enumerate(log["test"][test_1_auto_metrics]["context"])]
        test_2_cor_samples = [process_context(i, log["test"][test_2_auto_metrics]["cor"][ix]) for ix, i in
                              enumerate(log["test"][test_2_auto_metrics]["context"])]
        test_1_hyp_samples = [process_context(i, log["test"][test_1_auto_metrics]["hyp"][ix]) for ix, i in
                              enumerate(log["test"][test_1_auto_metrics]["context"])]
        test_2_hyp_samples = [process_context(i, log["test"][test_2_auto_metrics]["hyp"][ix]) for ix, i in
                              enumerate(log["test"][test_2_auto_metrics]["context"])]

        predictor = PersonalityPredictor(5, model_path, device_num)
        print("Starting Predictions")
        test_1_cor_samples_pred = predictor.predict(test_1_cor_samples)
        test_2_cor_samples_pred = predictor.predict(test_2_cor_samples)
        test_1_hyp_samples_pred = predictor.predict(test_1_hyp_samples)
        test_2_hyp_samples_pred = predictor.predict(test_2_hyp_samples)

        test1_correl = calculate_correlation(test_1_cor_samples_pred, test_1_hyp_samples_pred)
        test2_correl = calculate_correlation(test_2_cor_samples_pred, test_2_hyp_samples_pred)
        print("test1_correl", test1_correl)
        print("test2_correl", test2_correl)
        combined = [experiment_num, encoder, decoder, ds] + test1_correl + test2_correl
        print("combined", combined)
        df = pd.DataFrame([combined], columns=["id", "encoder", "decoder", "dataset", "T1_agreeableness", "T1_openness",
                                             "T1_conscientiousness", "T1_extraversion", "T1_neuroticism", "T1_overall",
                                             "T2_agreeableness", "T2_openness",  "T2_conscientiousness",
                                             "T2_extraversion", "T2_neuroticism", "T2_overall"])
        df["created_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if os.path.isfile(consolidated_results):
            consolidated_results_df = pd.read_csv(consolidated_results)
            consolidated_results_df = pd.concat([consolidated_results_df, df])
        else:
            consolidated_results_df = df

        consolidated_results_df.to_csv(consolidated_results, index=False)
        print(consolidated_results_df)
        print("time taken:", time.time()-st)
    else:
        print("Not a valid combination!")


if __name__ == '__main__':
    main()
