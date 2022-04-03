from IntentModel import SequenceClassifier
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os, sys
import logging
import time, datetime
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


class IPredictor(object):
    def __init__(self, model_path, device_num):
        self.num_classes = 12
        self.n_layers = 4
        self.DEVICE = torch.device("cpu") if device_num is None else torch.device("cuda:{}".format(device_num))
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

        model = SequenceClassifier(transformer_model, config, self.n_layers, self.num_classes)#.to(self.DEVICE)

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


def main():
    st = time.time()

    dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    logging.basicConfig(filename="intent_classifier_"+ dt +".log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info("Logger Initialized")

    op_file = "intent_output_" + dt+".tsv"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--text", type=str,
                        help="Single text on which Intent Classification is to be performed", default="")
    parser.add_argument("--input_file", type=str, help="Tab separated tsv file, with columns in first line",
                        default="")
    parser.add_argument("--input_file_text", type=str, help="Column name containing text",
                        default="sentence")
    parser.add_argument("--output_file", type=str, help="Results will be written to output file if provided.",
                        default="")
    parser.add_argument("--model_path", type=str, help="Path of NER model",
                        default="../drive/MyDrive/bert_intent_model/intent_classifier_bert")

    argv = parser.parse_args()
    text = argv.text
    input_file = argv.input_file
    output_file = argv.output_file
    input_file_text = argv.input_file_text
    model_path = argv.model_path

    if text == "" and input_file == "":
        print('No text or input file provided')
        logging.info('No text or input file provided')
        sys.exit()

    if input_file != "" and output_file == "":
        output_file = op_file
        print("No output file provided. Will write outputs to :{}".format(output_file))
        logging.info("No output file provided. Will write outputs to :{}".format(output_file))

    """ Creating a List of sentences """
    if input_file != "":
        try:
            df = pd.read_csv(input_file, sep="\t")
            text = list(df[input_file_text])
        except Exception as e:
            print("Error opening file")
            logging.info('Exception while opening file: {}'.format(e))
            sys.exit()
    else:
        text = [text]

    predictor = IPredictor(model_path)
    pred = predictor.predict(text)

    if input_file != "":
        df["intent"] = pred
        df.to_csv(output_file, sep="\t", index=False)
        print("Output written to file")
        logging.info("Output written to file")
    else:
        print("Text:{}, has Intent: {}".format(text, pred))

    et = time.time()
    print("Took ", str(et - st), "ms to execute")
    logging.info("Took :{} s to execute".format(str(et - st)))


if __name__ == "__main__":
    main()