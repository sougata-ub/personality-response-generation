import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import time
from tqdm import tqdm
import math
import argparse
import os
import random
import numpy as np
import copy
from datasets import load_metric
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from scipy.stats import pearsonr
from datasets import Dataset
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from IntentPredictor import IPredictor
from Big5Predictor import Big5Predictor
from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotConfig, BlenderbotForConditionalGeneration
from transformers.models.blenderbot.tokenization_blenderbot import BlenderbotTokenizer
import pandas as pd
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk import sent_tokenize

nltk.download('punkt')

# os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['LOCAL_RANK'] = "2"


class Trainer:
    def __init__(self, mname, ablation_remove, dataset, encoder_layers, training_file, trait_in_decoder=True,
                 local_rank=0, batch_size=8, num_workers=2, num_train=None, uni_agent=True, lr=2e-5, N_EPOCHS=8,
                 accum_iter=4, contradiction_penalty=False, intent_model_path=None, big5_model_path=None,
                 loss_alpha=0.7, pandora_personality=True, skip_training=False, use_gpu=True):

        self.set_random_seeds(random_seed=42)
        self.mname, self.trait_in_decoder = mname, trait_in_decoder
        self.ablation_remove, self.dataset, self.uni_agent = ablation_remove, dataset, uni_agent
        self.batch_size, self.accum_iter, self.contradiction_penalty = batch_size, accum_iter, contradiction_penalty
        self.num_workers, self.intent_model_path, self.big5_model_path = num_workers, None, None #intent_model_path, big5_model_path
        self.num_train, self.loss_alpha = num_train, loss_alpha
        self.lr, self.N_EPOCHS = lr, N_EPOCHS
        self.pandora_personality = pandora_personality

        if self.intent_model_path is not None:
            self.intent_predictor = IPredictor(intent_model_path)

        if self.big5_model_path is not None:
            self.big5_predictor = Big5Predictor(big5_model_path)

        self.big5_codes = ['<low_agreeableness>', '<high_agreeableness>',
                           '<low_openness>', '<high_openness>',
                           '<low_conscientiousness>', '<high_conscientiousness>',
                           '<low_extraversion>', '<high_extraversion>',
                           '<low_neuroticism>', '<high_neuroticism>']
        # self.big5_codes = ['<high_agreeableness>', '<high_openness>', '<high_conscientiousness>',
        #                    '<high_extraversion>', '<high_neuroticism>']

        self.corpus_based_codes = ['<low_opposing>', '<moderate_opposing>', '<fair_opposing>',
                                   '<prefers_experience>', '<prefers_fact>', '<prefers_both>',
                                   '<reserved>', '<talkative>']

        # self.strategy_codes = ['<no_seek_experience>', '<no_seek_knowledge>',
        #                        '<no_share_experience>', '<no_share_knowledge>',
        #                        '<seek_experience>', '<seek_knowledge>',
        #                        '<share_experience>', '<share_knowledge>']
        self.strategy_codes = ['<seek_experience>', '<seek_knowledge>',
                               '<share_experience>', '<share_knowledge>']
        self.max_pos_embeddings = 128 if "blender" in mname else 260
        if dataset is None:
            self.joint_training = True
        else:
            self.joint_training = False
        self.config, self.tokenizer, model = self.get_model(encoder_layers, loss_alpha)
        self.PAD_IDX = self.tokenizer.pad_token_id
        print("Base model and tokenizer initialized")
        # local_rank = 2
        if not use_gpu:
            self.device = "cpu"
        else:
            self.device = torch.device("cuda:{}".format(local_rank)) if torch.cuda.is_available() else "cpu"
        print("Device: ", self.device)
        model = model.to(self.device)
        self.model = model
        print("Model loaded")
        if num_workers > 2 and self.device != "cpu" and not skip_training:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                                       output_device=local_rank)
            print("DDP Model loaded")
        else:
            self.ddp_model = copy.deepcopy(self.model)
        print(f'The model has {self.count_parameters(self.ddp_model):,} trainable parameters')

        encoded_dict = pickle.load(open(training_file, "rb"))
        self.big5_ids = self.tokenizer.encode(self.big5_codes, add_special_tokens=False)
        self.corpus_based_ids = self.tokenizer.encode(self.corpus_based_codes, add_special_tokens=False)
        self.strategy_ids = self.tokenizer.encode(self.strategy_codes, add_special_tokens=False)

        self.train_dataloader = self.get_dataloader(encoded_dict["train.json"], training=True,
                                                    distributed=num_workers > 2)
        self.token_dict = {"pad_token_id": self.PAD_IDX}
        dict_tokens = ["share_experience", "share_knowledge", "seek_experience", "seek_knowledge", "high_agreeableness",
                       "high_openness", "high_conscientiousness", "high_extraversion", "high_neuroticism",
                       "low_agreeableness", "low_openness", "low_conscientiousness", "low_extraversion",
                       "low_neuroticism"]
        for i in dict_tokens:
            self.token_dict[i] = self.tokenizer.get_vocab()["<"+i+">"]

        if dataset is None or dataset == "topical_chat":
            self.valid_dataloader_1 = self.get_dataloader(encoded_dict["valid_freq.json"], training=False)
            self.valid_dataloader_2 = self.get_dataloader(encoded_dict["valid_rare.json"], training=False)
            self.test_dataloader_1 = self.get_dataloader(encoded_dict["test_freq.json"], training=False)
            self.test_dataloader_2 = self.get_dataloader(encoded_dict["test_rare.json"], training=False)
            self.test_dataloader_1_targets = self.get_labels_for_test(encoded_dict["test_freq.json"])
            self.test_dataloader_2_targets = self.get_labels_for_test(encoded_dict["test_rare.json"])
            self.test_gen_dataloader_1 = self.get_loader_for_generation(encoded_dict["test_freq.json"])
            self.test_gen_dataloader_2 = self.get_loader_for_generation(encoded_dict["test_rare.json"])

        if dataset is None or dataset == "wizard_of_wikipedia":
            self.valid_dataloader_1 = self.get_dataloader(encoded_dict["valid_random_split.json"], training=False)
            self.valid_dataloader_2 = self.get_dataloader(encoded_dict["valid_topic_split.json"], training=False)
            self.test_dataloader_1 = self.get_dataloader(encoded_dict["test_random_split.json"], training=False)
            self.test_dataloader_2 = self.get_dataloader(encoded_dict["test_topic_split.json"], training=False)
            self.test_dataloader_1_targets = self.get_labels_for_test(encoded_dict["test_random_split.json"])
            self.test_dataloader_2_targets = self.get_labels_for_test(encoded_dict["test_topic_split.json"])
            self.test_gen_dataloader_1 = self.get_loader_for_generation(encoded_dict["test_random_split.json"])
            self.test_gen_dataloader_2 = self.get_loader_for_generation(encoded_dict["test_topic_split.json"])

        print("Dataloaders loaded")

        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(self.ddp_model.parameters(), lr=self.lr)

    # def reformat_target(self, target_in, target_out):
    #     new_in, new_out = [], []
    #     comparator = []
    #     if "strategy" in self.ablation_remove:
    #         comparator += self.strategy_ids
    #     if "corpus_based" in self.ablation_remove:
    #         comparator += self.corpus_based_ids
    #     if "big5" in self.ablation_remove:
    #         comparator += self.big5_ids
    #     if self.ablation_remove == "all":
    #         comparator += self.big5_ids + self.corpus_based_ids + self.strategy_ids
    #
    #     comparator = set(comparator)
    #     for i in list(zip(target_in, target_out)):
    #         if i[0] not in comparator:
    #             new_in.append(i[0])
    #             new_out.append(i[1])
    #
    #     assert len(new_in) == len(new_out)
    #     return new_in, new_out
    def reformat_target(self, target_in, target_out):
        new_in, new_out = [], []
        comparator = []
        if "strategy" in self.ablation_remove:
            comparator += list(range(5, 9))#self.strategy_ids
        # if "corpus_based" in self.ablation_remove:
        #     comparator += self.corpus_based_ids
        if "big5" in self.ablation_remove:
            comparator += list(range(0, 5))#self.big5_ids
        if self.ablation_remove == "all":
            comparator += list(range(0, 9))#self.big5_ids + self.corpus_based_ids + self.strategy_ids

        comparator = set(comparator)
        for ix, i in enumerate(list(zip(target_in, target_out))):
            if ix not in comparator:
                new_in.append(i[0])
                new_out.append(i[1])

        assert len(new_in) == len(new_out)
        return new_in, new_out

    def reformat_input(self, input_ids, token_type_ids):
        new_in, new_out = [], []
        comparator = []
        if "strategy" in self.ablation_remove:
            comparator += self.strategy_ids
        if "corpus_based" in self.ablation_remove:
            comparator += self.corpus_based_ids
        if "big5" in self.ablation_remove:
            comparator += self.big5_ids
        if self.ablation_remove == "all":
            comparator += self.big5_ids + self.corpus_based_ids + self.strategy_ids

        comparator = set(comparator)

        for i in list(zip(input_ids, token_type_ids)):
            if i[0] not in comparator:
                new_in.append(i[0])
                new_out.append(i[1])

        while new_in[0] in [self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]:
            new_in = new_in[1:]
            new_out = new_out[1:]

        assert len(new_in) == len(new_out)
        return new_in, new_out

    def keep_example(self, lst):
        agent1, agent2 = [], []
        for ix, i in enumerate(lst):
            if self.tokenizer.decode(i) == "<agent_1>":
                agent1.append(ix)
            elif self.tokenizer.decode(i) == "<agent_2>":
                agent2.append(ix)

        if len(agent1) < 1:
            return False
        elif len(agent2) < 1:
            return True
        elif max(agent1) > max(agent2):
            return True
        else:
            return False

    def get_labels_for_test(self, data_dict):
        output_dict = {"tgt_big5_traits": [], "tgt_strategy": [], "tgt_fact_lbl": [], "tgt_opn_lbl": []}
        for (inpt_id, f_tgt, strategy_tgt, p_personality_tgt, e_personality_tgt) in zip(data_dict['input_ids'], data_dict['fact_label'],
                                                                   data_dict['tgt_strategy'],
                                                                   data_dict['tgt_pandora_personality'],
                                                                   data_dict['tgt_essays_personality']):
            if self.pandora_personality:
                personality_tgt = p_personality_tgt
            else:
                personality_tgt = e_personality_tgt

            if self.num_train is not None and len(output_dict["tgt_big5_traits"]) >= int(self.num_train / 10):
                break

            output_dict["tgt_big5_traits"].append(personality_tgt)
            output_dict["tgt_strategy"].append(strategy_tgt)
            output_dict["tgt_fact_lbl"].append(f_tgt)

        return output_dict

    def reformat_input_for_dataset(self, inpt_id, type_id):
        return inpt_id[1:], type_id[1:]

    def get_tgt_codes(self, tgt_id_in):
        idx = tgt_id_in.index(self.tokenizer.bos_token_id)
        return tgt_id_in[:idx+1]

    def data_process(self, data_dict, test=False):
        data = []

        for (inpt_id, type_id, tgt_id_in_p, tgt_id_out_p, tgt_id_in_e,
             tgt_id_out_e, f_inpt_id, f_tgt, p_strat_inpt_id, p_strat_type_id, e_strat_inpt_id,
             e_strat_type_id, strategy_tgt, p_personality_tgt, e_personality_tgt) in zip(data_dict['input_ids'],
                                                                                data_dict['token_type_ids'],
                                                                                data_dict['target_ids_in_pandora'],
                                                                                data_dict['target_ids_out_pandora'],
                                                                                data_dict['target_ids_in_essays'],
                                                                                data_dict['target_ids_out_essays'],
                                                                                data_dict['fact_input_ids'],
                                                                                data_dict['fact_label'],
                                                                                data_dict['pandora_strategy_input_ids'],
                                                                                data_dict['pandora_strategy_type_ids'],
                                                                                data_dict['essays_strategy_input_ids'],
                                                                                data_dict['essays_strategy_type_ids'],
                                                                                data_dict['tgt_strategy'],
                                                                                data_dict['tgt_pandora_personality'],
                                                                                data_dict['tgt_essays_personality']):

            if self.num_train is not None and len(data) >= self.num_train:
                break
            # example_dataset = self.tokenizer.decode(inpt_id[0])

            if self.pandora_personality:
                tgt_id_in, tgt_id_out = tgt_id_in_p, tgt_id_out_p
                strat_inpt_id, strat_type_id = p_strat_inpt_id, p_strat_type_id
                personality_tgt = p_personality_tgt
            else:
                tgt_id_in, tgt_id_out = tgt_id_in_e, tgt_id_out_e
                strat_inpt_id, strat_type_id = e_strat_inpt_id, e_strat_type_id
                personality_tgt = e_personality_tgt

            if self.ablation_remove != "none":
                tgt_id_in, tgt_id_out = self.reformat_target(tgt_id_in, tgt_id_out)
                if self.ablation_remove != "all":
                    strat_inpt_id, strat_type_id = self.reformat_input(strat_inpt_id, strat_type_id)

            if len(inpt_id) > self.max_pos_embeddings:
                inpt_id = inpt_id[-self.max_pos_embeddings:]
                type_id = type_id[-self.max_pos_embeddings:]

            inpt_id_t = torch.tensor(inpt_id).long()
            type_id_t = torch.tensor(type_id).long()

            if len(tgt_id_in) > self.max_pos_embeddings:
                tgt_id_in = tgt_id_in[-self.max_pos_embeddings:]
                tgt_id_out = tgt_id_out[-self.max_pos_embeddings:]

            tgt_in_id_t = torch.tensor(tgt_id_in).long()
            tgt_out_id_t = torch.tensor(tgt_id_out).long()

            if len(strat_inpt_id) > self.max_pos_embeddings:
                strat_inpt_id = strat_inpt_id[-self.max_pos_embeddings:]
                strat_type_id = strat_type_id[-self.max_pos_embeddings:]

            strat_inpt_id_t = torch.tensor(strat_inpt_id).long()
            strat_type_id_t = torch.tensor(strat_type_id).long()

            fact_tgt_t = torch.tensor(f_tgt).float()
            strategy_tgt_t = torch.tensor(strategy_tgt).float()
            personality_tgt_t = torch.tensor(personality_tgt).long()

            if test:
                tgt_id_in = self.get_tgt_codes(tgt_id_in)
                tgt_in_id_t = torch.tensor(tgt_id_in).long()
                data.append((inpt_id_t, type_id_t, tgt_in_id_t, tgt_out_id_t, strat_inpt_id_t,
                             strat_type_id_t, f_inpt_id, fact_tgt_t, strategy_tgt_t, personality_tgt_t))
            else:
                data.append(
                    (inpt_id_t, type_id_t, tgt_in_id_t, tgt_out_id_t, strat_inpt_id_t, strat_type_id_t, f_inpt_id,
                     fact_tgt_t, strategy_tgt_t, personality_tgt_t))

        return data

    def generate_batch(self, data_batch):
        in_ids, in_type_ids, tgt_in, labels, fact_in, \
        fact_labels, strat_inpt_ids, strat_type_ids = [], [], [], [], [], [], [], []

        strategy_lbl, personality_lbl = [], []
        for (inpt_id, type_id, tgt_id_in, tgt_id_out, strat_inpt_id, strat_type_id, f_inpt_id, f_tgt,
             strategy_tgt, personality_tgt) in data_batch:
            in_ids.append(inpt_id)
            in_type_ids.append(type_id)

            tgt_in.append(tgt_id_in)
            labels.append(tgt_id_out)

            fact_in.append(f_inpt_id)
            fact_labels.append(f_tgt)

            strat_inpt_ids.append(strat_inpt_id)
            strat_type_ids.append(strat_type_id)

            strategy_lbl.append(strategy_tgt)
            personality_lbl.append(personality_tgt)

        in_ids = pad_sequence(in_ids, padding_value=self.PAD_IDX, batch_first=True)
        in_type_ids = pad_sequence(in_type_ids, padding_value=0, batch_first=True)

        strat_inpt_ids = pad_sequence(strat_inpt_ids, padding_value=self.PAD_IDX, batch_first=True)
        strat_type_ids = pad_sequence(strat_type_ids, padding_value=0, batch_first=True)

        tgt_in = pad_sequence(tgt_in, padding_value=self.PAD_IDX, batch_first=True)
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)

        fact_in = self.pad_list_2_3dtensor(fact_in)
        fact_labels = torch.stack(fact_labels)

        strategy_lbl = torch.stack(strategy_lbl)
        personality_lbl = torch.stack(personality_lbl)

        return in_ids, in_type_ids, tgt_in, labels, fact_in, fact_labels, strat_inpt_ids, \
               strat_type_ids, strategy_lbl, personality_lbl

    def get_loader_for_generation(self, encoded_dict):
        data = self.data_process(encoded_dict, test=True)
        if self.num_train is not None:
            data = data[:int(self.num_train / 10)]
        return DataLoader(data, batch_size=self.batch_size, sampler=SequentialSampler(data),
                          collate_fn=self.generate_batch, num_workers=self.num_workers)

    def get_dataloader(self, encoded_dict, training=False, distributed=True):
        data = self.data_process(encoded_dict)
        if self.num_train is not None:
            if training:
                random.shuffle(data)
                data = data[:self.num_train]
            else:
                data = data[:int(self.num_train / 10)]
        print("data size: ", len(data))
        if training:
            if distributed:
                dataloader = DataLoader(data, batch_size=self.batch_size, sampler=DistributedSampler(data),
                                        collate_fn=self.generate_batch, num_workers=self.num_workers)
            else:
                dataloader = DataLoader(data, batch_size=self.batch_size, sampler=RandomSampler(data),
                                        collate_fn=self.generate_batch, num_workers=self.num_workers)
        else:
            dataloader = DataLoader(data, batch_size=self.batch_size, sampler=SequentialSampler(data),
                                    collate_fn=self.generate_batch, num_workers=self.num_workers)
        return dataloader

    def get_model(self, encoder_layers=2, loss_alpha=0.7):
        if "blenderbot" in self.mname:
            tokenizer = BlenderbotTokenizer.from_pretrained(self.mname)
            config = BlenderbotConfig.from_pretrained(self.mname)
        else:
            tokenizer = BartTokenizer.from_pretrained(self.mname)
            config = BartConfig.from_pretrained(self.mname)

        keys_to_add = ["<agent_1>", "<agent_2>"] + self.corpus_based_codes + self.big5_codes + self.strategy_codes
        special_tokens_dict = {'additional_special_tokens': keys_to_add}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        if "blenderbot" in self.mname and self.max_pos_embeddings > config.max_position_embeddings:
            config.max_position_embeddings = self.max_pos_embeddings
            config.encoder_layers, config.num_hidden_layers = encoder_layers, encoder_layers

        if "strategy" in self.ablation_remove or "all" in self.ablation_remove:
            config.predict_strategy = False
        else:
            config.predict_strategy = True
            config.n_strategies = 4

        if "big5" in self.ablation_remove or "all" in self.ablation_remove:
            config.predict_personality = False
        else:
            config.predict_personality = True
            config.n_personality = 5

        config.ablation_remove = self.ablation_remove
        config.loss_alpha, config.knowledge_labels = loss_alpha, [1, 3]
        config.trait_in_decoder = self.trait_in_decoder

        if "blenderbot" in self.mname:
            model = BlenderbotForConditionalGeneration.from_pretrained(self.mname, config=config)
        else:
            model = BartForConditionalGeneration.from_pretrained(self.mname, config=config)
        model.resize_token_embeddings(len(tokenizer))
        return config, tokenizer, model

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def set_random_seeds(self, random_seed=0):
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    def pad_list_2_3dtensor(self, lst):
        max_len = max([len(j) for i in lst for j in i])
        padded_lst = []
        for i in lst:
            padded_lst.append([j + [self.PAD_IDX] * (max_len - len(j)) for j in i])
        padded_tensor = torch.tensor(padded_lst)
        return padded_tensor

    def get_entailment_scores(self, logits, labels):
        argmax_preds = [i.strip() for i in self.tokenizer.batch_decode(torch.argmax(logits, -1),
                                                                       skip_special_tokens=True)]
        actual = self.tokenizer.batch_decode([[0 if j == -100 else j for j in i] for i in labels],
                                             skip_special_tokens=True)
        contradiction_payload = []
        for i in list(zip(argmax_preds, actual)):
            contradiction_payload.append({"premise": i[1], "hypothesis": i[0]})

        contradiction_wt = [i["probs"][1] for i in self.CONTRADICTION_PREDICTOR.predict(contradiction_payload)]
        return torch.tensor(contradiction_wt).to(logits.device)

    def train(self):

        self.ddp_model.train()

        ep_t_loss, lm_t_loss, fact_t_loss, lm_t_ce_loss = 0, 0, 0, 0
        strategy_t_loss, personality_t_loss = 0, 0
        batch_num = 0

        for ix, batch in enumerate(self.train_dataloader):
            if ix % 20 == 0:
                print("Training batch", ix)
            batch = tuple(t.to(self.device) for t in batch)
            in_ids, in_type_ids, tgt_in, labels, fact_in, fact_labels, strat_inpt_ids, \
            strat_type_ids, strategy_lbl, personality_lbl = batch

            # if self.num_workers > 2:
            #     print("in_ids:", in_ids)

            with autocast():
                if "fact" in self.ablation_remove:
                    fact_in, fact_labels = None, None
                if "strategy" in self.ablation_remove or self.ablation_remove == "all":
                    strategy_lbl = None
                if "big5" in self.ablation_remove or self.ablation_remove == "all":
                    personality_lbl = None
                if self.ablation_remove == "all":
                    strat_inpt_ids, strat_type_ids = None, None

                output = self.ddp_model(input_ids=in_ids, token_type_ids=in_type_ids, decoder_input_ids=tgt_in,
                                        fact_input_ids=fact_in, labels=labels, fact_labels=fact_labels,
                                        strategy_labels=strategy_lbl, personality_labels=personality_lbl,
                                        strategy_input_ids=strat_inpt_ids, strategy_type_ids=strat_type_ids)

                aux_sum = sum([1 if output.get("fact_selection_loss", None) is not None else 0,
                               1 if output.get("strategy_pred_loss", None) is not None else 0,
                               1 if output.get("personality_pred_loss", None) is not None else 0])
                if aux_sum == 0:
                    lmda = 1.0
                    aux = 0.0
                else:
                    lmda = 0.7
                    aux = (1.0-lmda) / aux_sum

                if self.loss_alpha is not None:
                    if self.contradiction_penalty:
                        contradiction_wt = self.get_entailment_scores(output["logits"].detach(), labels.detach())
                        loss = (output["loss"] + contradiction_wt).mean()
                    else:
                        loss = output["loss"].mean()
                else:
                    loss = output["loss"] * lmda
                lm_t_loss += output["loss"].mean().detach().item()
                if output.get("fact_selection_loss", None) is not None:
                    loss += output["fact_selection_loss"] * aux
                if output.get("strategy_pred_loss", None) is not None:
                    loss += output["strategy_pred_loss"] * aux
                if output.get("personality_pred_loss", None) is not None:
                    loss += output["personality_pred_loss"] * aux

                loss = loss / self.accum_iter

            self.scaler.scale(loss).backward()

            if (ix + 1) % self.accum_iter == 0 or ix + 1 == len(self.train_dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            ep_t_loss += loss.item()
            if fact_labels is not None:
                fact_t_loss += output["fact_selection_loss"].item()
            else:
                fact_t_loss = float("inf")

            if "strategy" not in self.ablation_remove and "all" not in self.ablation_remove:
                strategy_t_loss += output["strategy_pred_loss"].item()
            else:
                strategy_t_loss = float("inf")

            if "big5" not in self.ablation_remove and "all" not in self.ablation_remove and \
                    "trait" not in self.ablation_remove:
                personality_t_loss += output["personality_pred_loss"].item()
            else:
                personality_t_loss = float("inf")
            lm_t_ce_loss += output["ce_loss"]  # .item()
            batch_num += 1

        return ep_t_loss / batch_num, lm_t_loss / batch_num, fact_t_loss / batch_num, \
               strategy_t_loss / batch_num, personality_t_loss / batch_num, lm_t_ce_loss / batch_num

    def evaluate(self, model, dataloader):

        model.eval()

        ep_t_loss, lm_t_loss, fact_t_loss, lm_t_ce_loss = 0, 0, 0, 0
        strategy_t_loss, personality_t_loss = 0, 0
        batch_num = 0
        actual, pred = [], []
        fact_act, fact_pred = [], []
        strategy_act, strategy_pred, personality_act, personality_pred = [], [], [], []

        for ix, batch in enumerate(dataloader):
            if ix % 20 == 0:
                print("Validation batch", ix)
            batch = tuple(t.to(self.device) for t in batch)

            in_ids, in_type_ids, tgt_in, labels, fact_in, fact_labels, strat_inpt_ids, \
            strat_type_ids, strategy_lbl, personality_lbl = batch
            # if self.num_workers > 2:
            #     print("in_ids:", in_ids)

            with autocast():
                with torch.no_grad():
                    if "fact" in self.ablation_remove:
                        fact_in, fact_labels = None, None
                    if "strategy" in self.ablation_remove or self.ablation_remove == "all":
                        strategy_lbl = None
                    if "big5" in self.ablation_remove or self.ablation_remove == "all":
                        personality_lbl = None
                    if self.ablation_remove == "all":
                        strat_inpt_ids, strat_type_ids = None, None

                    output = model(input_ids=in_ids, token_type_ids=in_type_ids, decoder_input_ids=tgt_in,
                                   fact_input_ids=fact_in, labels=labels, fact_labels=fact_labels,
                                   strategy_labels=strategy_lbl, personality_labels=personality_lbl,
                                   strategy_input_ids=strat_inpt_ids, strategy_type_ids=strat_type_ids)
                aux_sum = sum([1 if output.get("fact_selection_loss", None) is not None else 0,
                               1 if output.get("strategy_pred_loss", None) is not None else 0,
                               1 if output.get("personality_pred_loss", None) is not None else 0])
                if aux_sum == 0:
                    lmda = 1.0
                    aux = 0.0
                else:
                    lmda = 0.7
                    aux = (1.0 - lmda) / aux_sum

                if self.loss_alpha is not None:
                    if self.contradiction_penalty:
                        contradiction_wt = self.get_entailment_scores(output["logits"].detach(), labels.detach())
                        loss = (output["loss"] + contradiction_wt).mean()
                    else:
                        loss = output["loss"].mean()
                else:
                    loss = output["loss"] * lmda

                lm_t_loss += output["loss"].mean().detach().item()
                if output.get("fact_selection_loss", None) is not None:
                    loss += output["fact_selection_loss"] * aux
                if output.get("strategy_pred_loss", None) is not None:
                    loss += output["strategy_pred_loss"] * aux
                if output.get("personality_pred_loss", None) is not None:
                    loss += output["personality_pred_loss"] * aux

            ep_t_loss += loss.item()
            if fact_labels is not None:
                fact_t_loss += output["fact_selection_loss"].item()
            else:
                fact_t_loss = float("inf")

            if "strategy" not in self.ablation_remove and "all" not in self.ablation_remove:
                strategy_t_loss += output["strategy_pred_loss"].item()
            else:
                strategy_t_loss = float("inf")

            if "big5" not in self.ablation_remove and "all" not in self.ablation_remove and \
                    "trait" not in self.ablation_remove:
                personality_t_loss += output["personality_pred_loss"].item()
            else:
                personality_t_loss = float("inf")
            lm_t_ce_loss += output["ce_loss"]  # .item()
            batch_num += 1

            actual.extend(labels.tolist())
            pred.extend(torch.argmax(output["logits"], -1).tolist())

            if "fact" not in self.ablation_remove:
                fact_act.extend(fact_labels.view(-1).tolist())
                fact_pred.extend(torch.round(torch.sigmoid(output["fact_preds"].view(-1))).long().tolist())

            if "strategy" not in self.ablation_remove and "all" not in self.ablation_remove:
                strategy_act.extend(strategy_lbl.tolist())
                strategy_pred.extend(torch.round(torch.sigmoid(output["strategy_prediction"])).tolist())

            if "big5" not in self.ablation_remove and "all" not in self.ablation_remove and \
                    "trait" not in self.ablation_remove:
                personality_act.extend(personality_lbl.tolist())
                personality_pred.extend(torch.argmax(output["personality_prediction"], 1).tolist())
                # personality_pred.extend(torch.round(torch.sigmoid(output["personality_prediction"])).tolist())

        actual = self.tokenizer.batch_decode([[0 if j == -100 else j for j in i] for i in actual],
                                             skip_special_tokens=True)
        actual = [i.strip() for i in actual]

        pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
        pred = [i.strip() for i in pred]
        pred = [i.split('</s>')[0].strip() for i in pred]

        tmp_stats = []
        if "fact" not in self.ablation_remove:
            print("\n FACT Classification Report:\n", classification_report(fact_act, fact_pred))
            print("\n FACT Confusion Matrix:\n", confusion_matrix(fact_act, fact_pred))
            tmp_stats.append(f1_score(fact_act, fact_pred))

        if "strategy" not in self.ablation_remove and "all" not in self.ablation_remove:
            strategy_act = np.asarray(strategy_act)
            strategy_pred = np.asarray(strategy_pred)
            print("\n share_experience Classification Report:\n",
                  classification_report(strategy_act[:, 0], strategy_pred[:, 0]))
            print("\n share_knowledge Classification Report:\n",
                  classification_report(strategy_act[:, 1], strategy_pred[:, 1]))
            print("\n seek_experience Classification Report:\n",
                  classification_report(strategy_act[:, 2], strategy_pred[:, 2]))
            print("\n seek_knowledge Classification Report:\n",
                  classification_report(strategy_act[:, 3], strategy_pred[:, 3]))

            print("\n share_experience Confusion Matrix:\n", confusion_matrix(strategy_act[:, 0], strategy_pred[:, 0]))
            print("\n share_knowledge Confusion Matrix:\n", confusion_matrix(strategy_act[:, 1], strategy_pred[:, 1]))
            print("\n seek_experience Confusion Matrix:\n", confusion_matrix(strategy_act[:, 2], strategy_pred[:, 2]))
            print("\n seek_knowledge Confusion Matrix:\n", confusion_matrix(strategy_act[:, 3], strategy_pred[:, 3]))

            tmp_stats.append(f1_score(strategy_act[:, 0], strategy_pred[:, 0]))
            tmp_stats.append(f1_score(strategy_act[:, 1], strategy_pred[:, 1]))
            tmp_stats.append(f1_score(strategy_act[:, 2], strategy_pred[:, 2]))
            tmp_stats.append(f1_score(strategy_act[:, 3], strategy_pred[:, 3]))

        if "big5" not in self.ablation_remove and "all" not in self.ablation_remove and \
                "trait" not in self.ablation_remove:
            personality_act = np.asarray(personality_act)
            personality_pred = np.asarray(personality_pred)
            print("\n agreeableness Classification Report:\n",
                  classification_report(personality_act[:, 0], personality_pred[:, 0]))
            print("\n openness Classification Report:\n",
                  classification_report(personality_act[:, 1], personality_pred[:, 1]))
            print("\n conscientiousness Classification Report:\n",
                  classification_report(personality_act[:, 2], personality_pred[:, 2]))
            print("\n extraversion Classification Report:\n",
                  classification_report(personality_act[:, 3], personality_pred[:, 3]))
            print("\n neuroticism Classification Report:\n",
                  classification_report(personality_act[:, 4], personality_pred[:, 4]))

            print("\n agreeableness Confusion Matrix:\n",
                  confusion_matrix(personality_act[:, 0], personality_pred[:, 0]))
            print("\n openness Confusion Matrix:\n", confusion_matrix(personality_act[:, 1], personality_pred[:, 1]))
            print("\n conscientiousness Confusion Matrix:\n",
                  confusion_matrix(personality_act[:, 2], personality_pred[:, 2]))
            print("\n extraversion Confusion Matrix:\n",
                  confusion_matrix(personality_act[:, 3], personality_pred[:, 3]))
            print("\n neuroticism Confusion Matrix:\n", confusion_matrix(personality_act[:, 4], personality_pred[:, 4]))

            tmp_stats.append(f1_score(personality_act[:, 0], personality_pred[:, 0], average="macro"))
            tmp_stats.append(f1_score(personality_act[:, 1], personality_pred[:, 1], average="macro"))
            tmp_stats.append(f1_score(personality_act[:, 2], personality_pred[:, 2], average="macro"))
            tmp_stats.append(f1_score(personality_act[:, 3], personality_pred[:, 3], average="macro"))
            tmp_stats.append(f1_score(personality_act[:, 4], personality_pred[:, 4], average="macro"))

        assert len(pred) == len(actual)

        return ep_t_loss / batch_num, lm_t_loss / batch_num, fact_t_loss / batch_num, \
               strategy_t_loss / batch_num, personality_t_loss / batch_num, lm_t_ce_loss / batch_num, actual, pred, tmp_stats

    def run_evaluation(self, model, dataloader, dataset, key):
        val_l, val_lm_l, val_f_l, val_strat_l, val_persona_l, \
        val_ce_l, val_cor, val_hyp, tmp_stats = self.evaluate(model, dataloader)

        print("EVAL Metrics for dataset: ", dataset, key)
        print(
            f'\tVal Total Loss: {val_l:.3f} | Val LM Loss: {val_lm_l:.3f} | Val CE Loss: {val_ce_l:.3f} | '
            f'Val Fact Loss: {val_f_l:.3f} | '
            f'Val Perplexity LM: {math.exp(val_lm_l):.3f} | Val Perplexity CE: {math.exp(val_ce_l):.3f}')
        print(
            f'\tVal Strategy Loss: {val_strat_l:.3f} | Val Personality Loss: {val_persona_l:.3f} ')

        print("-" * 10)
        print("\n")
        tmp = [val_l, val_lm_l, val_f_l, val_strat_l, val_persona_l, val_ce_l,  # val_cor, val_hyp,
               math.exp(val_lm_l), math.exp(val_ce_l)] + tmp_stats
        return val_lm_l, tmp

    # def generate_response(self, dataloader, rouge, key):
    #     self.model.eval()
    #     cor, hyp = [], []
    #     ctx, facts = [], []
    #     avoid_toks = self.corpus_based_codes + self.big5_codes + self.strategy_codes + \
    #                  [self.tokenizer.pad_token, self.tokenizer.bos_token, self.tokenizer.eos_token]
    #     avoid_keys = [self.tokenizer.get_vocab()[i] for i in avoid_toks]
    #     # print("Keys to avoid:", avoid_keys)
    #
    #     for ix, batch in tqdm(enumerate(dataloader)):
    #         batch = tuple(t.to(self.device) for t in batch)
    #         in_ids, in_type_ids, tgt_in, labels, fact_in, fact_labels, strat_inpt_ids, \
    #         strat_type_ids, strategy_lbl, personality_lbl = batch
    #
    #         with autocast():
    #             with torch.no_grad():
    #                 n = 9
    #                 if "big5" in self.config.ablation_remove or self.config.ablation_remove == "trait":
    #                     n -= 5
    #                 if "strategy" in self.config.ablation_remove:
    #                     n -= 4
    #                 if "all" in self.config.ablation_remove:
    #                     n -= 9
    #                 n = max(n, 0)
    #                 if n == 0:
    #                     res = self.model.generate(input_ids=in_ids, token_type_ids=in_type_ids,
    #                                               fact_input_ids=fact_in, fact_label=fact_labels,
    #                                               do_sample=False, early_stopping=True,
    #                                               num_beams=5,
    #                                               no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3,
    #                                               num_return_sequences=1,
    #                                               strategy_prediction=False)
    #                 else:
    #                     if self.config.ablation_remove == "all":
    #                         res = self.model.generate(input_ids=in_ids, token_type_ids=in_type_ids,
    #                                                   do_sample=False, early_stopping=True,
    #                                                   num_beams=5,
    #                                                   no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3,
    #                                                   num_return_sequences=1,
    #                                                   decoder_input_ids=tgt_in[:, :n],
    #                                                   strategy_prediction=False)
    #                     else:
    #                         res = self.model.generate(input_ids=in_ids, token_type_ids=in_type_ids,
    #                                                   fact_input_ids=fact_in, fact_label=fact_labels,
    #                                                   strategy_input_ids=strat_inpt_ids,
    #                                                   strategy_type_ids=strat_type_ids,
    #                                                   do_sample=False, early_stopping=True,
    #                                                   num_beams=5,
    #                                                   no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3,
    #                                                   num_return_sequences=1,
    #                                                   decoder_input_ids=tgt_in[:, :n],
    #                                                   strategy_prediction=False)
    #                 act = self.tokenizer.batch_decode([[j for j in i if j != -100] for i in labels],
    #                                                   skip_special_tokens=True)
    #                 res = self.tokenizer.batch_decode(res, skip_special_tokens=True)
    #
    #                 hyp.extend(res)
    #                 cor.extend(act)
    #                 ctx.extend(
    #                     self.tokenizer.batch_decode([[j for j in i if j not in avoid_keys] for i in in_ids.tolist()]))
    #                 facts.extend([self.tokenizer.batch_decode(i, skip_special_tokens=True) for i in fact_in])
    #
    #     assert len(hyp) == len(cor)
    #     references = [[i.split()] for i in cor]
    #     candidates = [i.split() for i in hyp]
    #     bleu_score = corpus_bleu(references, candidates)
    #     bleu_1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    #     bleu_2 = corpus_bleu(references, candidates, weights=(0, 1, 0, 0))
    #     bleu_3 = corpus_bleu(references, candidates, weights=(0, 0, 1, 0))
    #     bleu_4 = corpus_bleu(references, candidates, weights=(0, 0, 0, 1))
    #
    #     rouge_score = rouge.compute(predictions=hyp, references=cor)
    #     rouge1, rouge2, rougeL, rougeLsum = rouge_score["rouge1"].mid.fmeasure, rouge_score["rouge2"].mid.fmeasure, \
    #                                         rouge_score["rougeL"].mid.fmeasure, rouge_score["rougeLsum"].mid.fmeasure
    #
    #     test_scores = {"bleu": bleu_score, "bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
    #                    "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL, "rougeLsum": rougeLsum}
    #     print(self.dataset, key, " TEST scores: ", test_scores)
    #     test_scores["cor"], test_scores["hyp"] = cor, hyp
    #     test_scores["context"], test_scores["facts"] = ctx, facts
    #     return test_scores

    def generate_response(self, dataloader, rouge, key, golden=True):
        self.model.eval()
        cor, hyp = [], []
        ctx, facts = [], []
        avoid_toks = self.corpus_based_codes + self.big5_codes + self.strategy_codes + \
                     [self.tokenizer.pad_token, self.tokenizer.bos_token, self.tokenizer.eos_token]
        avoid_keys = [self.tokenizer.get_vocab()[i] for i in avoid_toks]

        for ix, batch in tqdm(enumerate(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            in_ids, in_type_ids, tgt_in, labels, fact_in, fact_labels, strat_inpt_ids, \
            strat_type_ids, strategy_lbl, personality_lbl = batch

            with autocast():
                with torch.no_grad():
                    if golden:
                        if self.ablation_remove == "all":
                            res = self.model.generate(input_ids=in_ids, token_type_ids=in_type_ids,
                                                      fact_input_ids=fact_in, fact_label=fact_labels,
                                                      do_sample=False, early_stopping=True,
                                                      num_beams=5,
                                                      no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3,
                                                      num_return_sequences=1,
                                                      decoder_input_ids=tgt_in,
                                                      strategy_prediction=False,
                                                      max_length=max(120, self.config.max_length))
                        else:
                            res = self.model.generate(input_ids=in_ids, token_type_ids=in_type_ids,
                                                      fact_input_ids=fact_in, fact_label=fact_labels,
                                                      strategy_input_ids=strat_inpt_ids,
                                                      strategy_type_ids=strat_type_ids,
                                                      do_sample=False, early_stopping=True,
                                                      num_beams=5,
                                                      no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3,
                                                      num_return_sequences=1,
                                                      decoder_input_ids=tgt_in,
                                                      strategy_prediction=False,
                                                      max_length=max(120, self.config.max_length))
                    else:
                        if self.ablation_remove == "all":
                            res = self.model.generate(input_ids=in_ids, token_type_ids=in_type_ids,
                                                      fact_input_ids=fact_in, fact_label=fact_labels,
                                                      do_sample=False, early_stopping=True,
                                                      num_beams=5,
                                                      no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3,
                                                      num_return_sequences=1,
                                                      strategy_prediction=True,
                                                      max_length=max(120, self.config.max_length),
                                                      token_dict=self.token_dict)
                        else:
                            res = self.model.generate(input_ids=in_ids, token_type_ids=in_type_ids,
                                                      fact_input_ids=fact_in, fact_label=fact_labels,
                                                      strategy_input_ids=strat_inpt_ids,
                                                      strategy_type_ids=strat_type_ids,
                                                      do_sample=False, early_stopping=True,
                                                      num_beams=5,
                                                      no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3,
                                                      num_return_sequences=1,
                                                      strategy_prediction=True,
                                                      max_length=max(120, self.config.max_length),
                                                      token_dict=self.token_dict)
                    act = self.tokenizer.batch_decode([[j for j in i if j != -100] for i in labels],
                                                      skip_special_tokens=True)
                    res = self.tokenizer.batch_decode(res, skip_special_tokens=True)

                    hyp.extend(res)
                    cor.extend(act)
                    ctx.extend(
                        self.tokenizer.batch_decode([[j for j in i if j not in avoid_keys] for i in in_ids.tolist()]))
                    facts.extend([self.tokenizer.batch_decode(i, skip_special_tokens=True) for i in fact_in])

        assert len(hyp) == len(cor)
        references = [[i.split()] for i in cor]
        candidates = [i.split() for i in hyp]
        bleu_score = corpus_bleu(references, candidates)
        bleu_1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, candidates, weights=(0, 1, 0, 0))
        bleu_3 = corpus_bleu(references, candidates, weights=(0, 0, 1, 0))
        bleu_4 = corpus_bleu(references, candidates, weights=(0, 0, 0, 1))

        rouge_score = rouge.compute(predictions=hyp, references=cor)
        rouge1, rouge2, rougeL, rougeLsum = rouge_score["rouge1"].mid.fmeasure, rouge_score["rouge2"].mid.fmeasure, \
                                            rouge_score["rougeL"].mid.fmeasure, rouge_score["rougeLsum"].mid.fmeasure

        test_scores = {"bleu": bleu_score, "bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
                       "rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL, "rougeLsum": rougeLsum}
        tmp_lst = [key, bleu_score, bleu_4, rougeL]#, bs_50, bs_90, bls_50, bls_90]
        # print(self.dataset, key, " TEST scores: ", test_scores)
        test_scores["cor"], test_scores["hyp"] = cor, hyp
        test_scores["context"], test_scores["facts"] = ctx, facts
        return test_scores, tmp_lst

    def intent2strategy(self, intent_list):
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

    def get_strategy(self, text_list):
        st = time.time()
        examples = []
        for ix, text in enumerate(text_list):
            sents = sent_tokenize(text)
            examples.extend(list(zip(sents, [ix * len(sents)], [text] * len(sents))))
        text_df_tokenized = pd.DataFrame(examples, columns=["sentence", "idx", "text"])

        intent_pred = self.intent_predictor.predict(list(text_df_tokenized["sentence"]))
        text_df_tokenized["intent"] = intent_pred
        text_df_tokenized.groupby(["idx", "text"]).agg({"sentence": list, "intent": list}).reset_index()
        text_df_tokenized["strategy"] = text_df_tokenized["intent"].apply(self.intent2strategy)
        print("Intent Predictor Took ", str(time.time() - st), "ms to execute")
        return list(text_df_tokenized["strategy"])

    def compute_strategy_metrics(self, actual_strategy_labels, generated_responses):
        output = {"share_experience": {}, "share_knowledge": {}, "seek_experience": {}, "seek_knowledge": {}}
        predicted_strategy_labels = self.get_strategy(generated_responses)
        actual_strategy_labels = np.asarray(actual_strategy_labels)
        predicted_strategy_labels = np.asarray(predicted_strategy_labels)

        print("\nStrategy Array Shapes: ", actual_strategy_labels.shape, predicted_strategy_labels.shape)
        print(actual_strategy_labels[:3, :], predicted_strategy_labels[:3, :], "\n")

        output["share_experience"]["f1"] = f1_score(actual_strategy_labels[:, 0], predicted_strategy_labels[:, 0])
        output["share_experience"]["acc"] = accuracy_score(actual_strategy_labels[:, 0],
                                                           predicted_strategy_labels[:, 0])

        output["share_knowledge"]["f1"] = f1_score(actual_strategy_labels[:, 1], predicted_strategy_labels[:, 1])
        output["share_knowledge"]["acc"] = accuracy_score(actual_strategy_labels[:, 1], predicted_strategy_labels[:, 1])

        output["seek_experience"]["f1"] = f1_score(actual_strategy_labels[:, 2], predicted_strategy_labels[:, 2])
        output["seek_experience"]["acc"] = accuracy_score(actual_strategy_labels[:, 2], predicted_strategy_labels[:, 2])

        output["seek_knowledge"]["f1"] = f1_score(actual_strategy_labels[:, 3], predicted_strategy_labels[:, 3])
        output["seek_knowledge"]["acc"] = accuracy_score(actual_strategy_labels[:, 3], predicted_strategy_labels[:, 3])

        return output

    def compute_big5_metrics(self, actual_big5_labels, generated_responses):

        output = {"agreeableness": {}, "openness": {}, "conscientiousness": {}, "extraversion": {}, "neuroticism": {}}
        predicted_big5_traits, predicted_big5_raw_score = self.big5_predictor.predict(generated_responses, self.dataset)

        predicted_big5_traits = np.asarray(predicted_big5_traits)
        actual_big5_labels = np.asarray(actual_big5_labels)
        predicted_big5_raw_score = np.asarray(predicted_big5_raw_score)

        print("\nBig5 Array Shapes: ", actual_big5_labels.shape, predicted_big5_traits.shape)
        print(actual_big5_labels[:3, :], predicted_big5_traits[:3, :], "\n")

        output["agreeableness"]["f1"] = f1_score(actual_big5_labels[:, 0], predicted_big5_traits[:, 0])
        output["agreeableness"]["acc"] = accuracy_score(actual_big5_labels[:, 0], predicted_big5_traits[:, 0])
        output["agreeableness"]["mean"] = predicted_big5_raw_score[:, 0].mean()

        output["openness"]["f1"] = f1_score(actual_big5_labels[:, 1], predicted_big5_traits[:, 1])
        output["openness"]["acc"] = accuracy_score(actual_big5_labels[:, 1], predicted_big5_traits[:, 1])
        output["openness"]["mean"] = predicted_big5_raw_score[:, 1].mean()

        output["conscientiousness"]["f1"] = f1_score(actual_big5_labels[:, 2], predicted_big5_traits[:, 2])
        output["conscientiousness"]["acc"] = accuracy_score(actual_big5_labels[:, 2], predicted_big5_traits[:, 2])
        output["conscientiousness"]["mean"] = predicted_big5_raw_score[:, 2].mean()

        output["extraversion"]["f1"] = f1_score(actual_big5_labels[:, 3], predicted_big5_traits[:, 3])
        output["extraversion"]["acc"] = accuracy_score(actual_big5_labels[:, 3], predicted_big5_traits[:, 3])
        output["extraversion"]["mean"] = predicted_big5_raw_score[:, 3].mean()

        output["neuroticism"]["f1"] = f1_score(actual_big5_labels[:, 4], predicted_big5_traits[:, 4])
        output["neuroticism"]["acc"] = accuracy_score(actual_big5_labels[:, 4], predicted_big5_traits[:, 4])
        output["neuroticism"]["mean"] = predicted_big5_raw_score[:, 4].mean()

        return output
