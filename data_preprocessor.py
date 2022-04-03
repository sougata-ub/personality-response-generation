import copy
from tqdm import tqdm
import torch
import argparse
from trainer import Trainer
import os
import pickle
import json
import os.path

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['LOCAL_RANK'] = "2, 3"


class DataProcesor(Trainer):
    def __init__(self, mname, ablation_remove, dataset, encoder_layers, training_file, trait_in_decoder,
                 local_rank, batch_size, num_workers, num_train, uni_agent, lr, N_EPOCHS, accum_iter,
                 contradiction_penalty, intent_model_path, big5_model_path, loss_alpha, pandora_personality,
                 skip_training, use_gpu):
        super().__init__(mname, ablation_remove, dataset, encoder_layers, training_file, trait_in_decoder,
                         local_rank, batch_size, num_workers, num_train, uni_agent, lr, N_EPOCHS, accum_iter,
                         contradiction_penalty, intent_model_path, big5_model_path, loss_alpha, pandora_personality,
                         skip_training, use_gpu)

        self.common_dict = {"input_ids": [], "token_type_ids": [], "target_ids_in_pandora": [],
                            "target_ids_out_pandora": [], "target_ids_in_essays": [], "target_ids_out_essays": [],
                            "fact_input_ids": [], "fact_label":[], "pandora_strategy_input_ids":[],
                            "pandora_strategy_type_ids":[], "essays_strategy_input_ids":[],
                            "essays_strategy_type_ids":[], "tgt_strategy":[], "tgt_pandora_personality":[],
                            "tgt_essays_personality":[]
                            }

    def annotate_strategy_v2(self, txt):
        tmp = [0.0] * 4
        if "<share_experience>" in txt:
            tmp[0] = 1.0
        if "<share_knowledge>" in txt:
            tmp[1] = 1.0
        if "<seek_experience>" in txt:
            tmp[2] = 1.0
        if "<seek_knowledge>" in txt:
            tmp[3] = 1.0
        return tmp

    def annotate_turn_trait_v2(self, txt):
        tmp = [0] * 5
        if "<high_agreeableness>" in txt:
            tmp[0] = 1
        elif "<low_agreeableness>" in txt:
            tmp[0] = 2

        if "<high_openness>" in txt:
            tmp[1] = 1
        elif "<low_openness>" in txt:
            tmp[1] = 2

        if "<high_conscientiousness>" in txt:
            tmp[2] = 1
        elif "<low_conscientiousness>" in txt:
            tmp[2] = 2

        if "<high_extraversion>" in txt:
            tmp[3] = 1
        elif "<low_extraversion>" in txt:
            tmp[3] = 2

        if "<high_neuroticism>" in txt:
            tmp[4] = 1
        elif "<low_neuroticism>" in txt:
            tmp[4] = 2
        return tmp

    def adjust_encoding(self, enc, nested=True):
        if not nested:
            if enc["input_ids"][0] == self.tokenizer.bos_token_id:
                input_ids = enc["input_ids"][1:]
                attention_mask = enc["attention_mask"][1:]
            else:
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
        else:
            input_ids, attention_mask = [], []
            for ix, i in enumerate(enc["input_ids"]):
                if i[0] == self.tokenizer.bos_token_id:
                    input_ids.append(i[1:])
                    attention_mask.append(enc["attention_mask"][ix][1:])
                else:
                    input_ids.append(i)
                    attention_mask.append(enc["attention_mask"][ix])
        return input_ids, attention_mask

    def make_conv_ctx(self, x, ctx_strategy, pandora_persona, essays_persona, corpus_based, threshold=128):
        agents = [i.split()[0] for i in x]

        if len(ctx_strategy) > len(pandora_persona):
            pandora_persona = [""] * (len(ctx_strategy) - len(pandora_persona)) + pandora_persona
        elif len(ctx_strategy) < len(pandora_persona):
            pandora_persona.reverse()
            pandora_persona = pandora_persona[:len(ctx_strategy)]
            pandora_persona.reverse()

        if len(ctx_strategy) > len(essays_persona):
            essays_persona = [""] * (len(ctx_strategy) - len(essays_persona)) + essays_persona
        elif len(ctx_strategy) < len(essays_persona):
            essays_persona.reverse()
            essays_persona = essays_persona[:len(ctx_strategy)]
            essays_persona.reverse()

        assert len(agents) == len(ctx_strategy)
        ctx_strategy_pandora = [agents[ix] + pandora_persona[ix] + i for ix, i in enumerate(ctx_strategy)]
        ctx_strategy_essays = [agents[ix] + essays_persona[ix] + i for ix, i in enumerate(ctx_strategy)]

        enc = self.tokenizer.batch_encode_plus(x, truncation=True, max_length=threshold)
        enc["input_ids"], enc["attention_mask"] = self.adjust_encoding(enc)

        tmp_input_ids = []
        counter = 0

        for i in reversed(enc["input_ids"]):
            if len(i) + counter + 1> threshold:
                break
            else:
                tmp_input_ids.append(i)
                counter += len(i)+1

        if len(tmp_input_ids) == 0:
            tmp_input_ids.append(i)
            counter += len(i)+1

        tmp_input_ids = [[self.tokenizer.bos_token_id]+i if ix != len(tmp_input_ids) -1 else i for ix, i in enumerate(tmp_input_ids)]
        input_ids = [j for i in reversed(tmp_input_ids) for j in i]

        enc_ctx_strategy_pandora = self.tokenizer.batch_encode_plus(ctx_strategy_pandora[-len(tmp_input_ids):])
        enc_ctx_strategy_essays = self.tokenizer.batch_encode_plus(ctx_strategy_essays[-len(tmp_input_ids):])
        enc_corpus_based = self.tokenizer.encode_plus(corpus_based)

        enc_ctx_strategy_pandora["input_ids"], enc_ctx_strategy_pandora["attention_mask"] = self.adjust_encoding(enc_ctx_strategy_pandora)
        enc_ctx_strategy_essays["input_ids"], enc_ctx_strategy_essays["attention_mask"] = self.adjust_encoding(enc_ctx_strategy_essays)
        enc_corpus_based["input_ids"], enc_corpus_based["attention_mask"] = self.adjust_encoding(enc_corpus_based, nested=False)

        token_type_ctx_strategy_pandora_ids = [[0]*(len(i) + 1) if ix % 2 == 0 else [1]*(len(i) + 1) for ix, i in enumerate(reversed(enc_ctx_strategy_pandora["input_ids"]))]
        token_type_ctx_strategy_pandora_ids = [1] * len(enc_corpus_based["input_ids"]) + [j for i in reversed(token_type_ctx_strategy_pandora_ids) for j in i]

        token_type_ctx_strategy_essays_ids = [[0]*(len(i) + 1) if ix % 2 == 0 else [1]*(len(i) + 1) for ix, i in enumerate(reversed(enc_ctx_strategy_essays["input_ids"]))]
        token_type_ctx_strategy_essays_ids = [1] * len(enc_corpus_based["input_ids"]) + [j for i in reversed(token_type_ctx_strategy_essays_ids) for j in i]

        enc_ctx_strategy_pandora = enc_corpus_based["input_ids"] + [j for i in enc_ctx_strategy_pandora["input_ids"] for j in [self.tokenizer.bos_token_id]+i]
        enc_ctx_strategy_essays = enc_corpus_based["input_ids"] + [j for i in enc_ctx_strategy_essays["input_ids"] for j in [self.tokenizer.bos_token_id]+i]

        token_type_ids = [[0]*len(i) if ix % 2 == 0 else [1]*len(i) for ix, i in enumerate(tmp_input_ids)]
        token_type_ids = [j for i in reversed(token_type_ids) for j in i]
        token_type_ids = token_type_ids

        assert len(input_ids) == len(token_type_ids)
        assert len(token_type_ctx_strategy_pandora_ids) == len(enc_ctx_strategy_pandora)
        assert len(enc_ctx_strategy_essays) == len(token_type_ctx_strategy_essays_ids)
        return input_ids, token_type_ids, enc_ctx_strategy_pandora, token_type_ctx_strategy_pandora_ids, enc_ctx_strategy_essays, token_type_ctx_strategy_essays_ids

    def format_inputs(self, formatted_examples):
        encoded_dict = {"inference.json": copy.deepcopy(self.common_dict)}
        for k, v in formatted_examples.items():
            for ix, i in tqdm(enumerate(v["x"])):
                input_ids, token_type_ids, \
                pan_ids, pan_type, essay_ids, essay_type = self.make_conv_ctx(i, v["ctx_strategy"][ix],
                                                                              v["ctx_trait_pandora"][ix],
                                                                              v["ctx_trait_essays"][ix],
                                                                              v["corpus_based"][ix],
                                                                              threshold=124)
                encoded_dict[k]["input_ids"].append(input_ids)
                encoded_dict[k]["token_type_ids"].append(token_type_ids)

                encoded_dict[k]["fact_input_ids"].append(self.tokenizer.batch_encode_plus(v["unused_facts"][ix],
                                                                                          padding="longest",
                                                                                          truncation=True,
                                                                                          max_length=90)["input_ids"])

                encoded_dict[k]["pandora_strategy_input_ids"].append(pan_ids)
                encoded_dict[k]["pandora_strategy_type_ids"].append(pan_type)

                encoded_dict[k]["essays_strategy_input_ids"].append(essay_ids)
                encoded_dict[k]["essays_strategy_type_ids"].append(essay_type)

                encoded_dict[k]["tgt_pandora_personality"].append(self.annotate_turn_trait_v2(v["tgt_trait_pandora"][ix]))
                encoded_dict[k]["tgt_essays_personality"].append(self.annotate_turn_trait_v2(v["tgt_trait_essays"][ix]))
                encoded_dict[k]["tgt_strategy"].append(self.annotate_strategy_v2(v["tgt_strategy"][ix]))

            tgt_pandora = list(zip(v["tgt_trait_pandora"], v["tgt_strategy"], v["y"]))
            tgt_essays = list(zip(v["tgt_trait_essays"], v["tgt_strategy"], v["y"]))

            tgt_pandora_in, tgt_essays_in = [], []
            for i in tgt_pandora:
                t_pin = self.tokenizer.encode_plus(i[0] + i[1] + self.tokenizer.bos_token + i[-1], add_special_tokens=False,
                                              truncation=True, max_length=124)["input_ids"]
                if i[0].startswith("<pad>") and "blenderbot" in self.mname:
                    t_pin = t_pin[1:]
                tgt_pandora_in.append(t_pin)

            for i in tgt_essays:
                t_esy = self.tokenizer.encode_plus(i[0] + i[1] + self.tokenizer.bos_token + i[-1], add_special_tokens=False,
                                              truncation=True, max_length=124)["input_ids"]
                if i[0].startswith("<pad>") and "blenderbot" in self.mname:
                    t_esy = t_esy[1:]
                tgt_essays_in.append(t_esy)

            idx_pandora_lst = [i.index(self.tokenizer.bos_token_id) for i in tgt_pandora_in]
            idx_essays_lst = [i.index(self.tokenizer.bos_token_id) for i in tgt_essays_in]

            tgt_pandora_out = [[-100] * len(i[0][:i[1]]) + i[0][i[1] + 1:] + [self.tokenizer.eos_token_id] for i in
                               list(zip(tgt_pandora_in, idx_pandora_lst))]
            tgt_essays_out = [[-100] * len(i[0][:i[1]]) + i[0][i[1] + 1:] + [self.tokenizer.eos_token_id] for i in
                              list(zip(tgt_essays_in, idx_essays_lst))]

            encoded_dict[k]["target_ids_in_pandora"] = tgt_pandora_in
            encoded_dict[k]["target_ids_out_pandora"] = tgt_pandora_out

            encoded_dict[k]["target_ids_in_essays"] = tgt_essays_in
            encoded_dict[k]["target_ids_out_essays"] = tgt_essays_out

            encoded_dict[k]["fact_label"] = v["fact_labels"]
        return encoded_dict


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment_num", type=str, help="Experiment from config")
    parser.add_argument("--input_dict", type=str, help="Experiment from config")
    argv = parser.parse_args()
    experiment_num = argv.experiment_num
    input_dict = argv.input_dict
    local_rank = 0
    contradiction_penalty = False
    use_gpu = False

    config_dict = json.load(open("./config.json", "r"))
    ablation_remove = config_dict[experiment_num]["ablation_remove"]
    dataset = config_dict[experiment_num]["dataset"]
    encoder_layers = int(config_dict[experiment_num]["encoder_layers"])
    mname = config_dict[experiment_num]["mname"]
    training_file = config_dict[experiment_num]["training_file"]
    trait_in_decoder = config_dict[experiment_num]["trait_in_decoder"]
    uni_agent = config_dict[experiment_num]["uni_agent"]
    max_pos_embeddings = config_dict[experiment_num]["max_pos_embeddings"]
    model_name = str(config_dict[experiment_num]["dataset"]) + "_dataset_experiment_" + str(experiment_num) + ".pt"
    formatted_examples = pickle.load(open(input_dict, "rb"))

    if uni_agent == "false":
        uni_agent = False
    else:
        uni_agent = True

    if trait_in_decoder == "false":
        trait_in_decoder = False
    else:
        trait_in_decoder = True

    try:
        if "pandora" in config_dict[experiment_num]["encoder"].lower() or \
                "pandora" in config_dict[experiment_num]["decoder"].lower():
            pandora_personality = True
        else:
            pandora_personality = False
    except:
        pandora_personality = False

    inferencer = DataProcesor(mname, ablation_remove, dataset, encoder_layers, training_file, trait_in_decoder,
                              local_rank=local_rank, batch_size=32, num_workers=2, num_train=None, uni_agent=uni_agent,
                              lr=2e-5, N_EPOCHS=1, accum_iter=1, contradiction_penalty=contradiction_penalty,
                              intent_model_path=None, big5_model_path=None, loss_alpha=None,
                              pandora_personality=pandora_personality, skip_training=True)

    encoded_dict = inferencer.format_inputs(formatted_examples)
