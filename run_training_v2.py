from datasets import load_metric
import torch
import argparse
from trainer import Trainer
import time
import tqdm
import math
import pickle
import json
import shutil
import pandas as pd
import os
import os.path
from datetime import datetime

# os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


def main():
    N_EPOCHS = 8
    batch_size = 8
    lr = 2e-5
    mname = 'facebook/blenderbot-400M-distill'
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    model_name = "least_loss_model.pt"
    training_file = "./tc_encoded_dict_aug_2021.pkl"
    pretrained_model, pretrained_log = None, None
    accum_iter = 4
    best_valid_loss = float('inf')
    ablation_remove = "none"
    num_workers = 4
    max_pos_embeddings = 220
    dataset, num_train = None, None
    # n_codes = 10
    trait_in_decoder = "false"
    uni_agent = "false"
    encoder_layers = 2
    experiment_num = None
    config_dict = json.load(open("./config.json", "r"))
    path = "./"
    contradiction_penalty = "false"
    intent_model_path = "./drive/MyDrive/protoai_labs/intent_classifier_model/intent_classifier_bert"
    big5_model_path = "./drive/MyDrive/PANDORA_dataset/pandora_big5_model_sentence_v1.pt"
    loss_alpha = None #0.7
    results_folder = "./drive/MyDrive/wikibot_naacl_results/"
    results_file = "ExperimentResults.csv"
    skip_training = "false"
    device_num = 0

    """
    none -> Keep traits + strategy + fact pred + opinion pred
    trait -> Keep strategy + fact pred + opinion pred
    strategy -> Keep traits + fact pred + opinion pred
    trait|strategy -> Keep fact pred + opinion pred
    ...etc
    """

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--local_rank", type=int,
    #                     help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--mname", type=str, help="Pretrained transformer.", default=mname)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=N_EPOCHS)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=lr)
    parser.add_argument("--model_name", type=str, help="Model Name.", default=model_name)
    parser.add_argument("--pre_validate", type=int, help="Run validation first.", default=0)
    parser.add_argument("--training_file_name", type=str, help="File name cointaining examples", default=training_file)
    parser.add_argument("--pretrained_model", type=str, help="Pre-trained pt file from where weights will be loaded",
                        default=pretrained_model)
    parser.add_argument("--pretrained_log", type=str, help="Pre-trained .pkl log file from where stats will be loaded",
                        default=pretrained_log)
    parser.add_argument("--accum_iter", type=int, help="Gradient Accumulation steps", default=accum_iter)
    parser.add_argument("--best_valid_loss", type=float, help="Best validation loss", default=best_valid_loss)
    parser.add_argument("--ablation_remove", type=str, help="Things to remove for ablation study",
                        default=ablation_remove)
    parser.add_argument("--num_workers", type=int, help="Number of dataloader workers", default=num_workers)
    parser.add_argument("--max_pos_embeddings", type=int, help="Max length of input", default=max_pos_embeddings)
    parser.add_argument("--dataset", type=str, help="ID of dataset to filter", default=dataset)
    parser.add_argument("--num_train", type=int, help="Number of training examples to use", default=num_train)
    parser.add_argument("--trait_in_decoder", type=str, help="Is trait in encoder or decoder", default=trait_in_decoder)
    parser.add_argument("--encoder_layers", type=int, help="Num layers in encoder", default=encoder_layers)
    parser.add_argument("--uni_agent", type=str, help="Model 1 or 2 speakers in TC", default=uni_agent)
    parser.add_argument("--experiment_num", type=str, help="Experiment from config", default=experiment_num)
    parser.add_argument("--path", type=str, help="Path to training file", default=path)
    parser.add_argument("--contradiction_penalty", type=str, help="Add entailment penalty",
                        default=contradiction_penalty)
    parser.add_argument("--intent_model_path", type=str, help="Intent model path", default=intent_model_path)
    parser.add_argument("--big5_model_path", type=str, help="Big5 Pred model path", default=big5_model_path)
    parser.add_argument("--loss_alpha", type=float, help="Adaptive smoothening factor", default=loss_alpha)
    parser.add_argument("--results_folder", type=str, help="Location of results", default=results_folder)
    parser.add_argument("--model_folder", type=str, help="Location of trained model", default=results_folder)
    parser.add_argument("--skip_training", type=str, help="Location of trained model", default=skip_training)
    parser.add_argument("--device_num", type=int, help="CUDA device number", default=device_num)

    argv = parser.parse_args()

    # local_rank = argv.local_rank
    experiment_num = argv.experiment_num
    if experiment_num is not None:
        print("Reading experiment parameters from config file!")
        print("CONFIG:",config_dict[experiment_num],"\n")
        ablation_remove = config_dict[experiment_num]["ablation_remove"]
        dataset = config_dict[experiment_num]["dataset"]
        encoder_layers = int(config_dict[experiment_num]["encoder_layers"])
        mname = config_dict[experiment_num]["mname"]
        training_file = config_dict[experiment_num]["training_file"]
        trait_in_decoder = config_dict[experiment_num]["trait_in_decoder"]
        uni_agent = config_dict[experiment_num]["uni_agent"]
        max_pos_embeddings = config_dict[experiment_num]["max_pos_embeddings"]
        model_name = str(config_dict[experiment_num]["dataset"]) + "_dataset_experiment_" + str(experiment_num) + ".pt"
    else:
        print("Not reading parameters from config file!")
        mname = argv.mname
        model_name = argv.model_name
        training_file = argv.training_file_name
        pretrained_model = argv.pretrained_model
        pretrained_log = argv.pretrained_log
        ablation_remove = argv.ablation_remove
        max_pos_embeddings = argv.max_pos_embeddings
        dataset = argv.dataset
        uni_agent = argv.uni_agent
        encoder_layers = argv.encoder_layers
        trait_in_decoder = argv.trait_in_decoder

    """ Generic Args """
    skip_training = argv.skip_training
    device_num = argv.device_num
    if skip_training == "false":
        skip_training = False
    else:
        skip_training = True
    print("\nskip_training\n",skip_training)
    if not skip_training:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = device_num
    N_EPOCHS = argv.num_epochs
    batch_size = argv.batch_size
    lr = argv.learning_rate
    pre_validate = argv.pre_validate
    accum_iter = argv.accum_iter
    best_valid_loss = argv.best_valid_loss
    num_workers = argv.num_workers
    num_train = argv.num_train
    path = argv.path
    training_file = path + training_file
    contradiction_penalty = argv.contradiction_penalty
    loss_alpha = argv.loss_alpha
    intent_model_path = argv.intent_model_path
    big5_model_path = argv.big5_model_path
    results_folder = argv.results_folder
    model_folder = argv.model_folder

    if uni_agent == "false":
        uni_agent = False
    else:
        uni_agent = True

    if trait_in_decoder == "false":
        trait_in_decoder = False
    else:
        trait_in_decoder = True

    if contradiction_penalty == "false":
        contradiction_penalty = False
    else:
        contradiction_penalty = True

    try:
        if "pandora" in config_dict[experiment_num]["encoder"].lower() or \
                "pandora" in config_dict[experiment_num]["decoder"].lower():
            pandora_personality = True
        else:
            pandora_personality = False
    except:
        pandora_personality = False

    print("Got Args: ")
    print("mname", mname)
    print("local_rank", local_rank)
    print("num_epochs", N_EPOCHS)
    print("batch_size", batch_size)
    print("learning_rate", lr)
    print("model_name", model_name)
    print("pre_validate", pre_validate)
    print("training_file_name", training_file)
    print("pretrained_model", pretrained_model)
    print("accum_iter", accum_iter)
    print("best_valid_loss", best_valid_loss)
    print("ablation_remove", ablation_remove)
    print("num_workers", num_workers)
    print("max_pos_embeddings", max_pos_embeddings)
    print("dataset", dataset)
    print("num_train", num_train)
    print("trait_in_decoder", trait_in_decoder)
    print("encoder_layers", encoder_layers)
    print("loss_alpha", loss_alpha)

    logfile_name = results_folder + model_name + "_LOG.pkl"
    if experiment_num is not None:
        log = {"config": config_dict[experiment_num], "train": {}, "valid": {}, "test": {}}
    else:
        log = {"train": {}, "valid": {}, "test": {}}
    torch.distributed.init_process_group(backend="nccl")

    trainer = Trainer(mname, ablation_remove, dataset, encoder_layers, training_file, trait_in_decoder=trait_in_decoder,
                      local_rank=local_rank, batch_size=batch_size, num_workers=num_workers, num_train=num_train,
                      uni_agent=uni_agent, lr=lr, N_EPOCHS=N_EPOCHS, accum_iter=accum_iter,
                      contradiction_penalty=contradiction_penalty, intent_model_path=intent_model_path,
                      big5_model_path=big5_model_path, loss_alpha=loss_alpha, pandora_personality=pandora_personality,
                      skip_training=skip_training)

    if not skip_training:
        for epoch in range(N_EPOCHS):

            print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
            start_time = time.time()
            tr_l, tr_lm, tr_fl, tr_strat, tr_persona, tr_ce = trainer.train()
            if local_rank == 0:
                print(f'\tTrain Total Loss: {tr_l:.3f} | Train LM Loss: {tr_lm:.3f} | Train CE Loss: {tr_ce:.3f} | '
                      f'Train Fact Loss: {tr_fl:.3f} | '
                      f'Train Perplexity LM: {math.exp(tr_lm):.3f} | Train Perplexity CE: {math.exp(tr_ce):.3f}')
                print(f'\tTrain Strategy Loss: {tr_strat:.3f} | Train Personality Loss: {tr_persona:.3f}')
                print("-" * 10)
                print("\n")
                log["train"][epoch] = [tr_l, tr_lm, tr_fl, tr_strat, tr_persona,
                                       tr_ce, math.exp(tr_lm), math.exp(tr_ce)]

            end_time1 = time.time()
            epoch_mins, epoch_secs = trainer.epoch_time(start_time, end_time1)
            print(f'Training Time: {epoch_mins}m {epoch_secs}s')

            log["valid"][epoch] = {}
            if local_rank == 0:
                tmp_lm_loss = []

                val_1, tmp = trainer.run_evaluation(trainer.ddp_model, trainer.valid_dataloader_1, trainer.dataset, "val_1")
                log["valid"][epoch][trainer.dataset + "_val_1"] = tmp

                val_2, tmp = trainer.run_evaluation(trainer.ddp_model, trainer.valid_dataloader_2, trainer.dataset, "val_2")
                log["valid"][epoch][trainer.dataset + "_val_2"] = tmp

                tmp_lm_loss.extend([val_1, val_2])

                end_time2 = time.time()
                epoch_mins, epoch_secs = trainer.epoch_time(end_time1, end_time2)

                print("\n")
                print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print("=" * 75)
                print(f'Validation Time: {epoch_mins}m {epoch_secs}s')
                if sum(tmp_lm_loss) / len(tmp_lm_loss) < best_valid_loss:
                    best_valid_loss = sum(tmp_lm_loss) / len(tmp_lm_loss)
                    torch.save(trainer.ddp_model.state_dict(), model_folder + model_name)
                    print("\nBest Model Saved !! Best Validation loss", round(best_valid_loss, 5))
                else:
                    torch.save(trainer.ddp_model.state_dict(), model_folder + "epoch_" + str(epoch) + "_" + model_name)
                    print("\n Non best Model Saved. Terminating Training")
                    break

    if local_rank == 0 or skip_training:
        print("\nLoading best weights. \n")
        state_dict = torch.load(model_folder + model_name)
        if num_workers > 2:
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        trainer.model.load_state_dict(state_dict)
        print("Best model loaded")
        test_lm_loss = []
        log["test"] = {}

        print("Evaluating on Test Set. \n")
        test_1, tmp = trainer.run_evaluation(trainer.model, trainer.test_dataloader_1, trainer.dataset, "test_1")
        ppl1 = tmp[7]
        log["test"][trainer.dataset + "_test_1"] = tmp

        test_2, tmp = trainer.run_evaluation(trainer.model, trainer.test_dataloader_2, trainer.dataset, "test_2")
        ppl2 = tmp[7]
        log["test"][trainer.dataset + "_test_2"] = tmp
        test_lm_loss.extend([test_1, test_2])

        print("Generating responses ! \n")

        test_1_dict, test1_lst = trainer.generate_response(trainer.test_gen_dataloader_1, rouge, "test_1", golden=True)
        test_2_dict, test2_lst = trainer.generate_response(trainer.test_gen_dataloader_2, rouge, "test_2", golden=True)

        test_1_dict_ng, test1_lst_ng = trainer.generate_response(trainer.test_gen_dataloader_1, rouge, "test_1",
                                                                 golden=False)
        test_2_dict_ng, test2_lst_ng = trainer.generate_response(trainer.test_gen_dataloader_2, rouge, "test_2",
                                                                 golden=False)

        test1_lst = [experiment_num, ppl1] + test1_lst + test1_lst_ng[1:]
        test2_lst = [experiment_num, ppl2] + test2_lst + test2_lst_ng[1:]
        auto_stats_df = pd.DataFrame([test1_lst, test2_lst], columns=["id", "PPL", "split", "bleu_score_gold",
                                                                      "bleu_4_gold", "rougeL_gold", "bleu_score_ng",
                                                                      "bleu_4_ng", "rougeL_ng"])
        print("Automatic metrics:\n")
        print(auto_stats_df)
        # auto_stats_df.to_csv("./tmp_results.csv", index=False)

        print("Adding labels to output dict. \n")
        test_1_dict["fact_labels"] = trainer.test_dataloader_1_targets["tgt_fact_lbl"]
        test_1_dict["strategy_labels"] = trainer.test_dataloader_1_targets["tgt_strategy"]
        test_1_dict["big5_traits"] = trainer.test_dataloader_1_targets["tgt_big5_traits"]
        test_1_dict["non_golden_dict"] = test_1_dict_ng

        test_2_dict["fact_labels"] = trainer.test_dataloader_2_targets["tgt_fact_lbl"]
        test_2_dict["strategy_labels"] = trainer.test_dataloader_2_targets["tgt_strategy"]
        test_2_dict["big5_traits"] = trainer.test_dataloader_2_targets["tgt_big5_traits"]
        test_2_dict["non_golden_dict"] = test_2_dict_ng

        log["test"][trainer.dataset + "_test_1_auto_metrics"] = test_1_dict
        log["test"][trainer.dataset + "_test_2_auto_metrics"] = test_2_dict
        print("-" * 10)
        pickle.dump(log, open(logfile_name, "wb"))
        print("Dumped log to file:", logfile_name)

        # f = open("./tmp_file.txt", "a")
        # f.write(logfile_name+"\n")
        # f.close()
    os._exit(1)


if __name__ == '__main__':
    main()
