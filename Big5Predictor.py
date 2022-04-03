import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaTokenizerFast
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from torch.cuda.amp import autocast


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


class Big5Predictor(object):
    def __init__(self, model_path):
        self.n_class = 5
        self.DEVICE = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else torch.device('cpu')
        self.low = {"<topical_chat>": [-0.09307861,  0.30639648, -0.22143555, -0.30737305,  0.08392334],
                    "<wizard_of_wikipedia>": [-0.10498047,  0.32348633, -0.21337891, -0.31713867,  0.07226562]}
        self.model, self.tokenizer = self._load_model(model_path)
        self.batch_size = 64

    def _load_model(self, model_path):
        config = RobertaConfig.from_pretrained("distilroberta-base")
        config.output_hidden_states = True
        base = RobertaModel.from_pretrained("distilroberta-base", config=config)
        model = Model(base, config.hidden_size, self.n_class)#.to(self.DEVICE)

        state_dict = torch.load(model_path)
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
        print("\nBig5 Model Loaded\n")
        return model, tokenizer

    def _process_input(self, text_list):
        test_dict = {"x": text_list}
        test_dataset = Dataset.from_dict(test_dict)
        test_dataset = test_dataset.map(
            lambda e: self.tokenizer(e['x'], truncation=True, padding="max_length", max_length=100), batched=True,
            batch_size=self.batch_size)
        test_dataset.set_format(type='torch', columns=['attention_mask', 'input_ids'])
        test_dataloader = DataLoader(dataset=test_dataset, sampler=SequentialSampler(test_dataset),
                                     batch_size=self.batch_size)
        return test_dataloader

    def predict(self, text_list, dataset):
        dataloader = self._process_input(text_list)
        predicted_list, predicted_class = [], []
        for ix, batch in tqdm(enumerate(dataloader)):
            batch = {k: v.to(self.DEVICE) for k, v in batch.items()}
            with autocast():
                with torch.no_grad():
                    pred = self.model(batch)
            predicted_list.extend(pred.tolist())
        print("{} Big5 Trait predictions made".format(len(predicted_list)))
        for i in predicted_list:
            tmp = []
            for ix, j in enumerate(i):
                if j < self.low[dataset][ix]:
                    tmp.append(0)
                else:
                    tmp.append(1)
            predicted_class.append(tmp)
        return predicted_class, predicted_list
