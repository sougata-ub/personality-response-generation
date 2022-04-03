# Stylistic Response Generation by Controlling Personality Traits and Intent
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the implementation of the paper:

Stylistic Response Generation by Controlling Personality Traits and Intent. [Sougata Saha](https://www.linkedin.com/in/sougata-saha-8964149a/), [Souvik Das](https://www.linkedin.com/in/souvikdas23/), [Rohini Srihari](https://www.linkedin.com/in/rohinisrihari/)

## Abstract
Personality traits influence human actions and thoughts, which is manifested in day to day conversations. Although glimpses of personality traits are observable in existing open domain conversation corpora, leveraging generic language modelling for response generation overlooks the interlocutor idiosyncrasies, resulting in non-customizable personality agnostic responses. With the motivation of enabling stylistically configurable response generators, in this paper we experiment with end-to-end mechanisms to ground neural response generators based on both (i) interlocutor Big-5 personality traits, and (ii) discourse intent as stylistic control codes. Since most of the existing large scale open domain chat corpora do not include Big-5 personality traits and discourse intent, we employ automatic annotation schemes to enrich the corpora with noisy estimates of personality and intent annotations, and further assess the impact of using such features as control codes for response generation using automatic evaluation metrics, ablation studies and human judgement. Our experiments illustrate the effectiveness of this strategy resulting in improvements to existing benchmarks. Additionally, we yield two silver standard annotated corpora with intents and personality traits annotated, which can be of use to the research community.

### Training and Inference
Our model is build on top of [Huggingface](https://huggingface.co/) transformer library, and can be found in [this](https://github.com/sougata-ub/transformers/tree/encoder_experiment_v2) repo.

You can train and evaluate all experiments by uncommenting the desired command in the `runner.sh` script, and executing the shell script. Example: `nohup bash runner.sh 104 107 > log.txt 2>&1 &` runs experiment numbers 104 to 106 sequentially. All the different configurations for the experiments can be found in the `config.json` file.
In order to experiment with different parameters, you can directly execute the `run_training_v2.py` script. Sample command below:

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port 9999 ./run_training_v2.py --batch_size 16 --num_epochs 15 --learning_rate 0.00002 --experiment_num "$i" --path "/home/sougatas/wikibot_naacl_2022/data/" --num_workers 4 --accum_iter 2 --results_folder "/home/sougatas/wikibot_naacl_2022/results/results_wow_blender_v4/" --model_folder "/home/sougatas/wikibot_naacl_2022/models/models_v2/"
```

### Datasets:
1. Wizard of Wikipedia data for BART: https://drive.google.com/file/d/1wFjltDqb8HCY2zaXr2bEzU8_yYIaUyWQ/view?usp=sharing
2. Wizard of Wikipedia data for Blenderbot: https://drive.google.com/file/d/1CetIysBWgM6-DQs3GCnciAg-AX-z5OdB/view?usp=sharing
3. Topical Chat data for BART: https://drive.google.com/file/d/1OcL1_9dJEQoL6svsF8ANt0NaBykVm3O3/view?usp=sharing
4. Topical Chat data for Blenderbot: https://drive.google.com/file/d/1NzLyYM3i5fWQaDKGFnhNNsayJkSmDsrF/view?usp=sharing

### Additional Models
1. BERT based intent classifier: https://drive.google.com/file/d/1jbGeLyfuaTRh9N3o3VSYiNqiX71d93lS/view?usp=sharing
2. Big 5 personality predictor trained on PANDORA dataset: https://drive.google.com/file/d/1Ltn53jj-0wfk5UjY2idxWX9HOkASgDnp/view?usp=sharing
