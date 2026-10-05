import luigi
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer, util
from sentence_transformers.sentence_transformer.losses import GISTEmbedLoss, MultipleNegativesRankingLoss
from sentence_transformers.base.sampler import BatchSamplers
from peft import LoraConfig, TaskType
import json
import torch
from transformers import EarlyStoppingCallback

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("Allocated:", torch.cuda.memory_allocated(0) / 1024**3, "GiB")
    print("Reserved:", torch.cuda.memory_reserved(0) / 1024**3, "GiB")


lr = {
    'all-MiniLM-L6-v2': {
        'default': 2e-5,
        'LoRA': 1e-4
    },
    'all-mpnet-base-v2': {
        'default': 2e-5,
        'LoRA': 1e-4
    }
}

class TaskTrainSBERT(luigi.Task):

    sbert = luigi.Parameter(default="intfloat/e5-base-v2")
    guide = luigi.Parameter(default='all-mpnet-base-v2')
    loss = luigi.ChoiceParameter(choices=['gist','mnrl', 'rand'])
    is_lora = luigi.IntParameter(default=0)
    batch = luigi.IntParameter(default=32)
    epoch = luigi.IntParameter(default=3)
    dataset = luigi.Parameter()
    output_dir = luigi.Parameter()
    evaluation_step = luigi.FloatParameter(default=0.01)
    seed = luigi.IntParameter(default=42)
    margin = luigi.FloatParameter(default=0)

       
    def run(self):
        # Read train and validation datasets
        df_train = pd.read_csv(f'{self.dataset}/train.tsv', delimiter='\t', header=0)
        #df_val = pd.read_csv(f'{self.dataset}/val.tsv', delimiter='\t', header=0)

        
        # Convert text columns to string
        for df in [df_train]:
            df['text1'] = df['text1'].astype(str)
            df['text2'] = df['text2'].astype(str)
            if self.loss in ['gist','mnrl']:
                df['label'] = 1
      
        # Create training dataset
        train_dataset = Dataset.from_dict(
            {
                'text1': df_train['text1'].tolist(),
                'text2': df_train['text2'].tolist(),
                'label': df_train['label'].tolist()
            }
        )

        d_evaluator = dict()

        for domain in ['restaurant', 'hotel', 'laptop']:
            d = dict()
            for key in ['doc', 'map', 'query']:
                with open(f"data/ir/{domain}/ACSA_{key}.json") as f:
                    d[key] = json.load(f)
            d_evaluator[domain] = InformationRetrievalEvaluator(d['query'], d['doc'], d['map'], name=domain)
            d_evaluator['nanobeir'] = NanoBEIREvaluator()
        # # Create validation dataset
        # val_dataset = Dataset.from_dict(
        #     {
        #         'text1': df_val['text1'].tolist(),
        #         'text2': df_val['text2'].tolist(),
        #         'label': df_val['label'].tolist()
        #     }
        # )
       
        model = SentenceTransformer(
            self.sbert
            # model_kwargs={
            #     "attn_implementation": "sdpa"  # Enable SDPA attention
            # }
        )
               
        if self.is_lora > 0:

            target_modules = ["query", "key", "value"]
            if 'mpnet' in self.sbert:
                target_modules = ["q", "k", "v", "o"]
            if 'modernbert' in self.sbert:
                target_modules = ["attn.Wqkv", "attn.Wo"]

            # Same as StanceSBERT
            peft_config = LoraConfig(
                target_modules=target_modules,
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=self.is_lora,
                lora_alpha=self.is_lora,
                lora_dropout=0.05
            )
        
            model.add_adapter(peft_config)

        loss = None
        if self.loss == 'gist':
            guide = SentenceTransformer(self.guide)
            loss = GISTEmbedLoss(model=model, guide=guide, temperature=0.05, margin=self.margin, margin_strategy='relative')
        elif self.loss == 'mnrl':
            loss = MultipleNegativesRankingLoss(model=model, scale=20, similarity_fct=util.cos_sim)
        elif self.loss == 'rand':
            self.loss = GISTEmbedRandom(model=model, mask_ratio=0.5, seed=self.seed, temperature=0.05)


        training_args = SentenceTransformerTrainingArguments(
                output_dir=str(self.output().path),
                num_train_epochs=self.epoch,
                learning_rate=lr[self.sbert.split('/')[-1]]['LoRA' if self.is_lora else 'default'],
                warmup_steps=0.1 if 'modernbert' in self.sbert else 0.0,
                lr_scheduler_type="linear",
                weight_decay=0.0,
                per_device_train_batch_size=self.batch,
                per_device_eval_batch_size=self.batch,
                save_total_limit=1,
                bf16=True,
                save_steps=self.evaluation_step,
                batch_sampler=BatchSamplers.NO_DUPLICATES if self.loss in ['gist', 'mnrl'] else BatchSamplers.BATCH_SAMPLER,
                eval_strategy="steps",
                eval_on_start=True,
                eval_steps=self.evaluation_step,
                metric_for_best_model="eval_restaurant_cosine_map@100",
                logging_steps=500,
                greater_is_better=True,
                load_best_model_at_end=False,
                seed=self.seed,
            )

        early = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.0
        )

        trainer = SentenceTransformerTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    loss=loss,
                    evaluator=[v for v in d_evaluator.values()],
                    callbacks=[early]
        )

        trainer.train()

        model.save_pretrained(str(self.output().path))

    def output(self):
        return luigi.LocalTarget(f'data/sbert/{self.output_dir}')

if __name__ == "__main__":
    import sys
    import csv

    tsv_file = sys.argv[1]
    task_index = int(sys.argv[2])  # zero-based index in TSV, excluding header

    with open(tsv_file, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter="\t"))
        params = reader[task_index]

    model = params["model"].strip()
    loss = params["loss"].strip()
    lora = int(params["lora"].strip())
    dataset = params["dataset"].strip()
    batch = int(params["batch"].strip())
    guide = params["guide"].strip()
    seed = int(params["seed"].strip())
    margin = float(params["margin"].strip())

    guided_suffix = {
        "gist": f"_guided_{guide.split('/')[-1]}",
        "rand": "_rand",
    }.get(loss, "") 

    margin_suffix = f"_margin_{margin}" if loss == "gist" and margin != 0 else ""
    lora_suffix = f"_lora_{str(lora)}" if lora > 0 else ""


    output_dir = (
        f"{dataset.replace('/', '_')}_"
        f"{model.split('/')[-1]}"
        f"{lora_suffix}"
        f"{guided_suffix}"
        f"{margin_suffix}"
        f"_batch_{batch}_seed_{seed}"
    )

    luigi.build(
        [
            TaskTrainSBERT(
                sbert=model,
                loss=loss,
                is_lora=lora,
                margin=margin if margin != 0 else 0,
                dataset=f"data/training/{dataset}",
                batch=batch,
                seed=seed,
                guide=guide,
                output_dir=output_dir,
            )
        ],
        local_scheduler=True,
        workers=1,
    )
