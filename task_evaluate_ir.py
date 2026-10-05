import luigi
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator
from task_generate_evaluation_ir_acos import TaskGenerateEvaluationIRACOS
from task_generate_evaluation_ir_oats import TaskGenerateEvaluationIROATS
from utils.senticse_embeddings import SenticseEmbeddings




class TaskEvaluateIR(luigi.Task):

    domain = luigi.ChoiceParameter(choices=['restaurant', "hotel", "laptop", "nano"])
    sbert = luigi.Parameter(default="intfloat/e5-base-v2")
    subset = luigi.ChoiceParameter(choices=['ACSA', 'ACD', 'ASC','ACSA-Neutral', 'ASC-Neutral'])
    file = luigi.Parameter()

    def requires(self):
        if self.domain == 'hotel':
            return TaskGenerateEvaluationIROATS(subset=self.subset)
        elif self.domain in ['restaurant', 'laptop']:
            return TaskGenerateEvaluationIRACOS(subset=self.subset, domain=self.domain)
    
    def run(self):
        
        if 'SentiCSE' in self.sbert:
            sbert =  SenticseEmbeddings(self.sbert)
        else:
            

            if 'Qwen' in self.sbert:
                sbert = SentenceTransformer(
                    self.sbert,
                    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                    tokenizer_kwargs={"padding_side": "left"},
                )
            else:
                sbert = SentenceTransformer(self.sbert)

        if self.domain in ['hotel', 'laptop', 'restaurant']:
            with self.input()['doc'].open('r') as f:
                d = json.load(f)

            with self.input()['map'].open('r') as f:
                q_d = json.load(f)

            with self.input()['query'].open('r') as f:
                q = json.load(f)

                # Use InformationRetrievalEvaluator
            if self.sbert == "intfloat/e5-base-v2":
                e = InformationRetrievalEvaluator(q, d, q_d, name=f"{self.domain}", batch_size=2, query_prompt="query: ", corpus_prompt="query: ")
            elif self.sbert == "intfloat/e5-mistral-7b-instruct":
                e = InformationRetrievalEvaluator(q, d, q_d, name=f"{self.domain}", batch_size=2, query_prompt="Instruct: Retrieve semantically similar text.\nQuery: ")
            elif self.sbert == "Qwen/Qwen3-Embedding-8B":
                e = InformationRetrievalEvaluator(q, d, q_d, name=f"{self.domain}", batch_size=2, query_prompt="Instruct: Retrieve semantically similar text.\nQuery: ")
            else:
                e = InformationRetrievalEvaluator(q, d, q_d, name=f"{self.domain}", batch_size=2)
            e(sbert, output_path=self.output()["ir"].path)
        else:
            if self.sbert == "intfloat/e5-base-v2":
                e = NanoBEIREvaluator(batch_size=2, query_prompts="query: ", corpus_prompts="passage: ")
            elif self.sbert == "intfloat/e5-mistral-7b-instruct":
                e = NanoBEIREvaluator(batch_size=2, query_prompts="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ")
            elif self.sbert == "Qwen/Qwen3-Embedding-8B":
                e = NanoBEIREvaluator(batch_size=2, query_prompts="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ")
            else:
                e = NanoBEIREvaluator(batch_size=2)

            e(sbert, output_path=self.output()["ir"].path)

    def output(self):
        return {
            "ir": luigi.LocalTarget(f'data/ir_eval_{self.subset}/{self.domain}/{self.file}')
        }

if __name__ == '__main__':
    from glob import glob
    tasks = []

    models = []

    models.extend([
        "activebus/BERT_Review",
        "veroman/TourBERT",
        "intfloat/e5-base-v2",
        "intfloat/e5-mistral-7b-instruct",
        "google/embeddinggemma-300m",
        "all-mpnet-base-v2",
        "Qwen/Qwen3-Embedding-8B",
        "vahidthegreat/StanceAware-SBERT",
        "DILAB-HYU/SentiCSE",
        "Alibaba-NLP/gte-modernbert-base"
    ])

    
    for m in glob("data/sbert/*"):
        if 'relative' not in m: continue
        models.append(m)

    for subset in ['ACSA']:

        for d in ['laptop', 'restaurant', 'hotel', 'nano']:

            if d == "nano" and subset != 'ACSA': continue

            for model in models:
                tasks.append(TaskEvaluateIR(
                    domain=d, 
                    sbert=model, 
                    file=f"{model.split('/')[-1]}", 
                    subset=subset
                ))


    luigi.build(tasks, local_scheduler=True)