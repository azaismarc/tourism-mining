import luigi
import pandas as pd
import json

from task_extract_sample import TaskExtractSample

class TaskTopicGPTExtractSampleGenerateTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    N = luigi.Parameter(default=100)

    def requires(self):
        return TaskExtractSample(self.domain)

    def run(self):


        df = pd.read_csv(self.input().path, delimiter='\t')

        df = df.groupby('rating').sample(n=self.N, random_state=42)

        # If you want lists
        reviews = df['review'].tolist()
        
        data = [{"text": r} for r in reviews]

        # Save as JSONL
        with self.output().open('w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def output(self):
        return luigi.LocalTarget(
            f'data/topicgpt/sample/{self.domain}/generation_sample.jsonl'
        )    
    
if __name__ == '__main__':
    tasks = [
        TaskTopicGPTExtractSampleGenerateTopic(domain=domain) for domain in ['restaurant', 'hotel']
    ]
    luigi.build(tasks, local_scheduler=True)