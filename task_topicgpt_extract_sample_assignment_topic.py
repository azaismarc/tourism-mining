import luigi
import pandas as pd
import json
import random
import numpy as np
from task_extract_sample import TaskExtractSample

SAMPLE = 100_000

class TaskTopicGPTExtractSampleAssignmentTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    eval = luigi.BoolParameter(default=False)

    def requires(self):
        return TaskExtractSample(self.domain)

    def run(self):

        if not self.eval:

            df = pd.read_csv(self.input().path, delimiter='\t')

            data = []
            N = df['rating'].nunique()
            for rating in df['rating'].unique():
            
                rating_df = df[df['rating'] == rating]
                reviews = rating_df['review'].tolist()
                
                # Create triplets with rating as label and split into train/val
                for _ in range(0,SAMPLE//N):
                    review = random.choice(reviews)

                    _id = f"{str(len(data))}-{rating}"

                    data.append({
                        "id": _id,
                        "text": review
                    })

        else:
            _path = "data/absa/restaurant/acos/all_data.tsv" if self.domain == "restaurant" else "data/absa/hotel/hotelOATS/all_data.tsv" 
            df = pd.read_csv(_path, sep="\t")
            data = []
            for _id, sentence in zip(df['ID'], df['Sentence']):
                if not sentence or sentence == np.nan: continue
                data.append({
                        "id": _id,
                        "text": sentence
                    })

        with self.output().open('w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        

    def output(self):
        return luigi.LocalTarget(
            f'data/topicgpt/sample/{self.domain}/assignment_sample{"_eval" if self.eval else ""}.jsonl'
        )
    
if __name__ == '__main__':
    tasks = [
        TaskTopicGPTExtractSampleAssignmentTopic(domain=domain, eval=True) for domain in ['restaurant', 'hotel']
    ]
    luigi.build(tasks, local_scheduler=True)