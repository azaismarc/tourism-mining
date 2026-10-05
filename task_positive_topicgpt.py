import luigi
import os
import csv
import random
from task_extract_topicgpt import TaskExtractTopicGPT
import pandas as pd
from tqdm import tqdm

SAMPLE = 500_000


class TaskPositiveTopicGPT(luigi.Task):

    generation = luigi.ChoiceParameter(choices=[1, 2, 3], var_type=int)
    # 1 -> default seed
    # 2 -> generation
    # 3 -> default prompt
    model = luigi.Parameter(default="meta-llama/Llama-3.1-8B-Instruct")

    def requires(self):
        return {
            'restaurant': TaskExtractTopicGPT(domain='restaurant', model=self.model, generation=self.generation),
            'hotel': TaskExtractTopicGPT(domain='hotel', model=self.model, generation=self.generation)
        }

    def run(self):
        train_datasets = []
        
        # Process both domains
        for domain in ['restaurant', 'hotel']:
            df = pd.read_csv(self.input()[domain].path, delimiter='\t')
            
            reviews = df['review'].tolist()

            N = df['rating'].nunique()
            # Group by rating to ensure equal split
            for rating in df['rating'].unique():
            
                rating_df = df[df['rating'] == rating]
                reviews = rating_df['review'].to_list()              
               
                
                # Create triplets with rating as label and split into train/val
                for _ in tqdm(range(0,SAMPLE//N)):
                    triplet = [random.choice(reviews), random.choice(reviews), rating]
                    train_datasets.append(triplet)
                    
        # Randomize datasets before saving
        random.shuffle(train_datasets)

        # Write training dataset
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            writer.writerows(train_datasets)
        
     

    def output(self):
        model_path = self.model.split("/")[-1]

        # Map generation types to readable names
        gen_type = {
            1: "seed",
            2: "generated",
            3: "default"
        }[self.generation]
        
        folder_path = f'data/training/topicgpt/{model_path}/{gen_type}'

        # Create folder if it doesn't already exist
        os.makedirs(folder_path, exist_ok=True)

        return luigi.LocalTarget(f'{folder_path}/train.tsv')
        


if __name__ == "__main__":    
   
    luigi.build(
        TaskPositiveTopicGPT(generation=1),
        local_scheduler=True,
        workers=1,
    )