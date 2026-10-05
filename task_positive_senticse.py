import luigi
import csv
import os
import random
import numpy as np
import pandas as pd
from task_extract_sample import TaskExtractSample
from tqdm import tqdm


SAMPLE = 500_000


class TaskPositiveSentiCSE(luigi.Task):

    def requires(self):
        return {
            'restaurant': TaskExtractSample(domain='restaurant'),
            'hotel': TaskExtractSample(domain='hotel')
        }

    def run(self):
        train_datasets = []
        
        # Process both domains
        for domain in ['restaurant', 'hotel']:
            df = pd.read_csv(self.input()[domain].path, delimiter='\t')

            N = df['rating'].nunique()
            # Group by rating to ensure equal split
            for rating in df['rating'].unique():

                rating_df = df[df['rating'] == rating]
                reviews = rating_df['review'].tolist()
                
                # Create triplets with rating as label and split into train/val
                for _ in range(0,SAMPLE//N):
                    triplet = [random.choice(reviews), random.choice(reviews), int(rating)]
                    train_datasets.append(triplet)
        
        # Randomize datasets before saving
        random.shuffle(train_datasets)
        
        # Create output directory if it doesn't exist
        
        # Write training dataset
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            writer.writerows(train_datasets)

    def output(self):
        os.makedirs('data/training/senticse', exist_ok=True)
        return luigi.LocalTarget(f'data/training/senticse/train.tsv')
        
    
if __name__ == '__main__':
    luigi.build([TaskPositiveSentiCSE()], local_scheduler=True)
