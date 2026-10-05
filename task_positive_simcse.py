import luigi
import csv
import os
import random
import numpy as np
import pandas as pd
import nltk
from task_extract_sample import TaskExtractSample

SAMPLE = 500_000

class TaskPositiveSimCSE(luigi.Task):

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
            
            reviews = df['review'].tolist()
            
            # Create triplets and split into train/val
            for _ in range(0,SAMPLE):
                review = random.choice(reviews)
                triplet = [review, review, 1]
                train_datasets.append(triplet)
        
        # Randomize datasets before saving
        random.shuffle(train_datasets)
        
        # Create output directory if it doesn't exist
        os.makedirs('data/training/simcse', exist_ok=True)
        
        # Write training dataset
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            writer.writerows(train_datasets)

    def output(self):
        return luigi.LocalTarget(f'data/training/simcse/train.tsv')
    
if __name__ == '__main__':
    luigi.build([TaskPositiveSimCSE()], local_scheduler=True)
