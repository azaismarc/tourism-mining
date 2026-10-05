import luigi
from glob import glob
from tqdm import tqdm
import csv

class TaskExtractReview(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])

    def run(self):
        import fasttext

        clf_lg = fasttext.load_model('data/fasttext/lid.176.bin')
        
        datasets = []
        for path in glob(f'data/raw/{self.domain}/*'):
            
            with open(path, 'r') as f:

                total_lines = sum(1 for _ in f) - 1
                f.seek(0)
                reader = csv.DictReader(f)

                for row in tqdm(reader, total=total_lines, desc=f"Processing {path}"):

                    if self.domain == 'restaurant':
                        rating = row.get('rating_review', '')
                        title = row.get('title_review', '')
                        review = row.get('review_full', '')
                        url = row.get('url_restaurant', '')
                    else:
                        rating = row.get('rating', '')
                        title = row.get('title', '')
                        review = row.get('text', '')
                        url = row.get('hotel_url', '')
                        
                    if title: title =  ' '.join(title.split()).strip().lower()
                    if review: review = ' '.join(review.split()).strip().lower()

                    if type(title) == str and type(review) == str:
                        text = ' '.join([title, review])
                        lg = clf_lg.predict(text)[0][0][9:]
                        if lg == 'en':
                            datasets.append([url, int(rating.split('.')[0]), title, review])
        
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['url','rating','title','review'])
            writer.writerows(datasets)

    def output(self):
        return luigi.LocalTarget(f'data/review/{self.domain}.tsv')
    
if __name__ == '__main__':
    tasks = [
        TaskExtractReview(domain=domain) for domain in ["hotel", 'restaurant']
    ]
    luigi.build(tasks, local_scheduler=True)