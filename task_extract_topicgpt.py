import luigi
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm
import csv

from task_topicgpt_assignment_topic import TaskTopicGPTAssignmentTopic
from task_topicgpt_generate_topic import TaskTopicGPTGenerateTopic

topic_regex = re.compile(r'(?<=\]\s)([A-Za-z\s]+)(?=:)')

class TaskExtractTopicGPT(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    generation = luigi.ChoiceParameter(choices=[0,1,2,3], var_type=int)
    # 1 -> default seed
    # 2 -> generation
    # 3 -> default prompt
    model = luigi.Parameter(default=None)

    def requires(self):
        d = dict()
        if self.generation >= 2:
            d["topic"] = TaskTopicGPTGenerateTopic(domain=self.domain, model=self.model, is_default=True if self.generation == 3 else False)
        d["sample"] = TaskTopicGPTAssignmentTopic(domain=self.domain, model=self.model, generation=self.generation)
        return d

        
    def run(self):

        allowed_topics = set()

        # Determine which file to read
        if self.generation in [2, 3]:
            file_path = self.input()["topic"]["topic_output"].path
        else:
            file_path = f"prompt/{self.domain}/seed_count.md"

        # Single reading block
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Reading allowed topics"):
                match = re.match(r'\[1\]\s([A-Za-z\s]+)\s\(Count:', line)
                if match:
                    allowed_topics.add(match.group(1).strip())
        
        data = defaultdict(list)
        # read jsonl
        with self.input()["sample"].open('r') as f:
            for line in tqdm(f, desc="Processing assignment entries"):
                entry = json.loads(line.strip())
                _id = entry.get("id")
                responses = entry.get("responses", "")
                text = entry.get("text", "")
                rating = _id.split("-")[-1]
                
                # Extract all topics in this entry
                topics = topic_regex.findall(responses)
                
                for topic in topics:
                    t = topic.strip()
                    if t in allowed_topics:
                        data[topic.strip()+"-"+rating].append(text)
                    else:
                        print(f"Not allowed : {t}")       
        
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['review', 'rating'])  # header
            
            for topic_rating, texts in data.items():
                for text in texts:
                    writer.writerow([text, topic_rating])
        

    def output(self):
        model_path = self.model.split("/")[-1]
        folder_path = f'data/topicgpt/extract/{self.domain}/{model_path}'

        # Crée le dossier s'il n'existe pas déjà
        os.makedirs(folder_path, exist_ok=True)

        # Map generation types to readable names
        gen_type = {
            0: "eval",
            1: "seed",
            2: "generated",
            3: "default"
        }[self.generation]

        return luigi.LocalTarget(f'{folder_path}/extract_{gen_type}.tsv')


if __name__ == "__main__":
    import csv

    tasks = []
    
    with open("experiments_topicgpt.tsv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        
        for params in reader:
            model = params["model"].strip()
            domain = params["domain"].strip()
            generation = int(params["generation"].strip())
            
            tasks.append(
                TaskExtractTopicGPT(
                    model=model,
                    domain=domain,
                    generation=generation,
                )
            )

    luigi.build(
        tasks,
        local_scheduler=True,
        workers=1,
    )