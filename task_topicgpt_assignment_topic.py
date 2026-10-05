import luigi
import os
from task_topicgpt_extract_sample_assignment_topic import TaskTopicGPTExtractSampleAssignmentTopic
from task_topicgpt_generate_topic import TaskTopicGPTGenerateTopic

class TaskTopicGPTAssignmentTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    generation = luigi.ChoiceParameter(choices=[0,1,2,3], var_type=int)

    # 0 -> eval
    # 1 -> default seed
    # 2 -> generation
    # 3 -> default prompt
    model = luigi.Parameter(default=None)

    def requires(self):
        d = dict()
        if self.generation >= 2:
            d["topic"] = TaskTopicGPTGenerateTopic(domain=self.domain, model=self.model, is_default=True if self.generation == 3 else False)
        d["sample"] = TaskTopicGPTExtractSampleAssignmentTopic(domain=self.domain, eval=True if self.generation == 0 else False)
        return d

    def run(self):
        from topicgpt_python import assign_topics


        if self.generation < 2:
            assign_topics(
                "vllm",
                self.model,
                self.input()['sample'].path,
                f"prompt/{self.domain}/assignment.txt",
                self.output().path,
                f"prompt/{self.domain}/seed_count.md",
                True
            )

        if self.generation == 2:
            assign_topics(
                "vllm",
                self.model,
                self.input()['sample'].path,
                f"prompt/{self.domain}/assignment.txt",
                self.output().path,
                self.input()['topic']['topic_output'].path,
                True
            )

        if self.generation == 3:
            assign_topics(
                "vllm",
                self.model,
                self.input()['sample'].path,
                f"prompt/default/assignment.txt",
                self.output().path,
                self.input()['topic']['topic_output'].path,
                True
            )

    def output(self):
        model_path = self.model.split("/")[-1]
        folder_path = f'data/topicgpt/assignment/{self.domain}/{model_path}'

        # Crée le dossier s'il n'existe pas déjà
        os.makedirs(folder_path, exist_ok=True)

        # Map generation types to readable names
        gen_type = {
            0: "eval",
            1: "seed",
            2: "generated",
            3: "default"
        }[self.generation]

        return luigi.LocalTarget(f'{folder_path}/assignment_{gen_type}.jsonl')
        
    

if __name__ == "__main__":
    import sys
    import csv

    tsv_file = sys.argv[1]
    task_index = int(sys.argv[2])  # zero-based index in TSV, excluding header

    with open("experiments_topicgpt.tsv", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter="\t"))
        params = reader[task_index]

    model = params["model"].strip()
    domain = params["domain"].strip()
    generation = int(params["generation"].strip())
    

    luigi.build(
        [
            TaskTopicGPTAssignmentTopic(
                model=model,
                domain=domain,
                generation=generation,
            )
        ],
        local_scheduler=True,
        workers=1,
    )