import luigi
import os
import yaml

from task_topicgpt_extract_sample_generate_topic import TaskTopicGPTExtractSampleGenerateTopic


class TaskTopicGPTGenerateTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    is_default = luigi.BoolParameter(default=False)
    model = luigi.Parameter(default=None)

    def requires(self):
        return TaskTopicGPTExtractSampleGenerateTopic(self.domain)

    def run(self):
        from topicgpt_python import generate_topic_lvl1


        generate_topic_lvl1(
            "vllm",
            self.model,
            self.input().path,
            f"prompt/{self.domain if not self.is_default else 'default'}/generation.txt",
            f"prompt/{self.domain if not self.is_default else 'default'}/seed.md",
            self.output()["output"].path,
            self.output()["topic_output"].path,
            True
        )
        

    def output(self):
        model_path = self.model.split("/")[-1]
        base_dir = f'data/topicgpt/generation/{self.domain}/{model_path}'

        os.makedirs(base_dir, exist_ok=True)

        return {
            "output": luigi.LocalTarget(f'{base_dir}/generation{"_default" if self.is_default else ""}.jsonl'),
            "topic_output": luigi.LocalTarget(f'{base_dir}/generation{"_default" if self.is_default else ""}.md'),
        }
        
if __name__ == '__main__':
    tasks = [
        TaskTopicGPTGenerateTopic(domain=domain, model=model, is_dump_sample=True) for domain in ['restaurant', 'hotel'] for model in ["meta-llama/Llama-3.1-8B-Instruct"]
    ]
    luigi.build(tasks, local_scheduler=True)