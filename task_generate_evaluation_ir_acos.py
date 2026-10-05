import luigi
from collections import defaultdict
import json
from gensim.utils import simple_preprocess


class TaskGenerateEvaluationIRACOS(luigi.Task):
    """Generate evaluation data for IR evaluation on ACOS dataset."""

    domain = luigi.ChoiceParameter(choices=["restaurant", "laptop"], default='restaurant')

    # Define subset parameter with choices
    subset = luigi.ChoiceParameter(
        choices=['ACSA', 'ACD', 'ACSA-Neutral', 'ASC', 'ASC-Neutral'],
        description='Subset type for evaluation: ACSA, ACD, ASC, or Neutral'
    )

    def is_valid(self, opinions):
        """Validate that opinions have consistent sentiment and are not neutral/conflict."""
        if self.subset == 'ACD': 
            return True
        
        categories_sentiments = defaultdict(set)
        for opinion in opinions:
            opinion = opinion.split(" ")
            cat = opinion[1].split("#")[0]
            if cat in  ["Out_Of_Scope", "LAPTOP", "RESTAURANT"]:
                return False
            # if cat in ["RESTAURANT"]:  
            #     continue
            pol = opinion[2]
            if str(pol) == '1' and 'Neutral' not in self.subset: 
                return False
            categories_sentiments[cat].add(pol)

        for cat, s in categories_sentiments.items():
            if len(s) > 1: 
                return False
        return True
        
    def run(self):
        """Process hotel OATS data and generate query-document mappings."""

        query_hotel = defaultdict(set)
        sentence_hotel = defaultdict(set)

        # Process training, test, and dev splits
        for split in ["train", "test", "dev"]:
            if self.domain == 'restaurant':
                file_path = f"data/absa/restaurant/acos/rest16_quad_{split}.tsv"
            else:
                file_path = f"data/absa/laptop/acos/laptop_quad_{split}.tsv"
            try:
                with open(file_path, "r") as f:
                    sentences = f.readlines()
                    for sentence in sentences:
                        line = sentence.split('\t')
                        text = line[0].lower().strip()
                        opinions = line[1:]
                        
                        if not self.is_valid(opinions):
                            continue

                        tokens = text.split()
                        for opinion in opinions:
                            opinion = opinion.split(" ")
                            aspect = opinion[0]
                            if aspect == "-1,-1": 
                                aspect = "NULL"
                            else:
                                asp_s = int(aspect.split(",")[0])
                                asp_e = int(aspect.split(",")[1])
                                aspect = ' '.join(tokens[asp_s:asp_e])

                            category = opinion[1].split("#")[0]
                            if category in ["RESTAURANT", "LAPTOP"]:  
                                continue

                            pol = opinion[2]

                            adj = opinion[3]
                            if adj == "-1,-1": 
                                adj = "NULL"
                            else:
                                adj_s = int(adj.split(",")[0])
                                adj_e = int(adj.split(",")[1])
                                adj = ' '.join(tokens[adj_s:adj_e])

                            key = None
                            if self.subset == 'ACD':
                                key = category
                                sentence_hotel[key].add(" ".join(simple_preprocess(text)))
                            elif 'ASC' in self.subset:
                                key = pol
                                sentence_hotel[key].add(" ".join(simple_preprocess(text)))
                            elif 'ACSA' in self.subset:
                                key = f"{category}-{pol}"
                                sentence_hotel[key].add(" ".join(simple_preprocess(text)))
                            if aspect != "NULL" and adj != "NULL":

                                query = " ".join(simple_preprocess(f"{aspect} {adj}"))
                                if len(query.split()) > 1:
                                    query_hotel[key].add(query)
                        
                         

            except FileNotFoundError:
                self.logger.warning(f"File not found: {file_path}")
                continue

        # Build query ID -> query text mapping with consistent indexing
        query_id = dict()
        query_idx = 0
        
        for topic, queries_set in sorted(query_hotel.items()):
            for query_text in sorted(queries_set):
                query_key = f"{topic}-{query_idx}"
                query_id[query_key] = query_text
                query_idx += 1

        # FIX: Deduplicate sentences across topics
        # Map sentence text to doc_id to avoid duplicates
        sentence_to_doc_id = dict()
        doc_id = dict()
        topic_doc_ids = defaultdict(list)

        for topic, sentences in sentence_hotel.items():
            for sentence in sentences:
                if sentence not in sentence_to_doc_id:
                    # New sentence, assign a new doc_id
                    current_doc_id = str(len(doc_id))
                    doc_id[current_doc_id] = sentence
                    sentence_to_doc_id[sentence] = current_doc_id
                else:
                    # Sentence already exists, reuse its doc_id
                    current_doc_id = sentence_to_doc_id[sentence]
                
                # Add doc_id to topic mapping (avoid duplicates in the list)
                if current_doc_id not in topic_doc_ids[topic]:
                    topic_doc_ids[topic].append(current_doc_id)

        # Build query ID -> document IDs mapping (consistent with deduplicated docs)
        query_docs_map = dict()
        query_idx = 0

        for topic, queries in sorted(query_hotel.items()):
            for _ in sorted(queries):
                query_key = f"{topic}-{query_idx}"
                query_docs_map[query_key] = topic_doc_ids[topic]
                query_idx += 1

        # Write output files
        output_paths = self.output()
        
        with output_paths["query"].open("w") as f:
            json.dump(query_id, f, indent=2)
        
        with output_paths["doc"].open("w") as f:
            json.dump(doc_id, f, indent=2)
        
        with output_paths["map"].open("w") as f:
            json.dump(query_docs_map, f, indent=2)

    def output(self):
        """Define output targets for query, document, and mapping files."""
        base_path = f'data/ir/{self.domain}/{self.subset}'
        return {
            "query": luigi.LocalTarget(f'{base_path}_query.json'),
            "doc": luigi.LocalTarget(f'{base_path}_doc.json'),
            "map": luigi.LocalTarget(f'{base_path}_map.json')
        }


if __name__ == '__main__':

    tasks = []
    for subset in ['ACSA']:
        for domain in ['restaurant', 'laptop']:
            tasks.append(
                TaskGenerateEvaluationIRACOS(subset=subset, domain=domain)
            )
    luigi.build(tasks, local_scheduler=True)