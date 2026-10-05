import os
import csv
import random
from collections import defaultdict
from tqdm import tqdm
from nltk import sent_tokenize
from gensim.utils import simple_preprocess

import luigi
from task_extract_review import TaskExtractReview


class TaskExtractSample(luigi.Task):

    domain = luigi.ChoiceParameter(
        choices=["restaurant", "hotel"]
    )

    sample_train = luigi.IntParameter(
        default=1_000_000
    )

    def requires(self):
        return TaskExtractReview(self.domain)

    def output(self):
        return luigi.LocalTarget(
            f"data/sample/{self.domain}_{self.sample_train}.tsv"
        )

    def run(self):
        # Crée le dossier de sortie
        output_path = self.output().path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Lecture des données
        data = self._read_tsv(self.input().path)

        # Vérification des colonnes
        if not data:
            raise ValueError("Aucune donnée trouvée dans le fichier")

        required_columns = {"review", "rating"}
        missing_columns = required_columns - set(data[0].keys())

        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes : {missing_columns}"
            )

        # Supprime les reviews vides
        data = [
            row for row in data
            if row.get("review") and row.get("rating")
        ]

        if not data:
            raise ValueError("Aucune donnée valide après filtrage")

        # Nombre de ratings différents
        unique_ratings = set(row["rating"] for row in data)
        n_ratings = len(unique_ratings)

        if n_ratings == 0:
            raise ValueError("Aucun rating valide trouvé.")

        # Taille maximale par rating
        size_per_rating = self.sample_train // n_ratings

        # Échantillonnage équilibré par rating
        data_by_rating = defaultdict(list)
        for row in data:
            data_by_rating[row["rating"]].append(row)

        sampled_data = []
        for rating, rows in data_by_rating.items():
            sample_size = min(len(rows), size_per_rating)
            sampled = random.sample(rows, k=sample_size)
            sampled_data.extend(sampled)

        reviews_filter = []
        ratings_filter = []

        # Traitement des reviews
        for row in tqdm(
            sampled_data,
            desc="Processing reviews"
        ):
            review = row["review"]
            rating = row["rating"]

            # Tokenisation des phrases AVANT le preprocessing
            try:
                sentences = sent_tokenize(str(review))
            except LookupError:
                raise RuntimeError(
                    "Ressource NLTK manquante. "
                    "Installe-la avec : "
                    "python -m nltk.downloader punkt punkt_tab"
                )

            for sentence in sentences:
                # Nettoyage de la phrase
                words = simple_preprocess(
                    sentence,
                    deacc=False
                )

                # Recrée une phrase propre
                clean_sentence = " ".join(words)

                n_words = len(words)

                # Garde uniquement les phrases de 5 à 300 mots
                if 5 <= n_words <= 300:
                    reviews_filter.append(clean_sentence)
                    ratings_filter.append(rating)

        # Sauvegarde
        self._write_tsv(
            output_path,
            reviews_filter,
            ratings_filter
        )

    def _read_tsv(self, filepath):
        """Lit un fichier TSV et retourne une liste de dictionnaires"""
        data = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    data.append(row)
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la lecture du fichier : {e}")
        return data

    def _write_tsv(self, filepath, reviews, ratings):
        """Écrit les données dans un fichier TSV"""
        try:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["review", "rating"])
                for review, rating in zip(reviews, ratings):
                    writer.writerow([review, rating])
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'écriture du fichier : {e}")


if __name__ == "__main__":

    tasks = [
        TaskExtractSample(domain=domain)
        for domain in ["hotel", "restaurant"]
    ]

    luigi.build(
        tasks,
        local_scheduler=True
    )