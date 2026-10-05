import luigi
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sentence_transformers import SentenceTransformer

from utils.senticse_embeddings import SenticseEmbeddings

PROMPT = {
    "intfloat/e5-base-v2": "query: ",
    "intfloat/e5-mistral-7b-instruct": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "Qwen/Qwen3-Embedding-8B": "Instruct: Retrieve semantically similar text.\nQuery: "
}


class TaskEvalCluster(luigi.Task):

    sbert = luigi.Parameter(default="intfloat/e5-base-v2")
    file = luigi.Parameter(default="data/cluster_eval/results.json")

    # Dataset utilisé pour calculer les métriques
    metric_file = luigi.Parameter(default="data/cluster/data.csv")

    # Dataset utilisé pour la visualisation
    viz_file = luigi.Parameter(default="data/cluster/cluster.tsv")

    def run(self):

        # ============================================================
        # 1. Charger le modèle d'embeddings
        # ============================================================

        print(f"\nLoading model: {self.sbert}")

        if "SentiCSE" in self.sbert:
            sbert = SenticseEmbeddings(self.sbert)
        else:
            sbert = SentenceTransformer(self.sbert)

        # ============================================================
        # 2. Charger les DEUX datasets
        # ============================================================

        # Dataset pour les métriques
        df_metric = pd.read_csv(self.metric_file)

        # Dataset pour la visualisation
        df_viz = pd.read_csv(self.viz_file, sep="\t")

        print(f"\nMetric dataset: {len(df_metric)} samples")
        print(f"Visualization dataset: {len(df_viz)} samples")

        # Vérification des colonnes nécessaires
        required_columns = ["sentence", "topic", "sentiment"]

        for column in required_columns:
            if column not in df_metric.columns:
                raise ValueError(
                    f"Column '{column}' missing from metric dataset "
                    f"({self.metric_file})"
                )

            if column not in df_viz.columns:
                raise ValueError(
                    f"Column '{column}' missing from visualization dataset "
                    f"({self.viz_file})"
                )

        # ============================================================
        # 3. ÉCHANTILLON MÉTRIQUES
        # ============================================================

        sentences_metric = df_metric["sentence"].tolist()
        topics_metric = df_metric["topic"].tolist()
        sentiments_metric = df_metric["sentiment"].tolist()

        print("\nEncoding metric dataset...")

        embeddings_metric = sbert.encode(
            [PROMPT.get(self.sbert, "") + s for s in sentences_metric],
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Calcul des métriques UNIQUEMENT sur cet échantillon
        metrics = self._compute_metrics(
            embeddings_metric,
            topics_metric,
            sentiments_metric,
        )

        # ============================================================
        # 4. ÉCHANTILLON VISUALISATION
        # ============================================================

        sentences_viz = df_viz["sentence"].tolist()
        topics_viz = df_viz["topic"].tolist()
        sentiments_viz = df_viz["sentiment"].tolist()

        print("\nEncoding visualization dataset...")

        embeddings_viz = sbert.encode(
            [PROMPT.get(self.sbert, "") + s for s in sentences_viz],
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Réduction en 2D UNIQUEMENT sur l'échantillon de visualisation
        print("\nReducing visualization embeddings with UMAP...")

        emb_2d = self._reduce_embeddings(embeddings_viz)

        # Création de la visualisation
        self._create_visualization(
            emb_2d,
            topics_viz,
            sentiments_viz,
            metrics,
        )

        # ============================================================
        # 5. Sauvegarder les métriques
        # ============================================================

        with open(self.output()["metric"].path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMetrics saved to {self.output()['metric'].path}")


    def _reduce_embeddings(self, emb):
        """
        Reduce embeddings to 2D using UMAP.
        Cette opération est effectuée uniquement sur
        l'échantillon destiné à la visualisation.
        """

        umap = UMAP(
            n_components=2,
            min_dist=0.25,
            n_neighbors=15,
            metric="cosine",
            random_state=42,
        )

        emb_2d = umap.fit_transform(emb)

        # Normalisation des coordonnées entre 0 et 1
        scaler = MinMaxScaler()
        emb_2d_norm = scaler.fit_transform(emb_2d)

        return emb_2d_norm


    def _compute_metrics(self, emb, topics, sentiments):
        """
        Compute clustering evaluation metrics.

        IMPORTANT:
        Ces métriques sont calculées uniquement sur l'échantillon
        fourni dans metric_file.
        """

        print(f"\nComputing metrics for: {self.sbert}")

        results = {
            "model": self.sbert,
            "n_samples": len(emb),
            "sentiment_label": {},
            "topic_label": {},
            "combined_label": {},
        }

        # ============================================================
        # SENTIMENT
        # ============================================================

        print("\n--- Computing metrics for SENTIMENT label ---")

        results["sentiment_label"] = self._compute_label_metrics(
            emb,
            sentiments,
            "Sentiment",
        )

        # ============================================================
        # TOPIC
        # ============================================================

        print("\n--- Computing metrics for TOPIC label ---")

        results["topic_label"] = self._compute_label_metrics(
            emb,
            topics,
            "Topic",
        )

        # ============================================================
        # COMBINED
        # ============================================================

        print(
            "\n--- Computing metrics for COMBINED "
            "(Sentiment + Topic) label ---"
        )

        combined_labels = [
            f"{topic}_{sentiment}"
            for topic, sentiment in zip(topics, sentiments)
        ]

        results["combined_label"] = self._compute_label_metrics(
            emb,
            combined_labels,
            "Combined",
        )

        return results


    def _compute_label_metrics(self, emb, labels, label_type):
        """
        Compute clustering metrics for a specific label type.
        """

        try:

            # Vérification du nombre de classes
            unique_labels = np.unique(labels)

            if len(unique_labels) < 2:
                raise ValueError(
                    f"At least 2 classes are required, "
                    f"but only {len(unique_labels)} class was found."
                )

            sil = silhouette_score(
                emb,
                labels,
                metric="cosine",
            )

            cal = calinski_harabasz_score(
                emb,
                labels,
            )

            db = davies_bouldin_score(
                emb,
                labels,
            )

            metrics = {
                "n_classes": len(unique_labels),
                "silhouette": round(sil, 3),
                "calinski_harabasz": round(cal, 2),
                "davies_bouldin": round(db, 4),
            }

            print(
                f"{label_type} - "
                f"Silhouette Score: {metrics['silhouette']}"
            )

            print(
                f"{label_type} - "
                f"Calinski-Harabasz Score: "
                f"{metrics['calinski_harabasz']}"
            )

            print(
                f"{label_type} - "
                f"Davies-Bouldin Score: "
                f"{metrics['davies_bouldin']}"
            )

            return metrics

        except Exception as e:

            print(
                f"Error computing metrics for "
                f"{label_type}: {e}"
            )

            return {
                "n_classes": None,
                "silhouette": None,
                "calinski_harabasz": None,
                "davies_bouldin": None,
                "error": str(e),
            }


    def _create_visualization(
        self,
        emb_2d,
        topics,
        sentiments,
        metrics,
    ):
        """
        Create and save scatter plot.

        IMPORTANT:
        La visualisation utilise uniquement l'échantillon
        destiné à la visualisation.
        """

        # ============================================================
        # Marker shape per sentiment
        # ============================================================

        sentiment_markers = {
            "negative": "X",
            "positive": "o",
        }

        # ============================================================
        # Color per topic + sentiment
        # ============================================================

        topic_colors = {

            # ROOM
            "ROOM_positive": "#005AB5",
            "ROOM_negative": "#56B4E9",

            # STAFF
            "STAFF_positive": "#D55E00",
            "STAFF_negative": "#E69F00",

            # FOOD
            "FOOD_positive": "#007A5E",
            "FOOD_negative": "#66C2A5",
        }

        # ============================================================
        # Figure
        # ============================================================

        plt.figure(figsize=(6, 6))

        combinations = sorted(
            set(zip(topics, sentiments))
        )

        for topic, sentiment in combinations:

            idx = [
                i
                for i, (t, s) in enumerate(
                    zip(topics, sentiments)
                )
                if t == topic and s == sentiment
            ]

            if not idx:
                continue

            x = emb_2d[idx, 0]
            y = emb_2d[idx, 1]

            marker = sentiment_markers.get(
                sentiment,
                "o",
            )

            plt.scatter(
                x,
                y,
                color=topic_colors.get(
                    f"{topic}_{sentiment}",
                    "gray",
                ),
                marker=marker,
                label=f"{topic}-{sentiment}",
                edgecolor="black",
                s=120 if marker == "o" else 160,
                alpha=0.8,
                linewidth=2,
            )

        # ============================================================
        # Mise en forme
        # ============================================================

        plt.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        # plt.legend(
        #     bbox_to_anchor=(1.05, 1),
        #     loc="upper left",
        # )

        plt.tight_layout()

        # ============================================================
        # Save
        # ============================================================

        plt.savefig(
            self.output()["fig"].path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Figure saved to "
            f"{self.output()['fig'].path}"
        )


    def output(self):

        _dir = (
            f"data/cluster_eval/"
            f"{self.sbert.split('/')[-1]}"
        )

        os.makedirs(
            _dir,
            exist_ok=True,
        )

        return {
            "metric": luigi.LocalTarget(
                f"{_dir}/metric.json"
            ),
            "fig": luigi.LocalTarget(
                f"{_dir}/fig.png"
            ),
        }


if __name__ == "__main__":

   if __name__ == '__main__':
    from glob import glob
    tasks = []

    models = []

    models.extend([
        "intfloat/e5-base-v2",
        "intfloat/e5-mistral-7b-instruct",
        "google/embeddinggemma-300m",
        "all-mpnet-base-v2",
        "Qwen/Qwen3-Embedding-8B",
        "vahidthegreat/StanceAware-SBERT",
        "DILAB-HYU/SentiCSE",
        "Alibaba-NLP/gte-modernbert-base",
        "activebus/BERT_Review",
        "veroman/TourBERT",
    ])
    
    # for m in glob("data/sbert/*"):
    #     models.append(m)

    for model in models:
        tasks.append(TaskEvalCluster(
            sbert=model, 
        ))


    luigi.build(
        tasks,
        local_scheduler=True,
    )