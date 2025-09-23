import os

import pandas as pd

from tqdm import tqdm

from omegaconf import DictConfig

from ..utils import SetUp


def pipeline(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    recommendation_manager = setup.get_manager(manager_type="recommendation")
    report_manager = setup.get_manager(manager_type="report")

    eval_data_path = os.path.join(
        config.data_path,
        config.eval_file_name,
    )
    eval_df = pd.read_csv(eval_data_path)

    results = []
    for _, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0]):
        question = row["question"]

        reranked_candidates = recommendation_manager.recommend(
            input_value=question,
        )

        if reranked_candidates:
            context = "\n".join([item["chunk"] for item in reranked_candidates])
            prompt = config.instruction.generator.rag.format(
                context=context,
                question=question,
            )
            answer = report_manager.generate(recommendations=prompt)
        else:
            context = "No relevant context found."
            answer = "I could not find relevant information to answer the question."

        results.append(
            {
                "question": question,
                "retrieved_context": context,
                "answer": answer,
            }
        )

    results_df = pd.DataFrame(results)
    result_path = os.path.join(
        config.data_path,
        config.result_file_name,
    )
    results_df.to_csv(
        result_path,
        index=False,
    )
    print(f"RAG pipeline finished. Results saved to {result_path}")
