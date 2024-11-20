from typing import Any
from pandas.core.interchange.dataframe_protocol import DataFrame

from asyncio import run

from text_similarity_algos.preprocess import Preprocess
from text_similarity_algos.text_similarity_bert import TextSimilarityBert
from text_similarity_algos.text_similarity_tf_idf import TfIdfSimilarity


async def main() -> None:
    # here we can import and check various similarity techniques
    text_analysis: Preprocess = Preprocess()
    result: DataFrame = await text_analysis.fully_preprocessed_data()
    text_similarity_bert: TextSimilarityBert = TextSimilarityBert(data_texts=result['text'].to_list())
    text_similarity_tf_idf: TfIdfSimilarity = TfIdfSimilarity(data_texts=result['text'].to_list())
    response_bert: Any = await text_similarity_bert.bert_similarity()
    response_tf_idf: Any = await text_similarity_tf_idf.tf_idf_similarity_check()

    return response_bert


if __name__ == '__main__':
    run(main())
