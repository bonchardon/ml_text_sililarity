from asyncio import run

from text_similarity_algos.preprocess import Preprocess
from text_similarity_algos.text_similarity import TextSimilarity


async def main() -> None:
    text_analysis = Preprocess()
    result = await text_analysis.fully_preprocessed_data()
    # await Visualization().words_visualisation()

    text_similarity_check = TextSimilarity(data_texts=result['text'].to_list())
    response = await text_similarity_check.bert_similarity()
    print(response)


if __name__ == '__main__':
    run(main())
