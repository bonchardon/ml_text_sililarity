from asyncio import run

from core.preprocess import Preprocess
from core.visualization import Visualization
from core.text_similarity import TextSimilarity


async def main():
    text_analysis = Preprocess()
    result = await text_analysis.fully_preprocessed_data()
    await Visualization().words_visualisation()

    text_similarity_check = TextSimilarity(data_texts=result)
    response = await text_similarity_check.tf_idf_similarity_check()
    print(response)


if __name__ == '__main__':

    run(main())
