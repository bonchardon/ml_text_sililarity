from asyncio import run

from core.preprocess import Preprocess


async def main():
    text_analysis = Preprocess()
    result = await text_analysis.preprocess()

    result_visualization = await text_analysis.words_visualisation()
    print(result)


if __name__ == '__main__':

    run(main())
