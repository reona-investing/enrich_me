from utils.paths import Paths
from utils.browser import BrowserUtils
from utils.timekeeper import timekeeper
from acquisition.features_updater.single_feature_scraper import SingleFeatureScraper
from acquisition.features_updater.scraped_data_merger import ScrapedDataMerger
import pandas as pd



class FeaturesUpdater:
    def __init__(self):
        self.scraper = SingleFeatureScraper(BrowserUtils())
        self.merger = ScrapedDataMerger()
    
    @timekeeper
    async def update_all(self):
        config_df = self._get_features_to_scrape_df()
        config_df = config_df[config_df['is_adopted']]
        features_num = len(config_df)

        for index, row in config_df.iterrows():
            print(f"{index + 1}/{features_num}: {row['Name']}")
            existing_df = pd.read_parquet(path = row['Path'])
            additional_df = await self.scraper.scrape_feature(
                investing_code = row['URL'],
                additional_scrape = row['AdditionalScrape'],
                additional_code = row['AdditionalCode']
                )
            df = self.merger.merge_scraped_data(existing_df = existing_df, additional_df = additional_df)
            df.to_parquet(row['Path'])
            print(df.tail(2))
            print('---------------------------------------')
            
        print('---------------------------------------')
        print('全データのスクレイピングが完了しました。')


    def _get_features_to_scrape_df(self) -> pd.DataFrame:
        features_to_scrape_df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
        features_to_scrape_df['Path'] = Paths.SCRAPED_DATA_FOLDER + '/' + \
                                        features_to_scrape_df['Group'] + '/' + \
                                        features_to_scrape_df['Path']
        return features_to_scrape_df    


if __name__ == '__main__':
    import asyncio
    fu = FeaturesUpdater()
    asyncio.run(fu.update_all())
