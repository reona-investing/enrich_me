from utils.paths import Paths
from utils.browser import BrowserUtils
from utils.browser.browser_manager import BrowserManager
from utils.timekeeper import timekeeper
from acquisition.features_updater.single_feature_scraper import SingleFeatureScraper
from acquisition.features_updater.scraped_data_merger import ScrapedDataMerger
import pandas as pd
import asyncio


class FeaturesUpdater:
    def __init__(self):
        self.count = 1
    
    
    @timekeeper
    async def update_all(self):
        config_df = self._get_features_to_scrape_df()
        config_df = config_df[config_df['is_adopted']]
        features_num = len(config_df)

        semaphore = asyncio.Semaphore(5)
        browser_manager = BrowserManager()
        scraper = SingleFeatureScraper(browser_manager=browser_manager)
        merger = ScrapedDataMerger()
        tasks = [self._process_feature(scraper, merger, semaphore, row, features_num) for _, row in config_df.iterrows()]
        await asyncio.gather(*tasks)
        #await browser_manager.reset_session()
        
        print('---------------------------------------')
        print('全データのスクレイピングが完了しました。')


    def _get_features_to_scrape_df(self) -> pd.DataFrame:
        features_to_scrape_df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
        features_to_scrape_df['Path'] = Paths.SCRAPED_DATA_FOLDER + '/' + \
                                        features_to_scrape_df['Group'] + '/' + \
                                        features_to_scrape_df['Path']
        return features_to_scrape_df 
    

    async def _process_feature(self, scraper: SingleFeatureScraper, merger: ScrapedDataMerger, semaphore: asyncio.Semaphore, 
                               row, features_num):
        async with semaphore:
            existing_df = pd.read_parquet(path=row['Path'])
            await asyncio.sleep(1)
            additional_df = await scraper.scrape_feature(
                investing_code=row['URL'],
                additional_scrape=row['AdditionalScrape'],
                additional_code=row['AdditionalCode']
            )
            df = merger.merge_scraped_data(existing_df=existing_df, additional_df=additional_df)
            df.to_parquet(row['Path'])
            print(f"{self.count}/{features_num}: {row['Name']}")
            self.count += 1
            print(df.tail(2))
            print('---------------------------------------')



if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(FeaturesUpdater().update_all())