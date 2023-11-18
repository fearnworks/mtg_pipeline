from typing import List
from loguru import logger 
from scrython.cards.cards_object import CardsObject
from scrython.cards.collector import Collector
import pandas as pd

class ScrythonInterface:
    def get_card(self, code: str, collector_number: str) -> CardsObject:
        code = code.lower()
        card = Collector(code=code, collector_number=collector_number)
        return card

    def log_card_summary(self, card: CardsObject): 
        logger.info(card.object())
        logger.info(card.name())
        logger.info(card.id())
        logger.info(card.oracle_text())
        
    def convert_to_df(self, cards: List[CardsObject]) -> pd.DataFrame:
        data = []
        for card in cards:
            data.append({
                'object': card.object(),
                'id': card.id(),
                # 'multiverse_ids': card.multiverse_ids(),
                # 'mtgo_id': card.mtgo_id(),
                # 'mtgo_foil_id': card.mtgo_foil_id(),
                # 'tcgplayer_id': card.tcgplayer_id(),
                # 'tcgplayer_etched_id': card.tcgplayer_etched_id(),
                'name': card.name(),
                'uri': card.uri(),
                'scryfall_uri': card.scryfall_uri(),
                'layout': card.layout(),
                'highres_image': card.highres_image(),
                'cmc': card.cmc(),
                'type_line': card.type_line(),
                'oracle_text': card.oracle_text(),
                'mana_cost': card.mana_cost(),
                'colors': card.colors(),
                'color_identity': card.color_identity(),
                'legalities': card.legalities(),
                'reserved': card.reserved(),
                'reprint': card.reprint(),
                'set_code': card.set_code(),
                'set_name': card.set_name(),
                'set_uri': card.set_uri(),
                'set_search_uri': card.set_search_uri(),
                'scryfall_set_uri': card.scryfall_set_uri(),
                'rulings_uri': card.rulings_uri(),
                'prints_search_uri': card.prints_search_uri(),
                'collector_number': card.collector_number(),
                'digital': card.digital(),
                'rarity': card.rarity(),
                'illustration_id': card.illustration_id(),
                'artist': card.artist(),
                'frame': card.frame(),
                'frame_effects': card.frame_effects(),
                'full_art': card.full_art(),
                'border_color': card.border_color(),
                'edhrec_rank': card.edhrec_rank(),
                'prices': card.prices(mode="usd"),
                'related_uris': card.related_uris(),
                'purchase_uris': card.purchase_uris(),
                # 'life_modifier': card.life_modifier(),
                # 'hand_modifier': card.hand_modifier(),
                # 'color_indicator': card.color_indicator(),
                # 'all_parts': card.all_parts(),
                # 'card_faces': card.card_faces(),
                # 'watermark': card.watermark(),
                'story_spotlight': card.story_spotlight(),
                'power': card.power(),
                'toughness': card.toughness(),
                # 'loyalty': card.loyalty(),
                # 'flavor_text': card.flavor_text(),
                'arena_id': card.arena_id(),
                'lang': card.lang(),
                # 'printed_name': card.printed_name(),
                # 'printed_type_line': card.printed_type_line(),
                # 'printed_text': card.printed_text(),
                'oracle_id': card.oracle_id(),
                'oversized': card.oversized(),
                'games': card.games(),
                'promo': card.promo(),
                'released_at': card.released_at(),
                'preview': card.preview(),
                'image_status': card.image_status(),
                'finishes': card.finishes()
            })
        return pd.DataFrame(data)

def default_test():
    test_code = "neo"
    collector_number="368"
    interface = ScrythonInterface()
    card = interface.get_card(code=test_code, collector_number=collector_number)
    interface.log_card_summary(card)
    df = interface.convert_to_df([card])
    df.to_excel("output/mtgcards.xlsx")
    logger.info(df)
    
default_test()