"""Coverage matrix for RegexEntityExtractor:
- tld dotted/context/multi/exclude -> test_tld_*
- price under/over/between/currency-word -> test_price_*
- auction include/exclude              -> test_auction_*
- name length letters/chars/exact      -> test_name_length_*
- bids / age (older-newer inversion)   -> test_bids_*, test_age_*
- keyword prefix/suffix/contains       -> test_keyword_*
- char booleans hyphen/number/idn      -> test_char_*
- time remaining relative              -> test_time_remaining_*
- robustness empty/no-signal/bare-num  -> test_*_no_slot / classify contract
"""
from typing import Optional

import pytest

from semantic_search.config.loader import load_config
from semantic_search.config.models import AgentSearchConfig, QIL0RegexEntityConfig
from semantic_search.qi.regex_entity_extractor import RegexEntityExtractor

_EXTRACTOR: Optional[RegexEntityExtractor] = None


def _extractor(enabled: bool = True) -> RegexEntityExtractor:
    """Build RegexEntityExtractor from qi.entity_slots + l0_regex_entity YAML (no hardcoded slots)."""
    global _EXTRACTOR
    if enabled and _EXTRACTOR is not None:
        return _EXTRACTOR
    conf = AgentSearchConfig.from_dict(load_config())
    slots = conf.qi.entity_slots
    assert slots is not None, "qi.entity_slots required in base.yaml for regex tests"
    regex_cfg = conf.qi.l0_regex_entity
    assert regex_cfg is not None, "qi.l0_regex_entity required in base.yaml for regex tests"
    cfg = QIL0RegexEntityConfig(
        enabled=enabled,
        max_entities=regex_cfg.max_entities,
        confidence=regex_cfg.confidence,
        source_tag=regex_cfg.source_tag,
        fallback_only_when_llm_unavailable=regex_cfg.fallback_only_when_llm_unavailable,
    )
    ext = RegexEntityExtractor(
        cfg,
        hard_entity_names=slots.hard_entity_set,
        soft_slot_names=slots.soft_slot_set,
        known_tlds=frozenset(conf.qi.regex.known_tlds),
    )
    if enabled:
        _EXTRACTOR = ext
    return ext


def _slots(query: str) -> dict:
    return _extractor()._extract_slots(query)


def test_tld_dotted_and_price_currency_shared_query() -> None:
    s = _slots("search .net domains under $50")
    assert s['tld'] == ['net']
    assert s['price_max'] == 49.0  # under N → exclusive N-1 (L0 rule 2)
    assert 'price_min' not in s


def test_budget_prefix_in_dollar_price_max() -> None:
    s = _slots("find .net domains in $50")
    assert s['tld'] == ['net']
    assert s['price_max'] == 50
    assert 'price_min' not in s


def test_budget_prefix_around_band() -> None:
    s = _slots("domains around $75")
    assert s['price_min'] == 75
    assert s['price_max'] == 75


def test_budget_prefix_bare_in_without_dollar_skipped() -> None:
    s = _slots("domains in 50")
    assert 'price_max' not in s


def test_tld_multi_dotted() -> None:
    s = _slots("show .com or .io domains")
    assert set(s['tld']) == {'com', 'io'}


def test_tld_context_ending_in() -> None:
    assert _slots("names ending in io")['tld'] == ['io']


def test_tld_exclude_not_dotted() -> None:
    s = _slots("domains but not .net")
    assert s['tldExcludeList'] == ['net']
    assert 'tld' not in s or 'net' not in s.get('tld', [])


def test_price_over_sets_min() -> None:
    assert _slots("domains over $500")['price_min'] == 501.0


def test_price_currency_word() -> None:
    assert _slots("under 200 usd")['price_max'] == 199.0


def test_price_between_sets_both() -> None:
    s = _slots("priced between $100 and $500")
    assert s['price_min'] == 100.0 and s['price_max'] == 500.0


def test_price_leading_metric_noun() -> None:
    # under/below/less than → exclusive N-1 at all scales (L0 rule 2).
    assert _slots("price under 50")['price_max'] == 49.0
    assert _slots("price under 1k")['price_max'] == 999.0
    assert _slots("price under 2k")['price_max'] == 1999.0


def test_price_bare_under_defaults_to_price_max() -> None:
    """Comparator+number with no $ / unit → price (domain-auction convention)."""
    assert _slots("under 500")['price_max'] == 499.0
    assert _slots("over 100")['price_min'] == 101.0


def test_price_between_bare_defaults_to_price() -> None:
    s = _slots("between 100 and 500")
    assert s['price_min'] == 100.0 and s['price_max'] == 500.0


def test_buy_now_under_sets_buy_it_now_and_price_max() -> None:
    """Buy-now cue → buy_it_now bool; bare under N → price_max (not typeIncludeList / BIN price)."""
    s = _slots("buy now .com under 500")
    assert s['buy_it_now'] is True
    assert s['tld'] == ['com']
    assert s['price_max'] == 499.0
    assert 'auction_type' not in s
    assert 'buy_it_now_max' not in s


def test_buy_it_now_under_sets_price_max_not_bin_price() -> None:
    s = _slots("buy it now under 200")
    assert s['buy_it_now'] is True
    assert s['price_max'] == 199.0
    assert 'buy_it_now_max' not in s
    assert 'auction_type' not in s


def test_buy_it_now_price_metric_still_maps_to_bin_price() -> None:
    s = _slots("buy it now price under 200")
    assert s['buy_it_now'] is True
    assert s['buy_it_now_max'] == 200.0


def test_auction_include() -> None:
    # Prod _AUCTION_RE / keyword map: expiry + closeout + premium + backorder + partner/godaddy (no buynow).
    assert _slots("expiring domains")['auction_type'] == ['expiry']
    assert _slots("closeout auctions")['auction_type'] == ['closeout']
    assert _slots("premium auction")['auction_type'] == ['premium']
    assert _slots("backorder format")['auction_type'] == ['backorder']
    assert _slots("drop catch")['auction_type'] == ['dropcatch']
    assert _slots("firehose")['auction_type'] == ['firehose']
    partner = _slots("partner auction .com under 2k")
    assert partner['auction_type'] == ['partner']
    # Leading token before dotted TLD may also land in keyword_contains ("auction .com").
    assert _slots("godaddy auction")['auction_type'] == ['godaddy']
    assert 'auction_type' not in _slots("buy now listings")


def test_auction_type_numeric_id_kept_as_id() -> None:
    s = _slots("find .com domains for auction type 16 under $50")
    assert s['auction_type'] == ['16']
    assert s['tld'] == ['com']
    assert s['price_max'] == 49.0
    assert 'keyword_contains' not in s


def test_find_before_dotted_tld_not_keyword_contains() -> None:
    s = _slots("find .com domains under 100")
    assert s.get('tld') == ['com']
    assert 'keyword_contains' not in s


def test_auction_exclude() -> None:
    s = _slots("domains but not closeout")
    assert s['typeExcludeList'] == ['closeout']
    assert 'auction_type' not in s
    s2 = _slots("exclude backorder want direct")
    assert s2['typeExcludeList'] == ['backorder']
    assert 'auction_type' not in s2
    assert 'keyword_contains_exclude' not in s2
    assert 'keyword_contains' not in s2
    s3 = _slots("exclude partner auctions")
    assert s3['typeExcludeList'] == ['partner']
    assert 'keyword_contains_exclude' not in s3


def test_name_length_exact_letters() -> None:
    s = _slots("5 letter domains")
    assert s['name_length_min'] == 5 and s['name_length_max'] == 5


def test_name_length_max_characters() -> None:
    # LLM: exclusive under N chars → maxSldLen=N-1
    assert _slots("under 8 characters")['name_length_max'] == 7


def test_name_length_min_at_least() -> None:
    assert _slots("at least 3 characters")['name_length_min'] == 3


def test_min_letters_not_name_length() -> None:
    # LLM: 'minimum N letters' → minLetters ONLY
    s = _slots("minimum 6 letters")
    assert s['minLetters'] == 6
    assert 'name_length_min' not in s


def test_bids_min_requires_comparator() -> None:
    assert _slots("at least 5 bids")['bids_min'] == 5


def test_bids_exclusive_more_than() -> None:
    # LLM: 'more than 15 bids' → minBids=16
    assert _slots("more than 15 bids")['bids_min'] == 16


def test_bids_exclusive_fewer_than() -> None:
    # LLM: 'fewer than 5 bids' → maxBids=4
    assert _slots("fewer than 5 bids")['bids_max'] == 4


def test_exclude_letters() -> None:
    assert _slots("numbers only no letters")['excludeLetters'] is True
    assert 'tldExcludeList' not in _slots("numbers only no letters")


def test_no_hyphens_not_tld_exclude() -> None:
    s = _slots("eur under 800 no hyphens")
    assert s['has_hyphen'] is False
    assert 'hyphens' not in s.get('tldExcludeList', [])


def test_owner_member_include() -> None:
    assert _slots("from seller acme123")['ownerMemberIncludeList'] == 'acme123'


def test_stale_days_listed_min() -> None:
    assert _slots("stale over 2 weeks old")['days_listed_min'] == 14
    assert 'price_min' not in _slots("stale over 2 weeks old")


def test_time_remaining_next_hours() -> None:
    assert _slots("closing in the next 6 hours")['time_remaining_max'] == 21600


def test_unique_searches_min() -> None:
    assert _slots("min 1000 unique searches")['minUniqueSearches'] == 1000


def test_estibot_domain_count() -> None:
    # LLM exclusive: above N → N+1
    assert _slots("estibot domain count above 50")['minEstibotDomainCount'] == 51
    assert 'price_min' not in _slots("estibot domain count above 50")


def test_unknown_age_under_is_price() -> None:
    s = _slots("unknown age under 500")
    assert s['price_max'] == 499.0
    assert 'domain_age_max' not in s
    assert 'domain_age_is_unknown' not in s


def test_age_older_than_sets_min() -> None:
    assert _slots("older than 10 years")['domain_age_min'] == 10


def test_age_newer_than_sets_max() -> None:
    assert _slots("newer than 2 years")['domain_age_max'] == 2


def test_keyword_starts_with() -> None:
    assert _slots("domains starting with go")['keyword_starts_with'] == ['go']


def test_keyword_ends_with() -> None:
    assert _slots("names ending with hub")['keyword_ends_with'] == ['hub']


def test_keyword_contains() -> None:
    assert _slots("domains containing cloud")['keyword_contains'] == ['cloud']


def test_char_no_hyphen_false() -> None:
    assert _slots("no hyphens")['has_hyphen'] is False


def test_char_with_numbers_true() -> None:
    assert _slots("with numbers")['has_number'] is True


def test_char_idn_true() -> None:
    assert _slots("unicode domains")['is_idn'] is True


def test_time_remaining_hours() -> None:
    assert _slots("ending in 2 hours")['time_remaining_max'] == 7200


def test_bare_number_no_slot() -> None:
    assert _slots("top 50 domains") == {} or 'price_max' not in _slots("top 50 domains")


def test_no_signal_empty() -> None:
    assert _slots("cool tech startup names") == {} or 'price_max' not in _slots("cool tech startup names")


def test_seo_trust_flow_min() -> None:
    assert _slots("trust flow over 30")['majestic_tf_min'] == 31


def test_seo_search_volume_max() -> None:
    # Explicit search-volume unit keeps family (not budget-steal to price).
    assert _slots("search volume under 1000")['semrush_search_volume_max'] == 1000


def test_seo_cpc_float() -> None:
    assert _slots("cpc under 2")['semrush_cpc_max'] == 2.0


def test_seo_referring_domains_min() -> None:
    assert _slots("referring domains over 50")['semrush_ref_domains_min'] == 51


def test_seo_majestic_backlinks_leading() -> None:
    assert _slots("majestic backlinks over 500")['majestic_backlinks_min'] == 501


def test_digits_min() -> None:
    assert _slots("with at least 2 digits")['minDigits'] == 2


def test_lifecycle_pending_delete() -> None:
    assert _slots("pending delete domains")['lifecycle_state'] == 'pending_delete'


def test_lifecycle_dropping() -> None:
    # Bare "dropping" → typeIncludeList=backorder (LLM parity; not lifecycle_state).
    assert _slots("dropping domains")['auction_type'] == ['backorder']
    assert 'lifecycle_state' not in _slots("dropping domains")


def test_char_pattern() -> None:
    assert _slots("cvcv pattern names")['charPattern'] == 'cvcv'


def test_gem_domains() -> None:
    # Bare "gem domains" → price_below_market (LLM); not isGemDomain.
    s = _slots("gem domains only")
    assert s.get('price_below_market') is True
    assert 'isGemDomain' not in s


def test_similar_to() -> None:
    assert _slots("names similar to google.com")['similar_to'] == ['google.com']


def test_similar_to_ignores_generic_like() -> None:
    assert 'similar_to' not in _slots("cool domains like these")


def test_similar_to_like_brands() -> None:
    assert set(_slots("domains like stripe or plaid")['similar_to']) == {'stripe', 'plaid'}


def test_real_visitors_traffic_soft() -> None:
    assert _slots("premium ai domain with real visitors")['has_web_traffic_signal'] is True


def test_capped_at_price() -> None:
    s = _slots(".io domains capped at 500")
    assert s['price_max'] == 500.0
    assert s['tld'] == ['io']


def test_no_vibe_keyword_exclude() -> None:
    s = _slots("fintech under 2k no crypto vibe")
    assert s['keyword_contains_exclude'] == ['crypto']
    assert s['price_max'] == 1999.0  # k/m exclusive


def test_buy_it_not_bid_sets_buy_it_now() -> None:
    assert _slots("want to just buy it not bid")['buy_it_now'] is True


def test_has_somewhere_in_it_contains() -> None:
    assert _slots("has tech somewhere in it")['keyword_contains'] == ['tech']


def test_bare_tld_or_list() -> None:
    assert set(_slots("com or io under 1k")['tld']) == {'com', 'io'}
    assert set(_slots("ai or io closing tonight")['tld']) == {'ai', 'io'}


def test_short_name_max_sld_len() -> None:
    assert _slots("new this week short com or io under 1k")['name_length_max'] == 5


def test_topic_domains() -> None:
    assert _slots("fintech topic domains")['topic_include'] == ['fintech']


def test_estibot_count_under_exclusive() -> None:
    s = _slots("low estibot count under 20 emerging")
    assert s.get('maxEstibotDomainCount') == 19
    assert 'price_max' not in s


def test_hidden_gem_below_market_not_is_gem() -> None:
    s = _slots("hidden gem domains")
    assert s.get('price_below_market') is True
    assert 'isGemDomain' not in s


def test_monthly_visitors_minimum_traffic() -> None:
    assert _slots("5000 monthly visitors minimum")['traffic_min'] == 5000


def test_plus_monthly_traffic() -> None:
    assert _slots("1000 plus monthly traffic")['traffic_min'] == 1000


def test_bidding_n_plus() -> None:
    assert _slots("lots of bidding 20 plus")['bids_min'] == 20


def test_pending_delete_with_bids() -> None:
    s = _slots("pending delete with bids")
    assert s['lifecycle_state'] == 'pending_delete'
    assert s['bids_min'] == 1


def test_starts_with_multi_and_bare_tld() -> None:
    s = _slots("starts with get or go com under 500")
    assert s['keyword_starts_with'] == ['get', 'go']
    assert s['tld'] == ['com']
    assert s['price_max'] == 499.0


def test_ending_this_weekend_secs() -> None:
    assert _slots("ending this weekend")['time_remaining_max'] == 259200


def test_tf_cf_not_price_min() -> None:
    s = _slots("tf above 20 and cf above 15 under 2k")
    assert s['majestic_tf_min'] == 21
    assert s['majestic_cf_min'] == 16
    assert s['price_max'] == 1999.0
    assert 'price_min' not in s


def test_extension_saturation_not_price() -> None:
    s = _slots("extension saturation above 30")
    assert s['semrush_authority_min'] == 31
    assert 'price_min' not in s
    assert 'price_max' not in s


def test_exact_phrase_not_contains() -> None:
    s = _slots("exact phrase fintech in the name")
    assert s['keyword_phrase'] == 'fintech'
    assert 'keyword_contains' not in s


def test_name_has_x_in_it_contains() -> None:
    assert _slots("name has health in it")['keyword_contains'] == ['health']


def test_registered_before_year_min_age() -> None:
    from datetime import datetime, timezone
    year = datetime.now(timezone.utc).year
    assert _slots("registered before 2010")['domain_age_min'] == year - 2010


def test_strong_link_profile() -> None:
    assert _slots("strong link profile")['majestic_backlinks_min'] == 1


def test_high_cpc_soft_floor() -> None:
    s = _slots("high cpc finance or legal niche under 3k")
    assert s['semrush_cpc_min'] == 1.0
    assert s['price_max'] == 2999.0
    assert set(s['topic_include']) == {'finance', 'legal'}


def test_new_keyword_listing_recency() -> None:
    assert _slots("new keyword low estibot count")['days_listed_max'] == 1


@pytest.mark.asyncio
async def test_classify_async_returns_hybrid_slice() -> None:
    result = await _extractor().classify_async("search .net domains under $50")
    assert result is not None and result.query_type == 'hybrid'
    names = {e.name for e in result.entities}
    assert 'tld' in names and 'price_max' in names
    price = next(e for e in result.entities if e.name == 'price_max')
    assert price.value == 49.0
    source_tag = AgentSearchConfig.from_dict(load_config()).qi.l0_regex_entity.source_tag
    assert all(e.source == source_tag for e in result.entities)


@pytest.mark.asyncio
async def test_classify_async_none_on_no_match() -> None:
    assert await _extractor().classify_async("beautiful brandable names") is None


@pytest.mark.asyncio
async def test_classify_async_none_when_disabled() -> None:
    assert await _extractor(enabled=False).classify_async("under $50 .com") is None
