def test_speaker_matcher():
    from src.search.speaker import SpeakerMatcher
    
    matcher = SpeakerMatcher()
    
    # Test exact matches
    matcher.add_speaker("John Smith")
    assert matcher.match("John Smith") == "John Smith"
    assert matcher.match("john smith") == "John Smith"
    
    # Test partial matches
    assert matcher.match("John") == "John Smith"
    assert matcher.match("Smith") == "John Smith"
    
    # Test fuzzy matches
    assert matcher.match("Jon Smith") == "John Smith"  # Common typo
    assert matcher.match("J. Smith") == "John Smith"   # Initial
    
    # Test multiple speakers
    matcher.add_speaker("Jane Smith")
    assert matcher.match("Jane") == "Jane Smith"
    assert matcher.match("Smith") in {"John Smith", "Jane Smith"}  # Either is acceptable

def test_meta_caching():
    from src.search.cache import meta_cache, clear_caches
    from src.search.index import MasterMetaIndex
    
    # Create a test index
    idx = MasterMetaIndex()
    idx.by_workshop[("PEP", 2025, 1, 1)] = {
        "program": "Workshop",
        "cohort": "PEP",
        "cohort_year": 2025,
        "workshop_number": 1,
        "session_number": 1,
        "speakers": "John Smith"
    }
    
    # Clear cache and verify lookup
    clear_caches()
    assert len(meta_cache) == 0
    
    # Do a lookup - should cache
    result = idx.lookup_workshop("PEP", 2025, 1, 1)
    assert result is not None
    assert result["speakers"] == "John Smith"
    assert len(meta_cache) == 1
    
    # Same lookup - should hit cache
    result2 = idx.lookup_workshop("PEP", 2025, 1, 1)
    assert result2 is result  # Same object (from cache)
    
    # Clear cache - verify gone
    clear_caches()
    assert len(meta_cache) == 0
