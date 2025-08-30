"""Speaker name matching utilities."""

import re
from typing import Dict, List, Optional, Set, Tuple
from Levenshtein import ratio

class SpeakerMatcher:
    def __init__(self):
        self._speakers: Dict[str, Set[str]] = {}  # normalized -> variations
        self._cache: Dict[str, str] = {}  # exact match cache
    
    def add_speaker(self, name: str) -> None:
        """Add a speaker name and its variations to the matcher."""
        if not name:
            return
        
        # Normalize: lowercase, strip whitespace
        normalized = self._normalize(name)
        if not normalized:
            return
            
        # Add to speakers dict
        if normalized not in self._speakers:
            self._speakers[normalized] = set()
        self._speakers[normalized].add(name)
        
        # Cache exact match
        self._cache[name.lower()] = name
        
        # Add first/last name variations
        parts = normalized.split()
        if len(parts) > 1:
            self._speakers[normalized].add(parts[0])  # First name
            self._speakers[normalized].add(parts[-1])  # Last name
            
    def match(self, query: str, min_ratio: float = 0.8) -> Optional[str]:
        """Find the best matching speaker name.
        
        Args:
            query: The name to match
            min_ratio: Minimum Levenshtein ratio (0-1) to consider a match
            
        Returns:
            The original form of the best matching name, or None if no good match
        """
        if not query:
            return None
            
        # Check exact match cache first
        query_lower = query.lower().strip()
        if query_lower in self._cache:
            return self._cache[query_lower]
            
        # Normalize query
        normalized = self._normalize(query)
        if not normalized:
            return None
            
        # Find best match
        best_match = None
        best_ratio = min_ratio
        
        for speaker, variations in self._speakers.items():
            # Try exact match on normalized speaker or any variation
            if normalized == speaker or normalized in {self._normalize(v) for v in variations}:
                # Get the longest (most complete) variation as canonical form
                return max(variations, key=len)
            
            # Handle initials: "J. Smith" should match "John Smith"
            if "." in query:
                parts = query.lower().replace(".", "").split()
                if len(parts) >= 2:
                    initial = parts[0][0]  # Get "j" from "j smith"
                    last = parts[-1]       # Get "smith"
                    spk_parts = speaker.split()
                    if len(spk_parts) >= 2 and spk_parts[0][0].lower() == initial and spk_parts[-1].lower() == last:
                        return max(variations, key=len)
                
            # Try fuzzy match on speaker and variations
            for variation in variations:
                r = ratio(normalized, self._normalize(variation))
                if r > best_ratio:
                    best_ratio = r
                    best_match = variation
                    
        return best_match

    @staticmethod                
    def _normalize(name: str) -> str:
        """Normalize a name for matching: lowercase, strip whitespace, handle initials."""
        name = " ".join(name.lower().split())
        # Handle initials: "J. Smith" -> "j smith"
        return re.sub(r'([a-z])\.\s*', r'\1 ', name)
