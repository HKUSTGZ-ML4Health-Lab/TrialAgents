"""
TrialAgents Memory Bank - Storage Management

Responsible for persistent storage of ErrorCase, SuccessCase, PatternMeta
"""

import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from .schema import ErrorCase, SuccessCase, PatternMeta


class MemoryStore:
    """
    In-memory storage manager

    Provides CRUD operations for ErrorCase, SuccessCase, PatternMeta
    """

    def __init__(self, memory_dir: str = None):
        """
        Initialize storage manager

        Args:
            memory_dir: Memory data storage directory, defaults to data/memory
        """
        if memory_dir is None:
            # Default to data/memory directory
            from utils.data.paths import get_data_path
            memory_dir = os.path.join(get_data_path(), "memory")

        self.memory_dir = memory_dir
        self.error_dir = os.path.join(memory_dir, "errors")
        self.success_dir = os.path.join(memory_dir, "successes")
        self.pattern_dir = os.path.join(memory_dir, "patterns")

        # Ensure directories exist
        for d in [self.error_dir, self.success_dir, self.pattern_dir]:
            os.makedirs(d, exist_ok=True)

        # In-memory cache
        self._error_cache: Dict[str, ErrorCase] = {}
        self._success_cache: Dict[str, SuccessCase] = {}
        self._pattern_cache: Dict[str, PatternMeta] = {}

        # Load existing data
        self._load_all()

    def _load_all(self):
        """Load all existing data into memory cache"""
        self._load_errors()
        self._load_successes()
        self._load_patterns()

    def _load_errors(self):
        """Load error cases"""
        if not os.path.exists(self.error_dir):
            return

        for filename in os.listdir(self.error_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.error_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    error_case = ErrorCase.from_dict(data)
                    self._error_cache[error_case.case_id] = error_case
                except Exception as e:
                    print(f"Warning: Failed to load error case {filename}: {e}")

    def _load_successes(self):
        """Load success cases"""
        if not os.path.exists(self.success_dir):
            return

        for filename in os.listdir(self.success_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.success_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    success_case = SuccessCase.from_dict(data)
                    self._success_cache[success_case.case_id] = success_case
                except Exception as e:
                    print(f"Warning: Failed to load success case {filename}: {e}")

    def _load_patterns(self):
        """Load pattern metadata"""
        if not os.path.exists(self.pattern_dir):
            return

        patterns_file = os.path.join(self.pattern_dir, "patterns.json")
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, "r") as f:
                    data = json.load(f)
                for pattern_data in data:
                    pattern = PatternMeta.from_dict(pattern_data)
                    self._pattern_cache[pattern.pattern_id] = pattern
            except Exception as e:
                print(f"Warning: Failed to load patterns: {e}")

    def add_error(self, error_case: ErrorCase) -> str:
        """
        Add error case

        Args:
            error_case: Error case

        Returns:
            case_id
        """
        # Update frequency
        if error_case.case_id in self._error_cache:
            existing = self._error_cache[error_case.case_id]
            existing.frequency += 1
            error_case = existing

        self._error_cache[error_case.case_id] = error_case

        # Persist to disk
        filepath = os.path.join(self.error_dir, f"{error_case.case_id}.json")
        with open(filepath, "w") as f:
            json.dump(error_case.to_dict(), f, indent=2)

        return error_case.case_id

    def add_success(self, success_case: SuccessCase) -> str:
        """
        Add success case

        Args:
            success_case: Success case

        Returns:
            case_id
        """
        self._success_cache[success_case.case_id] = success_case

        # Persist to disk
        filepath = os.path.join(self.success_dir, f"{success_case.case_id}.json")
        with open(filepath, "w") as f:
            json.dump(success_case.to_dict(), f, indent=2)

        return success_case.case_id

    def add_pattern(self, pattern: PatternMeta) -> str:
        """
        Add or update pattern metadata

        Args:
            pattern: Pattern metadata

        Returns:
            pattern_id
        """
        if pattern.pattern_id in self._pattern_cache:
            existing = self._pattern_cache[pattern.pattern_id]
            existing.frequency += 1
            pattern = existing

        self._pattern_cache[pattern.pattern_id] = pattern
        self._save_patterns()

        return pattern.pattern_id

    def _save_patterns(self):
        """Save all patterns to file"""
        patterns_file = os.path.join(self.pattern_dir, "patterns.json")
        patterns_data = [p.to_dict() for p in self._pattern_cache.values()]

        with open(patterns_file, "w") as f:
            json.dump(patterns_data, f, indent=2)

    def get_error(self, case_id: str) -> Optional[ErrorCase]:
        """Get error case"""
        return self._error_cache.get(case_id)

    def get_success(self, case_id: str) -> Optional[SuccessCase]:
        """Get success case"""
        return self._success_cache.get(case_id)

    def get_pattern(self, pattern_id: str) -> Optional[PatternMeta]:
        """Get pattern metadata"""
        return self._pattern_cache.get(pattern_id)

    def get_all_errors(self) -> List[ErrorCase]:
        """Get all error cases"""
        return list(self._error_cache.values())

    def get_all_successes(self) -> List[SuccessCase]:
        """Get all success cases"""
        return list(self._success_cache.values())

    def get_all_patterns(self) -> List[PatternMeta]:
        """Get all pattern metadata"""
        return list(self._pattern_cache.values())

    def get_errors_by_type(self, error_type: str) -> List[ErrorCase]:
        """Filter by error type"""
        from .schema import ErrorType
        et = ErrorType(error_type)
        return [e for e in self._error_cache.values() if e.error_type == et]

    def get_errors_by_dataset(self, dataset: str) -> List[ErrorCase]:
        """Filter by dataset"""
        return [e for e in self._error_cache.values() if e.dataset == dataset]

    def get_successes_by_confidence(self, min_confidence: float) -> List[SuccessCase]:
        """Filter high-confidence success cases"""
        return [s for s in self._success_cache.values() if s.confidence >= min_confidence]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        errors = list(self._error_cache.values())
        successes = list(self._success_cache.values())

        error_types = {}
        for e in errors:
            error_types[e.error_type.value] = error_types.get(e.error_type.value, 0) + 1

        return {
            "total_errors": len(errors),
            "total_successes": len(successes),
            "total_patterns": len(self._pattern_cache),
            "error_types_distribution": error_types,
            "avg_error_frequency": sum(e.frequency for e in errors) / max(len(errors), 1),
            "avg_success_confidence": sum(s.confidence for s in successes) / max(len(successes), 1),
        }

    def clear(self):
        """Clear all caches (use with caution)"""
        self._error_cache.clear()
        self._success_cache.clear()
        self._pattern_cache.clear()

        for d in [self.error_dir, self.success_dir, self.pattern_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
