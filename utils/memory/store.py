"""
TrialAgents Memory Bank - 存储管理

负责 ErrorCase, SuccessCase, PatternMeta 的持久化存储
"""

import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from .schema import ErrorCase, SuccessCase, PatternMeta


class MemoryStore:
    """
    内存存储管理器

    提供 ErrorCase, SuccessCase, PatternMeta 的增删改查操作
    """

    def __init__(self, memory_dir: str = None):
        """
        初始化存储管理器

        Args:
            memory_dir: 内存数据存储目录，默认使用 data/memory
        """
        if memory_dir is None:
            # 默认使用 data/memory 目录
            from utils.data.paths import get_data_path
            memory_dir = os.path.join(get_data_path(), "memory")

        self.memory_dir = memory_dir
        self.error_dir = os.path.join(memory_dir, "errors")
        self.success_dir = os.path.join(memory_dir, "successes")
        self.pattern_dir = os.path.join(memory_dir, "patterns")

        # 确保目录存在
        for d in [self.error_dir, self.success_dir, self.pattern_dir]:
            os.makedirs(d, exist_ok=True)

        # 内存缓存
        self._error_cache: Dict[str, ErrorCase] = {}
        self._success_cache: Dict[str, SuccessCase] = {}
        self._pattern_cache: Dict[str, PatternMeta] = {}

        # 加载已有数据
        self._load_all()

    def _load_all(self):
        """加载所有已有数据到内存缓存"""
        self._load_errors()
        self._load_successes()
        self._load_patterns()

    def _load_errors(self):
        """加载错误案例"""
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
        """加载成功案例"""
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
        """加载模式元数据"""
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
        添加错误案例

        Args:
            error_case: 错误案例

        Returns:
            case_id
        """
        # 更新频率
        if error_case.case_id in self._error_cache:
            existing = self._error_cache[error_case.case_id]
            existing.frequency += 1
            error_case = existing

        self._error_cache[error_case.case_id] = error_case

        # 持久化
        filepath = os.path.join(self.error_dir, f"{error_case.case_id}.json")
        with open(filepath, "w") as f:
            json.dump(error_case.to_dict(), f, indent=2)

        return error_case.case_id

    def add_success(self, success_case: SuccessCase) -> str:
        """
        添加成功案例

        Args:
            success_case: 成功案例

        Returns:
            case_id
        """
        self._success_cache[success_case.case_id] = success_case

        # 持久化
        filepath = os.path.join(self.success_dir, f"{success_case.case_id}.json")
        with open(filepath, "w") as f:
            json.dump(success_case.to_dict(), f, indent=2)

        return success_case.case_id

    def add_pattern(self, pattern: PatternMeta) -> str:
        """
        添加或更新模式元数据

        Args:
            pattern: 模式元数据

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
        """保存所有模式到文件"""
        patterns_file = os.path.join(self.pattern_dir, "patterns.json")
        patterns_data = [p.to_dict() for p in self._pattern_cache.values()]

        with open(patterns_file, "w") as f:
            json.dump(patterns_data, f, indent=2)

    def get_error(self, case_id: str) -> Optional[ErrorCase]:
        """获取错误案例"""
        return self._error_cache.get(case_id)

    def get_success(self, case_id: str) -> Optional[SuccessCase]:
        """获取成功案例"""
        return self._success_cache.get(case_id)

    def get_pattern(self, pattern_id: str) -> Optional[PatternMeta]:
        """获取模式元数据"""
        return self._pattern_cache.get(pattern_id)

    def get_all_errors(self) -> List[ErrorCase]:
        """获取所有错误案例"""
        return list(self._error_cache.values())

    def get_all_successes(self) -> List[SuccessCase]:
        """获取所有成功案例"""
        return list(self._success_cache.values())

    def get_all_patterns(self) -> List[PatternMeta]:
        """获取所有模式元数据"""
        return list(self._pattern_cache.values())

    def get_errors_by_type(self, error_type: str) -> List[ErrorCase]:
        """按错误类型筛选"""
        from .schema import ErrorType
        et = ErrorType(error_type)
        return [e for e in self._error_cache.values() if e.error_type == et]

    def get_errors_by_dataset(self, dataset: str) -> List[ErrorCase]:
        """按数据集筛选"""
        return [e for e in self._error_cache.values() if e.dataset == dataset]

    def get_successes_by_confidence(self, min_confidence: float) -> List[SuccessCase]:
        """筛选高置信度成功案例"""
        return [s for s in self._success_cache.values() if s.confidence >= min_confidence]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
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
        """清空所有缓存（谨慎使用）"""
        self._error_cache.clear()
        self._success_cache.clear()
        self._pattern_cache.clear()

        for d in [self.error_dir, self.success_dir, self.pattern_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
