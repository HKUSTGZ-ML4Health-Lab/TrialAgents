"""
TrialAgents 迁移验证脚本

检查迁移完整性，输出迁移状态报告
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()


def check_file_exists(path: str, description: str) -> bool:
    """检查文件是否存在"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists


def check_symbolic_link(path: str, description: str) -> bool:
    """检查符号链接"""
    is_link = os.path.islink(path)
    exists = os.path.exists(path)
    
    if is_link and exists:
        target = os.readlink(path)
        print(f"✅ {description}: {path} -> {target}")
        return True
    elif is_link and not exists:
        print(f"⚠️  {description}: {path} -> BROKEN LINK")
        return False
    else:
        print(f"❌ {description}: {path} (not a symlink)")
        return False


def verify_core_files() -> dict:
    """验证核心文件"""
    print("\\n" + "="*60)
    print("1. 验证核心文件")
    print("="*60)
    
    files = {
        "主文档": PROJECT_ROOT / "README.md",
        "快速开始": PROJECT_ROOT / "QUICKSTART.md",
        "许可证": PROJECT_ROOT / "LICENSE",
        "依赖配置": PROJECT_ROOT / "requirements.txt",
        "Git配置": PROJECT_ROOT / ".gitignore",
        "Cursor规则": PROJECT_ROOT / ".cursorrules",
        "主入口": PROJECT_ROOT / "main.py",
    }
    
    results = {}
    for desc, path in files.items():
        results[desc] = check_file_exists(path, desc)
    
    return results


def verify_core_modules() -> dict:
    """验证核心代码模块"""
    print("\\n" + "="*60)
    print("2. 验证核心代码模块")
    print("="*60)
    
    modules = {
        "Agent基类": PROJECT_ROOT / "src" / "agents" / "base_agent.py",
        "ClinicianAgent": PROJECT_ROOT / "src" / "agents" / "clinician_agent.py",
        "SupervisorAgent": PROJECT_ROOT / "src" / "agents" / "supervisor_agent.py",
        "SimulatorAgent": PROJECT_ROOT / "src" / "agents" / "simulator_agent.py",
        "JSON解析器": PROJECT_ROOT / "src" / "agents" / "json_parser.py",
        "协作引擎": PROJECT_ROOT / "src" / "collaboration" / "collaboration_engine.py",
        "阶段1脚本": PROJECT_ROOT / "src" / "pipeline" / "phase1_review.py",
        "阶段2脚本": PROJECT_ROOT / "src" / "pipeline" / "phase2_collaborate.py",
        "阶段3脚本": PROJECT_ROOT / "src" / "pipeline" / "phase3_execute.py",
        "路径工具": PROJECT_ROOT / "utils" / "data" / "paths.py",
        "适配器": PROJECT_ROOT / "utils" / "adapters.py",
        # 第一优先级迁移
        "wrap_data完整实现": PROJECT_ROOT / "utils" / "data" / "wrap_data.py",
        "LNL训练核心": PROJECT_ROOT / "src" / "training" / "lnl_train.py",
        "LNL数据加载器": PROJECT_ROOT / "src" / "training" / "lnl_dataloader.py",
        "Fusion模型": PROJECT_ROOT / "src" / "training" / "fusion.py",
        "软标签工具": PROJECT_ROOT / "src" / "training" / "soft_label_utils.py",
        "LNL训练入口": PROJECT_ROOT / "run_lnl_training.py",
        "小模型架构": PROJECT_ROOT / "baselines" / "small_models.py",
        "时序基线": PROJECT_ROOT / "baselines" / "run_ts_small_models.py",
        "表格基线": PROJECT_ROOT / "baselines" / "run_tabular_small_models.py",
    }
    
    results = {}
    for desc, path in modules.items():
        results[desc] = check_file_exists(path, desc)
    
    return results


def verify_data_symlinks() -> dict:
    """验证数据符号链接"""
    print("\\n" + "="*60)
    print("3. 验证数据符号链接")
    print("="*60)
    
    symlinks = {
        "原始数据(raw)": PROJECT_ROOT / "data" / "raw",
        "处理后数据(processed)": PROJECT_ROOT / "data" / "processed",
        "检查点(checkpoints)": PROJECT_ROOT / "data" / "checkpoints",
        "响应输出(responses)": PROJECT_ROOT / "data" / "outputs" / "responses",
        "数字孪生(digital_twins)": PROJECT_ROOT / "data" / "outputs" / "digital_twins",
    }
    
    results = {}
    for desc, path in symlinks.items():
        results[desc] = check_symbolic_link(path, desc)
    
    return results


def verify_imports() -> dict:
    """验证Python导入"""
    print("\\n" + "="*60)
    print("4. 验证Python导入")
    print("="*60)
    
    sys.path.insert(0, str(PROJECT_ROOT))
    
    imports = {
        "src.agents": ["Agent", "ClinicianAgent", "SupervisorAgent", "SimulatorAgent"],
        "src.collaboration": ["CollaborationEngine"],
        "src.pipeline": ["run_phase1", "run_phase2", "run_phase3"],
        "utils.config": ["LLM_CONFIGS", "DATASET_CONFIGS"],
        "utils.data.paths": ["get_data_path", "build_experiment_path"],
        # 第一优先级迁移
        "src.training": ["run_dividemix", "nct_dataloader", "Fusion", "build_soft_labels"],
        "src.training.lnl_train": ["run_dividemix"],
        "src.training.lnl_dataloader": ["nct_dataloader", "nct_dataset"],
        "src.training.fusion": ["Fusion", "TransformerEncoderLayer"],
        "src.training.soft_label_utils": ["build_soft_labels", "validate_soft_labels"],
        "utils.data.wrap_data": ["wrap_data", "serialize_time_series_to_tokens"],
        "baselines.small_models": ["NCTTimeSeriesTransformer", "LSTMPredictor"],
    }
    
    results = {}
    for module_name, items in imports.items():
        try:
            module = __import__(module_name, fromlist=items)
            for item in items:
                if hasattr(module, item):
                    print(f"✅ {module_name}.{item}")
                    results[f"{module_name}.{item}"] = True
                else:
                    print(f"❌ {module_name}.{item} (not found)")
                    results[f"{module_name}.{item}"] = False
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            for item in items:
                results[f"{module_name}.{item}"] = False
    
    return results


def verify_documentation() -> dict:
    """验证文档"""
    print("\\n" + "="*60)
    print("5. 验证文档文件")
    print("="*60)
    
    docs = {
        "项目概述": PROJECT_ROOT / "docs" / "project.md",
        "系统架构": PROJECT_ROOT / "docs" / "architecture.md",
        "迁移指南": PROJECT_ROOT / "docs" / "migration.md",
        "阶段1文档": PROJECT_ROOT / "docs" / "workflow" / "phase1_review_generation.md",
        "阶段2文档": PROJECT_ROOT / "docs" / "workflow" / "phase2_collaboration.md",
        "阶段3文档": PROJECT_ROOT / "docs" / "workflow" / "phase3_task_completion.md",
    }
    
    results = {}
    for desc, path in docs.items():
        results[desc] = check_file_exists(path, desc)
    
    return results


def generate_report(all_results: dict) -> str:
    """生成验证报告"""
    report_lines = [
        "\\n" + "="*60,
        "TrialAgents 迁移验证报告",
        "="*60,
        f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"项目路径: {PROJECT_ROOT}",
        "",
    ]
    
    total_checks = 0
    passed_checks = 0
    
    for category, results in all_results.items():
        report_lines.append(f"\\n【{category}】")
        for item, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            report_lines.append(f"  {status}: {item}")
            total_checks += 1
            if passed:
                passed_checks += 1
    
    report_lines.extend([
        "",
        "="*60,
        f"总计: {passed_checks}/{total_checks} 项检查通过",
        f"成功率: {passed_checks/total_checks*100:.1f}%" if total_checks > 0 else "成功率: N/A",
        "="*60,
    ])
    
    if passed_checks == total_checks:
        report_lines.append("\\n🎉 所有检查通过！迁移成功。")
    elif passed_checks >= total_checks * 0.8:
        report_lines.append("\\n⚠️  大部分检查通过，但有一些问题需要修复。")
    else:
        report_lines.append("\\n❌ 检查未通过较多，请检查迁移过程。")
    
    return "\\n".join(report_lines)


def main():
    """主函数"""
    print("\\n" + "="*60)
    print("TrialAgents 迁移验证")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行所有验证
    all_results = {
        "核心文件": verify_core_files(),
        "代码模块": verify_core_modules(),
        "数据符号链接": verify_data_symlinks(),
        "Python导入": verify_imports(),
        "文档文件": verify_documentation(),
    }
    
    # 生成并打印报告
    report = generate_report(all_results)
    print(report)
    
    # 保存报告到文件
    report_file = PROJECT_ROOT / "migration_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report.replace("\\n", "\n"))
    print(f"\\n报告已保存到: {report_file}")
    
    # 返回退出码
    total = sum(len(r) for r in all_results.values())
    passed = sum(sum(1 for v in r.values() if v) for r in all_results.values())
    
    if passed == total:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
