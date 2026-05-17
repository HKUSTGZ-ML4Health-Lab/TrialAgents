"""
TrialAgents 主入口

从 m-agents/collaboration_pipeline.py 迁移并重构
整合三阶段工作流的完整执行
"""

import json
import os
import traceback
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 导入TrialAgents模块
from src.agents import ClinicianAgent, SupervisorAgent
from src.collaboration import CollaborationEngine
from utils.config import DATASET_CONFIGS, AGENT_CONFIG, DEFAULT_LLM, DATA_PATHS
from utils.data_utils.paths import build_experiment_path, ensure_dir

# 导入适配器（从原项目导入工具函数）
try:
    from utils.adapters import RetrievalSystem, wrap_data, get_final_result
except ImportError:
    # 如果适配器还没准备好，创建占位符
    print("Warning: utils.adapters not found. Some features may not work.")
    RetrievalSystem = None
    wrap_data = None
    get_final_result = None

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_collaboration_pipeline(
    config_override: dict = None,
    start_idx: int = 0,
    end_idx: int = None,
    dataset_name: str = None,
    llm: str = None,
    task: str = "MRP",
    group: str = "experimental"
):
    """
    运行完整的TrialAgents协作管道
    
    三阶段工作流：
    1. Phase 1: 临床综述生成（Clinician Agents并行分析）
    2. Phase 2: 多轮协同诊断（Collaboration Engine）
    3. Phase 3: 终版报告生成（Supervisor Agent修订）
    
    Args:
        config_override: 配置覆盖参数
        start_idx: 处理开始索引
        end_idx: 处理结束索引
        dataset_name: 数据集名称
        llm: LLM模型名称
        task: 任务名称
        group: 组别（control/experimental）
        
    Returns:
        处理结果字典
    """
    # 构建配置
    config = {
        "dataset_name": dataset_name or "NCT00694382",
        "task": task,
        "llm": llm or DEFAULT_LLM,
        "group": group,
        "clinician_num": AGENT_CONFIG.get("clinician_num", 3),
        "max_round": AGENT_CONFIG.get("max_round", 3),
        "retriever_name": "MedCPT",
        "corpus_name": "MSD",
        "mode": task,
        "model_names": ["time_series_model"] * AGENT_CONFIG.get("clinician_num", 3),
        "trial_path": DATA_PATHS.get("trial_path", "/home/chunyu.lei/DEAP/SELF_DATA/mas/kbvae/data"),
    }
    
    # 应用配置覆盖
    if config_override:
        config.update(config_override)
    
    # 确定处理范围
    dataset_info = DATASET_CONFIGS.get(config["dataset_name"], {})
    total_patients = dataset_info.get("num_patients", 100)
    
    if end_idx is None:
        end_idx = total_patients
    
    # 创建实验输出目录
    save_dir = build_experiment_path(
        dataset_name=config["dataset_name"],
        task=config["task"],
        llm=config["llm"],
        corpus_name=config["corpus_name"],
        group=config["group"],
    )
    ensure_dir(save_dir)
    
    print('='*60)
    print("TrialAgents Pipeline Started")
    print(f"{'='*60}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Task: {config['task']}")
    print(f"LLM: {config['llm']}")
    print(f"Patients: {start_idx} to {end_idx}")
    print(f"Save Directory: {save_dir}")
    print('='*60)
    
    # 准备数据
    if wrap_data is None:
        raise ImportError("wrap_data not available. Please check utils.adapters")
    
    patients_list = wrap_data(config)
    patients_list = patients_list[start_idx:end_idx]
    
    # 提取标签（用于评估）
    label_lst = [int(patients_list[i][-1]) for i in range(len(patients_list))]
    print(f"Labels distribution - 0: {label_lst.count(0)}, 1: {label_lst.count(1)}")
    
    # 记录处理信息
    with open(f'{save_dir}/exp_info.txt', 'a') as f:
        f.write("\n")
        f.write(f"{'='*60}")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        f.write(f"Dataset: {config['dataset_name']}\\t Task: {config['task']}\\t Model: {config['llm']} ")
        f.write(f"Labels: {label_lst}")
        f.write(f"Labels 0: {label_lst.count(0)}, 1: {label_lst.count(1)}")
        f.write(f"{'='*60}")
        f.write("\n")

    # 初始化检索系统
    if RetrievalSystem is not None:
        retrieval_system = RetrievalSystem(
            retriever_name=config["retriever_name"],
            corpus_name=config["corpus_name"]
        )
    else:
        retrieval_system = None
        print("Warning: RetrievalSystem not available")

    # ========== Memory Bank 初始化 ==========
    memory_injector = None
    memory_updater = None
    try:
        from utils.memory import MemoryStore, MemoryEngine, MemoryInjector, MemoryUpdater
        memory_store = MemoryStore()
        memory_engine = MemoryEngine(store=memory_store)
        memory_injector = MemoryInjector(memory_engine)
        memory_updater = MemoryUpdater(store=memory_store, engine=memory_engine)
        print("Memory Bank initialized successfully")
    except ImportError as e:
        print(f"Warning: Memory Bank not available: {e}")
    except Exception as e:
        print(f"Warning: Memory Bank initialization failed: {e}")
    # ======================================

    # 初始化Clinician Agents（带Memory注入器）
    clinician_agents = [
        ClinicianAgent(
            config, i, retrieval_system,
            llm=config["llm"], save_dir=save_dir,
            memory_injector=memory_injector
        )
        for i in range(config["clinician_num"])
    ]

    # 初始化Supervisor Agent（带Memory注入器）
    supervisor_agent = SupervisorAgent(llm=config["llm"], memory_injector=memory_injector)
    print(f"Loaded SupervisorAgent with LLM: {supervisor_agent.llm}")

    for i, clinician_agent in enumerate(clinician_agents):
        print(f"Loaded ClinicianAgent {i + 1} with LLM: {clinician_agent.llm}")

    print("All agents loaded successfully")
    
    # 初始化统计变量
    result_json = {}
    all_prompt_tokens = 0
    all_completion_tokens = 0
    total_start = time.time()
    
    # 处理每个患者
    for patient_index, patient_df in tqdm(
        enumerate(patients_list),
        total=len(patients_list),
        desc=f"Processing patients in {config['dataset_name']}"
    ):
        patient_start = time.time()
        patient_id = patient_df[0]
        
        # 检查是否已经处理过
        sub_save_dir = f"{save_dir}/{patient_id}"
        final_result_file = os.path.join(sub_save_dir, "zz_final_logits.json")
        
        if os.path.exists(final_result_file):
            print(f"Skipping {patient_id} - already processed")
            continue
        
        ensure_dir(sub_save_dir)
        print(f"{'='*60}")
        print(f"Processing Patient: {patient_id}")
        print(f"{'='*60}")
        
        analysis_results = []
        prompt_tokens = 0
        completion_tokens = 0
        basic_context = None
        
        try:
            # ==================== Phase 1: Clinical Review Generation ====================
            print(">>> Phase 1: Clinical Review Generation")
            
            def process_clinician(clinician_agent, idx):
                """处理单个ClinicianAgent的分析"""
                response, context, messages, p_token, c_token = clinician_agent.analysis(
                    patient_index, patient_id
                )
                
                # 保存Clinician的综述
                with open(f"{sub_save_dir}/clinician{idx + 1}_review.json", "w") as f:
                    json.dump(response, f, indent=2)
                
                # with open(f"{sub_save_dir}/clinician{idx + 1}_review_messages.json", "w") as f:
                #     json.dump(messages, f, indent=2)
                
                if len(messages) > 1:
                    with open(f"{sub_save_dir}/clinician{idx + 1}_review_userprompt.txt", "w") as f:
                        f.write(messages[1].get("content", ""))
                
                return response, p_token, c_token, context
            
            # 并行执行所有ClinicianAgent
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_clinician, clinician_agents[i], i)
                    for i in range(config["clinician_num"])
                ]
                
                for future in futures:
                    response, p_token, c_token, context = future.result()
                    analysis_results.append(response)
                    prompt_tokens += p_token
                    completion_tokens += c_token
                    basic_context = context
            
            print(f"Phase 1 completed - Generated {len(analysis_results)} clinical reviews")
            
            # ==================== Phase 2: Multi-round Collaborative Diagnosis ====================
            print(">>> Phase 2: Multi-round Collaborative Diagnosis")
            
            # Supervisor生成初始综述
            supervisor_agent.set_basic_info(basic_context)
            summary_content, messages, p_token, c_token = supervisor_agent.summary(
                analysis_results, is_initial=True
            )
            
            with open(f"{sub_save_dir}/trialist_summary.json", "w") as f:
                json.dump(summary_content, f, indent=2)
            
            # with open(f"{sub_save_dir}/trialist_messages.json", "w") as f:
            #     json.dump(messages, f, indent=2)
            
            if len(messages) > 1:
                with open(f"{sub_save_dir}/trialist_userprompt.txt", "w") as f:
                    f.write(messages[1].get("content", ""))
            
            prompt_tokens += p_token
            completion_tokens += c_token
            
            print(f"Initial summary generated")
            
            # 多轮协作讨论
            collaboration = CollaborationEngine(
                supervisor_agent=supervisor_agent,
                clinician_agents=clinician_agents,
                supervisor_opinion=summary_content.get("answer", ""),
                supervisor_report=summary_content.get("report", ""),
                save_dir=sub_save_dir,
                clinician_num=config["clinician_num"],
                max_round=config["max_round"]
            )
            
            current_report, rounds_completed, p_tokens, c_tokens = collaboration.collaborate()
            
            with open(f"{sub_save_dir}/trialist_final_summary.json", "w") as f:
                json.dump({"final_report": current_report, "rounds": rounds_completed}, f, indent=2)
            
            prompt_tokens += p_tokens
            completion_tokens += c_tokens
            
            print(f"Phase 2 completed - {rounds_completed} rounds of collaboration")
            
            # ==================== Phase 3: Final Report Generation ====================
            print(">>> Phase 3: Final Report Generation")

            # 获取各Clinician的最终分析
            latest_analysis = []
            for i, clinician_agent in enumerate(clinician_agents):
                ans = clinician_agent.get_latest_analysis(sub_save_dir, i + 1)
                latest_analysis.append(ans)

            # 构建患者特征文本（用于Memory检索）
            patient_profile_text = f"Patient {patient_id} ({config['dataset_name']}/{config['group']})"

            # Supervisor修订最终预测（带Memory注入）
            logits, messages, p_token, c_token = supervisor_agent.revise_logits(
                latest_analysis, patient_profile_text=patient_profile_text
            )

            with open(f"{sub_save_dir}/supervisor_userprompt.txt", "w") as f:
                if len(messages) > 1:
                    f.write(messages[1].get("content", ""))

            # 获取最终结果
            if get_final_result is not None:
                logits['final_logit'], logits['final_conf'] = get_final_result(logits)
            else:
                # 占位实现
                logits['final_logit'] = logits.get('logit', 0.5)
                logits['final_conf'] = logits.get('conf_logit', 0.5)

            logits['ground_truth'] = label_lst[patient_index]

            with open(f"{sub_save_dir}/zz_final_logits.json", "w") as f:
                json.dump(logits, f, indent=2)

            # ========== Memory Bank: 登记预测结果 ==========
            if memory_updater is not None:
                try:
                    from utils.memory import PatientProfile
                    profile = PatientProfile(
                        patient_id=patient_id,
                        dataset=config["dataset_name"],
                        group=config["group"],
                        key_features={"dataset": config["dataset_name"]},
                        prediction=logits.get('final_logit', 0.5),
                        ground_truth=logits['ground_truth'],
                    )
                    # 获取第一个clinician的分析作为代表
                    clinician_analysis_text = latest_analysis[0].get('analysis', '') if latest_analysis else ''
                    is_error, _ = memory_updater.register_prediction_result(
                        case_id=f"{config['dataset_name']}_{patient_id}",
                        dataset=config["dataset_name"],
                        patient_profile=profile,
                        clinician_index=0,
                        clinician_analysis=clinician_analysis_text,
                        prediction=logits.get('final_logit', 0.5),
                        ground_truth=logits['ground_truth'],
                        clinician_confidence=logits.get('final_conf', 0.5),
                    )
                    if is_error:
                        print(f"  [Memory] Registered error case: {patient_id}")
                except Exception as e:
                    print(f"  [Memory] Warning: Failed to register result: {e}")
            # ==============================================

            prompt_tokens += p_token
            completion_tokens += c_token

            print(f"Phase 3 completed - Final prediction: {logits.get('final_logit', 'N/A')}")
            
            # 记录结果
            patient_end = time.time()
            result_json[patient_id] = {
                "report": current_report,
                "final_prediction": logits.get('final_logit'),
                "final_confidence": logits.get('final_conf'),
                "ground_truth": logits['ground_truth'],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "time": f"{patient_end - patient_start:.2f}s",
                "rounds": rounds_completed
            }
            
            all_prompt_tokens += prompt_tokens
            all_completion_tokens += completion_tokens
            
            print(f"Patient {patient_id} completed in {patient_end - patient_start:.2f}s")
            
        except Exception as e:
            print(f"Error processing patient {patient_id}: {str(e)}")
            traceback.print_exc()
            result_json[patient_id] = {"error": str(e)}
            continue
    
    # 保存总结果
    total_end = time.time()
    result_json["total_prompt_tokens"] = all_prompt_tokens
    result_json["total_completion_tokens"] = all_completion_tokens
    result_json["total_time"] = f"{total_end - total_start:.2f}s"
    result_json["processed_count"] = len([k for k in result_json.keys() if not k.startswith("total")])
    
    print(f"{'='*60}")
    print("TrialAgents Pipeline Completed")
    print(f"{'='*60}")
    print(f"Processed: {result_json['processed_count']} patients")
    print(f"Total time: {result_json['total_time']}")
    print(f"Total prompt tokens: {all_prompt_tokens:,}")
    print(f"Total completion tokens: {all_completion_tokens:,}")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}")
    
    return {
        "result_json": result_json,
        "save_dir": save_dir,
        "processed_count": result_json["processed_count"],
        "total_prompt_tokens": all_prompt_tokens,
        "total_completion_tokens": all_completion_tokens,
        "total_time": total_end - total_start
    }


def main():
    """主函数 - 命令行入口"""
    parser = argparse.ArgumentParser(
        description="TrialAgents - Virtual Clinical Trial Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on NCT00694382
  python main.py --dataset NCT00694382 --start_idx 0 --end_idx 10
  
  # Run with specific LLM
  python main.py --dataset NCT00694382 --llm gpt-4o-mini --group experimental
  
  # Run specific task
  python main.py --dataset NCT00694382 --task mortality_prediction
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="NCT00694382",
        help="Dataset name (e.g., NCT00694382, NCT00174655, NCT00003299)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="MRP",
        help="Task name (default: mortality_risk_prediction)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=DEFAULT_LLM,
        help=f"LLM model name (default: {DEFAULT_LLM})"
    )
    parser.add_argument(
        "--group",
        type=str,
        default="experimental",
        choices=["experimental", "control"],
        help="Patient group (experimental or control)"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for patient processing"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=10,
        help="End index for patient processing (default: 10)"
    )
    parser.add_argument(
        "--clinician_num",
        type=int,
        default=None,
        help=f"Number of clinician agents (default: {AGENT_CONFIG.get('clinician_num', 3)})"
    )
    parser.add_argument(
        "--max_round",
        type=int,
        default=3,
        help=f"Maximum collaboration rounds (default: {AGENT_CONFIG.get('max_round', 3)})"
    )
    
    args = parser.parse_args()
    
    # 构建配置覆盖
    config_override = {}
    if args.clinician_num is not None:
        config_override["clinician_num"] = args.clinician_num
    if args.max_round is not None:
        config_override["max_round"] = args.max_round
    
    # 运行管道
    result = run_collaboration_pipeline(
        config_override=config_override,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        dataset_name=args.dataset,
        llm=args.llm,
        task=args.task,
        group=args.group
    )

    # 保存results
    with open(f"{result['save_dir']}/result.json", "a") as f:
        json.dump(result['result_json'], f, indent=2)
        f.write("\n")
    print(f"Pipeline execution completed successfully!")
    print(f"Processed {result['processed_count']} patients")


if __name__ == "__main__":
    main()
