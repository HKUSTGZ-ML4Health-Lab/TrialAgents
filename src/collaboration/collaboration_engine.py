"""
TrialAgents Collaboration Engine

Migrated and refactored from m-agents/utils/framework.py
Manages the multi-round collaborative discussion workflow between agents.
"""

import json
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..agents.supervisor_agent import SupervisorAgent  # pyright: ignore[reportMissingImports]
from ..agents.clinician_agent import ClinicianAgent  # pyright: ignore[reportMissingImports]


class CollaborationEngine:
    """
    Collaboration Engine

    Manages the multi-round collaborative discussion workflow, coordinating
    between SupervisorAgent and multiple ClinicianAgents.

    Workflow:
    1. Initialization: Receive initial Supervisor opinion and report
    2. Multi-round discussion: Collect Clinician feedback each round, Supervisor integrates
    3. Termination: Decide whether to continue or finalize based on Supervisor's action
    4. Output: Final report and token statistics

    Attributes:
        supervisor_agent: SupervisorAgent instance
        clinician_agents: List of ClinicianAgent instances
        supervisor_opinion: Current Supervisor opinion
        supervisor_report: Current Supervisor report
        clinician_responses: Clinician response list
        clinician_num: Number of clinicians
        max_round: Maximum discussion rounds
        now_round: Current round number
        save_dir: Save directory
    """

    def __init__(
        self,
        supervisor_agent: SupervisorAgent,
        clinician_agents: List[ClinicianAgent],
        supervisor_opinion: str,
        supervisor_report: str,
        save_dir: str,
        clinician_num: int = 3,
        max_round: int = 3
    ):
        """
        Initialize the Collaboration Engine.

        Args:
            supervisor_agent: SupervisorAgent instance
            clinician_agents: List of ClinicianAgent instances
            supervisor_opinion: Initial Supervisor opinion
            supervisor_report: Initial Supervisor report
            save_dir: Save directory path
            clinician_num: Number of clinicians
            max_round: Maximum discussion rounds
        """
        self.supervisor_agent = supervisor_agent
        self.clinician_agents = clinician_agents
        self.supervisor_opinion = supervisor_opinion
        self.supervisor_report = supervisor_report
        self.save_dir = save_dir
        self.clinician_num = clinician_num
        self.max_round = max_round

        self.clinician_responses = []
        self.now_round = 0

    def collaborate(self) -> Tuple[str, int, int, int]:
        """
        Execute multi-round collaborative discussion.

        Core method for Phase 2 collaboration flow:
        1. Collect feedback from all Clinicians in parallel each round
        2. Supervisor integrates feedback and generates new report
        3. Supervisor decides whether to continue or finalize
        4. Exit when max_round reached or Supervisor decides to finalize

        Returns:
            (final_report, final_round, prompt_tokens, completion_tokens)
        """
        assert len(self.clinician_responses) == 0 and self.now_round == 0, \
            "Collaboration already started or invalid state"

        prompt_tokens = 0
        completion_tokens = 0

        while self.now_round < self.max_round:
            self.now_round += 1
            print(f"\n{'='*50}")
            print(f"Starting Round {self.now_round}/{self.max_round}")
            print(f"{'='*50}")

            self.clinician_responses = []

            def process_clinician(clinician_agent: ClinicianAgent, idx: int) -> Tuple[int, int]:
                """
                Process a single ClinicianAgent collaboration.

                Args:
                    clinician_agent: ClinicianAgent instance
                    idx: Clinician index

                Returns:
                    (prompt_tokens, completion_tokens)
                """
                review_file = f"{self.save_dir}/clinician{idx + 1}_review.json"
                try:
                    with open(review_file, 'r') as f:
                        last_review = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {review_file}: {e}")
                    last_review = {'logit': 0.5, 'conf_logit': 0.5}

                response, messages, prompt_token, completion_token = clinician_agent.collaboration(
                    supervisor_opinion=self.supervisor_opinion,
                    supervisor_report=self.supervisor_report,
                    last_review=last_review
                )

                self.clinician_responses.append(response)

                round_file = f"{self.save_dir}/clinician{idx + 1}_round{self.now_round}_review.json"
                with open(round_file, "w") as f:
                    json.dump(response, f, indent=2)

                if len(messages) > 1:
                    prompt_file = f"{self.save_dir}/clinician{idx + 1}_round{self.now_round}_review_userprompt.txt"
                    with open(prompt_file, "w") as f:
                        f.write(messages[1].get("content", ""))

                return prompt_token, completion_token

            with ThreadPoolExecutor(max_workers=self.clinician_num) as executor:
                futures = [
                    executor.submit(process_clinician, clinician_agent, i)
                    for i, clinician_agent in enumerate(self.clinician_agents)
                ]

                for future in futures:
                    try:
                        prompt_token, completion_token = future.result()
                        prompt_tokens += prompt_token
                        completion_tokens += completion_token
                    except Exception as e:
                        print(f"Error in doctor process: {e}")

            print(f"\nSupervisorAgent is summarizing round {self.now_round}...")
            action, summary_content, messages, prompt_token, completion_token = \
                self.supervisor_agent.summary(self.clinician_responses, is_initial=False)

            summary_file = f"{self.save_dir}/supervisor_round{self.now_round}_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary_content, f, indent=2)

            messages_file = f"{self.save_dir}/supervisor_round{self.now_round}_summary_messages.json"
            with open(messages_file, "w") as f:
                json.dump(messages, f, indent=2)

            prompt_tokens += prompt_token
            completion_tokens += completion_token

            self.supervisor_opinion = summary_content.get("answer", self.supervisor_opinion)
            self.supervisor_report = summary_content.get("report", self.supervisor_report)

            print(f"\nRound {self.now_round} completed.")
            print(f"Action: {'Continue' if action == 1 else 'Finalize'}")

            if action == 0:
                print(f"\n{'='*50}")
                print("Consensus reached! Finalizing report...")
                print(f"{'='*50}")
                break
            else:
                print("Continuing to next round...")

        if self.now_round == self.max_round:
            print(f"\n{'='*50}")
            print(f"Warning: No agreement reached after {self.max_round} rounds.")
            print("Using last round's report as final.")
            print(f"{'='*50}")

        final_round = self.now_round
        self.now_round = 0

        return self.supervisor_report, final_round, prompt_tokens, completion_tokens

    def get_discussion_summary(self) -> Dict:
        """
        Get discussion summary.

        Returns statistics and key results from the multi-round discussion.

        Returns:
            Discussion summary dict
        """
        return {
            "total_rounds": self.now_round,
            "max_rounds": self.max_round,
            "clinician_num": self.clinician_num,
            "final_opinion": self.supervisor_opinion,
            "discussion_complete": self.now_round < self.max_round or self.now_round == 0,
        }

    def save_intermediate_results(self) -> None:
        """
        Save intermediate results.

        Persists all intermediate results to the specified directory.
        """
        state_file = f"{self.save_dir}/collaboration_state.json"
        state = {
            "now_round": self.now_round,
            "max_round": self.max_round,
            "supervisor_opinion": self.supervisor_opinion,
            "supervisor_report": self.supervisor_report,
            "clinician_responses_count": len(self.clinician_responses),
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)


# Backwards-compatibility alias
Collaboration = CollaborationEngine
