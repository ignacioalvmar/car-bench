# DEPRECTAED: Please use analyze_results_v2.py for the latest version of this analysis script. This file is kept for analysing legacy results.

#!/usr/bin/env python3
"""
Comprehensive analysis script for tau-bench results.

This script analyzes benchmark results and saves detailed metrics.
Use generate_figures.py to create visualizations from saved results.

Metrics calculated:
1. Pass^k scores (all k trials must succeed)
2. Pass@k scores (at least one of k trials must succeed)
3. RDI (Retry Dependency Index) - sum of gaps between Pass@k and Pass^k
4. Detailed Pass^1 analysis
5. Cost and latency metadata
6. Multi-model comparison tables

Usage Examples:
--------------

# Single model analysis (you'll be prompted for model name)
python analyze_results.py results/model1.json

# Multi-model comparison (you'll be prompted for each model name)
python analyze_results.py results/model1.json results/model2.json results/model3.json

# Save to custom directory
python analyze_results.py results/*.json --output my_analysis

# Exclude specific tasks from analysis (e.g., tasks with errors)
python analyze_results.py results/*.json --exclude-tasks 5,23,72

# Individual analysis only (no comparison tables)
python analyze_results.py results/*.json --individual-only

Note: Results are saved to JSON and CSV. Use generate_figures.py to create visualizations.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_results(file_path: str) -> Tuple[List[Dict], str]:
    """Load results from JSON file and prompt for model name."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        print(f"\nLoaded {len(data)} entries from: {file_path}")

        # Prompt user for model name
        while True:
            model_name = input(f"Please enter the model name for this file: ").strip()
            if model_name:
                break
            print("Model name cannot be empty. Please try again.")

        print(f"✓ Assigned model name: {model_name}")
        return data, model_name

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)


def organize_data_by_task_and_trial(data: List[Dict], exclude_task_ids: List[int] = None) -> Dict[int, Dict[int, Dict]]:
    """Organize data by task_id and trial.
    
    Args:
        data: List of task entries
        exclude_task_ids: Optional list of task IDs to exclude from analysis
    
    Returns:
        Dictionary mapping task_id -> trial -> entry
    """
    organized = defaultdict(dict)
    exclude_set = set(exclude_task_ids) if exclude_task_ids else set()

    for entry in data:
        task_id = entry["task_id"]
        
        # Skip excluded tasks
        if task_id in exclude_set:
            continue
            
        trial = entry["trial"]
        organized[task_id][trial] = entry

    return dict(organized)


def calculate_pass_power_k_scores(
    organized_data: Dict[int, Dict[int, Dict]], max_trials: int
) -> Dict[str, float]:
    """
    Calculate Pass^k scores where all k trials must succeed.
    Score is 1 only if ALL k trials have reward=1.
    """
    scores = {}

    for n in range(1, max_trials + 1):
        successful_tasks = 0
        valid_tasks = 0

        for task_id, trials in organized_data.items():
            # Check if we have at least n trials for this task
            if len(trials) >= n:
                valid_tasks += 1
                # Check if all first n trials are successful
                all_successful = True
                for trial_idx in range(n):
                    if trial_idx not in trials or trials[trial_idx]["reward"] != 1.0:
                        all_successful = False
                        break

                if all_successful:
                    successful_tasks += 1

        if valid_tasks > 0:
            scores[f"Pass^{n}"] = successful_tasks / valid_tasks
        else:
            scores[f"Pass^{n}"] = 0.0

    return scores


def calculate_pass_at_k_scores(
    organized_data: Dict[int, Dict[int, Dict]], max_trials: int
) -> Dict[str, float]:
    """
    Calculate Pass@k scores where at least one of k trials must succeed.
    Score is 1 if ANY of the k trials has reward=1.
    """
    scores = {}

    for n in range(1, max_trials + 1):
        successful_tasks = 0
        valid_tasks = 0

        for task_id, trials in organized_data.items():
            # Check if we have at least n trials for this task
            if len(trials) >= n:
                valid_tasks += 1
                # Check if any of the first n trials is successful
                any_successful = False
                for trial_idx in range(n):
                    if trial_idx in trials and trials[trial_idx]["reward"] == 1.0:
                        any_successful = True
                        break

                if any_successful:
                    successful_tasks += 1

        if valid_tasks > 0:
            scores[f"Pass@{n}"] = successful_tasks / valid_tasks
        else:
            scores[f"Pass@{n}"] = 0.0

    return scores


def has_task_error(entry: Dict) -> bool:
    """Check if a task entry has an error (failed to run properly)."""
    return "error" in entry.get("info", {})


def analyze_error_tasks(organized_data: Dict[int, Dict[int, Dict]]) -> Dict[str, Any]:
    """Analyze tasks that failed with errors.
    
    Returns counts and details of tasks that encountered errors during execution.
    These are counted as failures (reward=0) in all metrics.
    """
    error_tasks_by_id = {}
    error_count_by_trial = defaultdict(int)
    total_errors = 0
    
    for task_id, trials in organized_data.items():
        for trial_idx, entry in trials.items():
            if has_task_error(entry):
                total_errors += 1
                error_count_by_trial[trial_idx] += 1
                
                # Store error info for the first occurrence of each task
                if task_id not in error_tasks_by_id:
                    error_msg = entry["info"].get("error", "Unknown error")
                    error_tasks_by_id[task_id] = {
                        "task_id": task_id,
                        "first_trial_with_error": trial_idx,
                        "error_message": error_msg[:200],  # Truncate long errors
                    }
    
    # Count tasks with errors in trial 0 (Pass^1)
    trial_0_errors = error_count_by_trial.get(0, 0)
    
    return {
        "total_error_occurrences": total_errors,
        "unique_tasks_with_errors": len(error_tasks_by_id),
        "trial_0_errors": trial_0_errors,
        "error_count_by_trial": dict(error_count_by_trial),
        "error_task_ids": sorted(error_tasks_by_id.keys()),
    }


def analyze_pass1_by_action_count(
    organized_data: Dict[int, Dict[int, Dict]],
) -> Dict[int, Dict[str, Any]]:
    """Analyze pass^1 rate categorized by number of ground truth actions.
    
    Note: Tasks with errors are counted as failures but excluded from action count breakdown
    since they may not have task metadata available.
    """
    action_count_analysis = defaultdict(lambda: {"successful": 0, "total": 0})

    for task_id, trials in organized_data.items():
        if 0 in trials:  # Only consider first trial for pass^1
            entry = trials[0]
            reward = entry["reward"]
            
            # Skip error tasks from action count analysis (no task metadata available)
            if has_task_error(entry):
                continue
            
            # Safely access task actions
            try:
                num_actions = len(entry["info"]["task"]["actions"])
            except (KeyError, TypeError):
                # Task metadata not available, skip this task
                continue

            action_count_analysis[num_actions]["total"] += 1
            if reward == 1.0:
                action_count_analysis[num_actions]["successful"] += 1

    # Calculate rates
    result = {}
    for num_actions, stats in action_count_analysis.items():
        rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0.0
        result[num_actions] = {
            "pass_rate": rate,
            "successful": stats["successful"],
            "total": stats["total"],
        }

    return dict(result)


def analyze_pass1_by_reward_components(
    organized_data: Dict[int, Dict[int, Dict]],
) -> Dict[str, Dict[str, Any]]:
    """Analyze distribution of reward component values across all tasks."""
    component_analysis = defaultdict(lambda: defaultdict(int))
    total_tasks = 0

    for task_id, trials in organized_data.items():
        if 0 in trials:  # Only consider first trial for pass^1
            entry = trials[0]
            # Safely access reward_info; skip reward-based analysis if missing
            info_block = entry.get("info") or {}
            reward_info_container = info_block.get("reward_info") or {}
            reward_info = reward_info_container.get("info")

            if not isinstance(reward_info, dict):
                continue

            total_tasks += 1

            # Count each reward component value
            for component, component_value in reward_info.items():
                if component.startswith("r_"):  # Only analyze reward components
                    component_analysis[component][component_value] += 1

    # Calculate distribution rates
    result = {}
    for component, value_counts in component_analysis.items():
        result[component] = {}
        for value, count in value_counts.items():
            rate = count / total_tasks if total_tasks > 0 else 0.0
            result[component][str(value)] = {
                "distribution_rate": rate,
                "count": count,
                "total_tasks": total_tasks,
            }

    return dict(result)


def extract_persona_attributes(persona_string: str) -> Dict[str, str]:
    """Extract conversation style and technological proficiency from persona string.
    
    Args:
        persona_string: The persona string from the task
        
    Returns:
        Dictionary with 'conversation_style' and 'tech_proficiency' keys
    """
    attributes = {
        "conversation_style": "unknown",
        "tech_proficiency": "not specified"
    }
    
    # Extract conversation style
    if "Commanding" in persona_string:
        attributes["conversation_style"] = "Commanding"
    elif "Conversational" in persona_string:
        attributes["conversation_style"] = "Conversational"
    elif "Questioning" in persona_string:
        attributes["conversation_style"] = "Questioning"
    
    # Extract technological proficiency
    if "anxious" in persona_string:
        attributes["tech_proficiency"] = "low"
    elif "HVAC" in persona_string:
        attributes["tech_proficiency"] = "high"
    # else remains "not specified"
    
    return attributes


def analyze_pass1_by_persona(
    organized_data: Dict[int, Dict[int, Dict]],
) -> Dict[str, Any]:
    """Analyze Pass^1 rate by user persona attributes (conversation style and tech proficiency).
    
    Returns:
        Dictionary with analysis broken down by conversation_style and tech_proficiency
    """
    # Track statistics by conversation style
    style_stats = defaultdict(lambda: {"successful": 0, "total": 0})
    # Track statistics by tech proficiency
    tech_stats = defaultdict(lambda: {"successful": 0, "total": 0})
    # Track statistics by combination
    combined_stats = defaultdict(lambda: {"successful": 0, "total": 0})
    
    for task_id, trials in organized_data.items():
        if 0 in trials:  # Only consider first trial for pass^1
            entry = trials[0]
            reward = entry["reward"]
            
            # Skip error tasks
            if has_task_error(entry):
                continue
            
            # Extract persona string
            try:
                persona_string = entry["info"]["task"]["persona"]
            except (KeyError, TypeError):
                # Persona not available, skip this task
                continue
            
            # Extract persona attributes
            attributes = extract_persona_attributes(persona_string)
            conversation_style = attributes["conversation_style"]
            tech_proficiency = attributes["tech_proficiency"]
            
            # Update statistics by conversation style
            style_stats[conversation_style]["total"] += 1
            if reward == 1.0:
                style_stats[conversation_style]["successful"] += 1
            
            # Update statistics by tech proficiency
            tech_stats[tech_proficiency]["total"] += 1
            if reward == 1.0:
                tech_stats[tech_proficiency]["successful"] += 1
            
            # Update statistics by combination
            combo_key = f"{conversation_style} + {tech_proficiency}"
            combined_stats[combo_key]["total"] += 1
            if reward == 1.0:
                combined_stats[combo_key]["successful"] += 1
    
    # Calculate pass rates
    result = {
        "by_conversation_style": {},
        "by_tech_proficiency": {},
        "by_combination": {}
    }
    
    for style, stats in style_stats.items():
        rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0.0
        result["by_conversation_style"][style] = {
            "pass_rate": rate,
            "successful": stats["successful"],
            "total": stats["total"],
        }
    
    for tech, stats in tech_stats.items():
        rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0.0
        result["by_tech_proficiency"][tech] = {
            "pass_rate": rate,
            "successful": stats["successful"],
            "total": stats["total"],
        }
    
    for combo, stats in combined_stats.items():
        rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0.0
        result["by_combination"][combo] = {
            "pass_rate": rate,
            "successful": stats["successful"],
            "total": stats["total"],
        }
    
    return result


def analyze_reward_component_correlations(
    organized_data: Dict[int, Dict[int, Dict]],
) -> Dict[str, Any]:
    """Analyze correlations between reward components to identify co-occurring failure patterns."""

    # Extract component data for each task
    component_data = []
    component_names = []

    for task_id, trials in organized_data.items():
        if 0 in trials:  # Only consider first trial for pass^1
            entry = trials[0]
            # Safely access reward_info; skip reward-based analysis if missing
            info_block = entry.get("info") or {}
            reward_info_container = info_block.get("reward_info") or {}
            reward_info = reward_info_container.get("info")

            if not isinstance(reward_info, dict):
                continue

            task_components = {}
            for component, value in reward_info.items():
                if (
                    component.startswith("r_") and component != "r_outputs"
                ):  # Skip r_outputs (often None)
                    # Convert to numeric (0.0 -> 0, 1.0 -> 1, None -> 0)
                    if value is None:
                        numeric_value = 0
                    else:
                        try:
                            numeric_value = int(value)
                        except Exception:
                            # Skip values that cannot be coerced to int
                            continue
                    task_components[component] = numeric_value

            component_data.append(task_components)

    if not component_data:
        return {"correlation_matrix": {}, "failure_patterns": {}, "component_names": []}

    # Get consistent component names
    component_names = sorted(set().union(*(task.keys() for task in component_data)))

    # Create matrix for correlation calculation
    matrix_data = []
    for task in component_data:
        row = [task.get(comp, 0) for comp in component_names]
        matrix_data.append(row)

    # Convert to pandas DataFrame for easier correlation calculation
    import pandas as pd

    df = pd.DataFrame(matrix_data, columns=component_names)

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Analyze specific failure patterns
    failure_patterns = {}

    for i, comp1 in enumerate(component_names):
        failure_patterns[comp1] = {}

        for j, comp2 in enumerate(component_names):
            if i != j:  # Don't compare component with itself
                # When comp1 fails (=0), what happens to comp2?
                comp1_failures = df[df[comp1] == 0]
                if len(comp1_failures) > 0:
                    comp2_also_fails = len(comp1_failures[comp1_failures[comp2] == 0])
                    comp2_failure_rate = comp2_also_fails / len(comp1_failures)

                    failure_patterns[comp1][comp2] = {
                        "when_comp1_fails_comp2_also_fails_count": comp2_also_fails,
                        "when_comp1_fails_total": len(comp1_failures),
                        "when_comp1_fails_comp2_failure_rate": comp2_failure_rate,
                        "correlation": correlation_matrix.loc[comp1, comp2],
                    }

    # Find interesting patterns (high co-failure rates)
    interesting_patterns = []
    for comp1, comp2_data in failure_patterns.items():
        for comp2, stats in comp2_data.items():
            if (
                stats["when_comp1_fails_comp2_failure_rate"] > 0.5
                and stats["when_comp1_fails_total"] >= 5
            ):  # At least 5 failures and 70%+ co-failure
                interesting_patterns.append(
                    {
                        "primary_failure": comp1,
                        "secondary_failure": comp2,
                        "co_failure_rate": stats["when_comp1_fails_comp2_failure_rate"],
                        "sample_size": stats["when_comp1_fails_total"],
                        "correlation": stats["correlation"],
                    }
                )

    # Sort by co-failure rate
    interesting_patterns.sort(key=lambda x: x["co_failure_rate"], reverse=True)

    return {
        "correlation_matrix": correlation_matrix.to_dict(),
        "failure_patterns": failure_patterns,
        "interesting_patterns": interesting_patterns,
        "component_names": component_names,
        "total_tasks": len(component_data),
    }


def calculate_metadata(organized_data: Dict[int, Dict[int, Dict]]) -> Dict[str, Any]:
    """Calculate cost and latency metadata.
    
    Note: Error tasks are excluded from latency calculations since they may not have
    complete metadata, but are included in cost calculations with 0 cost.
    """
    trial_1_entries = []
    trial_1_valid_entries = []  # Entries without errors

    # Collect all trial 1 entries
    for task_id, trials in organized_data.items():
        if 0 in trials:  # Trial 0 is the first trial
            entry = trials[0]
            trial_1_entries.append(entry)
            # Separate valid entries (without errors) for latency calculation
            if not has_task_error(entry):
                trial_1_valid_entries.append(entry)

    if not trial_1_entries:
        return {}

    # Calculate costs (include error tasks with 0 cost)
    total_user_cost = sum(
        (
            entry.get("info", {}).get("user_cost")
            if entry.get("info", {}).get("user_cost")
            else 0
        )
        for entry in trial_1_entries
    )
    total_agent_cost = sum(
        (
            entry.get("info", {}).get("total_agent_cost")
            if entry.get("info", {}).get("total_agent_cost")
            else 0
        )
        for entry in trial_1_entries
    )
    total_cost = total_user_cost + total_agent_cost

    # Calculate latency metrics (only for valid entries)
    total_llm_latency = sum(
        entry.get("info", {}).get("total_llm_induced_latency_ms", 0)
        for entry in trial_1_valid_entries
    )

    # Calculate latency per step (assistant messages in trajectory)
    latency_per_task_per_step = []
    latency_per_turn_all_tasks = []

    for entry in trial_1_valid_entries:
        traj = entry.get("traj", [])
        assistant_messages = sum(1 for msg in traj if msg.get("role") == "assistant")

        total_latency = entry.get("info", {}).get("total_llm_induced_latency_ms")
        if assistant_messages > 0 and total_latency is not None:
            task_latency_per_step = total_latency / assistant_messages
            latency_per_task_per_step.append(task_latency_per_step)

        # Per turn latency (already calculated in the data)
        avg_latency = entry.get("info", {}).get("average_llm_induced_latency_per_turn_ms")
        if avg_latency is not None:
            latency_per_turn_all_tasks.append(avg_latency)

    num_tasks = len(trial_1_entries)
    num_valid_tasks = len(trial_1_valid_entries)

    return {
        "total_user_cost_all_trial1": total_user_cost,
        "total_user_cost_per_task": total_user_cost / num_tasks if num_tasks > 0 else 0,
        "total_agent_cost_all_trial1": total_agent_cost,
        "total_agent_cost_per_task": (
            total_agent_cost / num_tasks if num_tasks > 0 else 0
        ),
        "total_cost_all_trial1": total_cost,
        "total_cost_per_task": (total_cost / num_tasks if num_tasks > 0 else 0),
        "total_llm_latency_per_task_ms": (
            total_llm_latency / num_valid_tasks if num_valid_tasks > 0 else 0
        ),
        "avg_llm_latency_per_step_ms": (
            np.mean(latency_per_task_per_step) if latency_per_task_per_step else 0
        ),
        "avg_llm_latency_per_turn_ms": (
            np.mean(latency_per_turn_all_tasks) if latency_per_turn_all_tasks else 0
        ),
        "num_tasks_analyzed": num_tasks,
        "num_valid_tasks_for_latency": num_valid_tasks,
    }


def categorize_tasks_by_success_count(
    organized_data: Dict[int, Dict[int, Dict]], max_trials: int
) -> Dict[str, Any]:
    """Categorize tasks by the number of successful trials.

    Returns a structure useful for explaining divergence between Pass^k and Pass@k.

    Definitions:
      success_count: number of trials (among those recorded) with reward == 1.0
      category buckets: 0,1,2,...,max_trials ("always succeeded" == success_count == trials_attempted == max_trials)

    Notes:
      - If some tasks have fewer than max_trials attempts, they are still counted; a task is marked 'incomplete' if attempts < max_trials.
      - This lets you see if divergence between Pass^k and Pass@k is driven by many single-shot successes or instability across attempts.
    """

    distribution = {}
    # success_count -> list of task_ids
    buckets: Dict[int, List[int]] = defaultdict(list)
    incomplete_task_ids: List[int] = []

    for task_id, trials in organized_data.items():
        attempts = len(trials)
        success_count = sum(1 for t in trials.values() if t.get("reward") == 1.0)
        buckets[success_count].append(task_id)
        if attempts < max_trials:
            incomplete_task_ids.append(task_id)

    total_tasks = sum(len(v) for v in buckets.values())

    for success_count in range(0, max_trials + 1):
        task_ids = buckets.get(success_count, [])
        count = len(task_ids)
        percentage = (count / total_tasks) * 100 if total_tasks else 0.0
        distribution[success_count] = {
            "task_count": count,
            "percentage": percentage,
            "task_ids": task_ids,
        }

    # Derived summary metrics to help interpret divergence
    # Instability proxy: tasks with partial success (between 1 and max_trials-1)
    partial_success_tasks = sum(
        distribution[c]["task_count"] for c in range(1, max_trials)
    )
    always_failed = distribution.get(0, {}).get("task_count", 0)
    always_succeeded = distribution.get(max_trials, {}).get("task_count", 0)
    summary = {
        "total_tasks": total_tasks,
        "max_trials": max_trials,
        "always_failed_tasks": always_failed,
        "always_failed_pct": (always_failed / total_tasks * 100) if total_tasks else 0,
        "always_succeeded_tasks": always_succeeded,
        "always_succeeded_pct": (always_succeeded / total_tasks * 100)
        if total_tasks
        else 0,
        "partial_success_tasks": partial_success_tasks,
        "partial_success_pct": (partial_success_tasks / total_tasks * 100)
        if total_tasks
        else 0,
        "incomplete_tasks": len(incomplete_task_ids),
    }

    return {"distribution": distribution, "summary": summary}


def calculate_rdi(
    pass_power_k_scores: Dict[str, float], pass_at_k_scores: Dict[str, float]
) -> float:
    """Calculate Retry Dependency Index (RDI).
    
    RDI = 1/K * Σ(Pass@k - Pass^k) for k from 1 to K
    
    Measures how much the model benefits from retries.
    Higher RDI indicates more inconsistency and greater benefit from multiple attempts.
    """
    rdi = 0.0
    
    # Extract k values from Pass^k keys
    k_values = []
    for key in pass_power_k_scores.keys():
        if key.startswith("Pass^"):
            k = int(key.replace("Pass^", ""))
            k_values.append(k)
    
    # Sum gaps
    for k in k_values:
        pass_at = pass_at_k_scores.get(f"Pass@{k}", 0.0)
        pass_power = pass_power_k_scores.get(f"Pass^{k}", 0.0)
        rdi += (pass_at - pass_power)
    
    return rdi / len(k_values)


def analyze_single_model(data: List[Dict], model_name: str, exclude_task_ids: List[int] = None) -> Dict[str, Any]:
    """Analyze results for a single model.
    
    Args:
        data: List of task entries
        model_name: Name of the model being analyzed
        exclude_task_ids: Optional list of task IDs to exclude from analysis
    
    Returns:
        Dictionary containing all analysis results
    """
    organized_data = organize_data_by_task_and_trial(data, exclude_task_ids)
    max_trials = (
        max(len(trials) for trials in organized_data.values()) if organized_data else 1
    )

    success_count_categorization = categorize_tasks_by_success_count(
        organized_data, max_trials
    )

    pass_power_k_scores = calculate_pass_power_k_scores(organized_data, max_trials)
    pass_at_k_scores = calculate_pass_at_k_scores(organized_data, max_trials)
    rdi = calculate_rdi(pass_power_k_scores, pass_at_k_scores)
    error_analysis = analyze_error_tasks(organized_data)
    
    # -- Per-subtype pass scores for hallucination and disambiguation tasks --
    def filter_organized_data(
        organized: Dict[int, Dict[int, Dict]], predicate
    ) -> Dict[int, Dict[int, Dict]]:
        filtered: Dict[int, Dict[int, Dict]] = {}
        for task_id, trials in organized.items():
            # Use trial 0 if available, else any existing trial for metadata
            entry = trials.get(0)
            if entry is None and trials:
                entry = next(iter(trials.values()))
            if entry is None:
                continue
            try:
                if predicate(entry):
                    filtered[task_id] = trials
            except Exception:
                # Skip tasks where predicate access fails
                continue
        return filtered
    
    def compute_pass_scores_for_subset(
        organized_subset: Dict[int, Dict[int, Dict]]
    ) -> Dict[str, Dict[str, float]]:
        # If subset empty, return empty dicts
        if not organized_subset:
            return {"pass_power_k_scores": {}, "pass_at_k_scores": {}}
        return {
            "pass_power_k_scores": calculate_pass_power_k_scores(organized_subset, max_trials),
            "pass_at_k_scores": calculate_pass_at_k_scores(organized_subset, max_trials),
        }
    
    # Hallucination type subsets (based on task_type)
    hallucination_types = [
        "hallucination_missing_tool",
        "hallucination_missing_tool_parameter",
        "hallucination_missing_tool_response",
    ]
    hallucination_type_pass_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for h_type in hallucination_types:
        subset = filter_organized_data(
            organized_data,
            lambda e: (e.get("info") or {}).get("task", {}).get("task_type") == h_type,
        )
        hallucination_type_pass_scores[h_type] = compute_pass_scores_for_subset(subset)
    
    # Disambiguation type subsets (based on which disambiguation element is present)
    # One of the two should be non-null per task per user's spec
    disambiguation_type_pass_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    # internal
    disamb_internal_subset = filter_organized_data(
        organized_data,
        lambda e: (e.get("info") or {}).get("task", {}).get("disambiguation_element_internal") is not None,
    )
    disambiguation_type_pass_scores["internal"] = compute_pass_scores_for_subset(
        disamb_internal_subset
    )
    # user
    disamb_user_subset = filter_organized_data(
        organized_data,
        lambda e: (e.get("info") or {}).get("task", {}).get("disambiguation_element_user") is not None,
    )
    disambiguation_type_pass_scores["user"] = compute_pass_scores_for_subset(
        disamb_user_subset
    )
    
    return {
        "model_name": model_name,
        "pass_power_k_scores": pass_power_k_scores,
        "pass_at_k_scores": pass_at_k_scores,
        "rdi": rdi,
        "error_analysis": error_analysis,
        "action_count_analysis": analyze_pass1_by_action_count(organized_data),
        "persona_analysis": analyze_pass1_by_persona(organized_data),
        "reward_component_analysis": analyze_pass1_by_reward_components(organized_data),
        "reward_component_correlations": analyze_reward_component_correlations(
            organized_data
        ),
        "hallucination_type_pass_scores": hallucination_type_pass_scores,
        "disambiguation_type_pass_scores": disambiguation_type_pass_scores,
        "metadata": calculate_metadata(organized_data),
        "max_trials": max_trials,
        "success_count_categorization": success_count_categorization,
    }


def create_comparison_tables(
    all_results: Dict[str, Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """Create comparison tables across all models."""
    comparison_tables = {}

    # Pass scores comparison
    pass_scores_data = []
    for model_name, results in all_results.items():
        for metric, score in {
            **results["pass_power_k_scores"],
            **results["pass_at_k_scores"],
        }.items():
            pass_scores_data.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "score": score,
                    "percentage": score * 100,
                }
            )
        # Add RDI
        pass_scores_data.append(
            {
                "model": model_name,
                "metric": "RDI",
                "score": results["rdi"],
                "percentage": results["rdi"] * 100,
            }
        )

    comparison_tables["pass_scores"] = (
        pd.DataFrame(pass_scores_data)
        .pivot(index="metric", columns="model", values="score")
        .fillna(0)
    )

    # Metadata comparison
    metadata_data = []
    for model_name, results in all_results.items():
        metadata = results["metadata"].copy()
        metadata["model"] = model_name
        metadata_data.append(metadata)

    comparison_tables["metadata"] = pd.DataFrame(metadata_data).set_index("model")

    # Action count comparison (simplified)
    action_summary = []
    for model_name, results in all_results.items():
        action_analysis = results["action_count_analysis"]
        total_tasks = sum(stats["total"] for stats in action_analysis.values())
        total_successful = sum(
            stats["successful"] for stats in action_analysis.values()
        )
        overall_rate = total_successful / total_tasks if total_tasks > 0 else 0

        action_summary.append(
            {
                "model": model_name,
                "overall_pass_rate": overall_rate,
                "total_tasks": total_tasks,
                "total_successful": total_successful,
            }
        )

    comparison_tables["action_summary"] = pd.DataFrame(action_summary).set_index(
        "model"
    )

    # Persona analysis comparison - by conversation style
    persona_style_data = []
    for model_name, results in all_results.items():
        persona_analysis = results["persona_analysis"]
        for style, stats in persona_analysis["by_conversation_style"].items():
            persona_style_data.append(
                {
                    "model": model_name,
                    "conversation_style": style,
                    "pass_rate": stats["pass_rate"],
                    "successful": stats["successful"],
                    "total": stats["total"],
                }
            )
    
    if persona_style_data:
        comparison_tables["persona_by_style"] = pd.DataFrame(persona_style_data)
    
    # Persona analysis comparison - by tech proficiency
    persona_tech_data = []
    for model_name, results in all_results.items():
        persona_analysis = results["persona_analysis"]
        for tech, stats in persona_analysis["by_tech_proficiency"].items():
            persona_tech_data.append(
                {
                    "model": model_name,
                    "tech_proficiency": tech,
                    "pass_rate": stats["pass_rate"],
                    "successful": stats["successful"],
                    "total": stats["total"],
                }
            )
    
    if persona_tech_data:
        comparison_tables["persona_by_tech"] = pd.DataFrame(persona_tech_data)

    # Per-type comparison tables
    # Hallucination types
    halluc_rows = []
    for model_name, results in all_results.items():
        ht = results.get("hallucination_type_pass_scores") or {}
        for h_type, scores in ht.items():
            for metric, value in {**scores.get("pass_power_k_scores", {}), **scores.get("pass_at_k_scores", {})}.items():
                halluc_rows.append({
                    "model": model_name,
                    "type": h_type,
                    "metric": metric,
                    "score": value,
                    "percentage": value * 100,
                })
    if halluc_rows:
        comparison_tables["hallucination_type_pass_scores"] = pd.DataFrame(halluc_rows)

    # Disambiguation types
    disamb_rows = []
    for model_name, results in all_results.items():
        dt = results.get("disambiguation_type_pass_scores") or {}
        for d_type, scores in dt.items():
            for metric, value in {**scores.get("pass_power_k_scores", {}), **scores.get("pass_at_k_scores", {})}.items():
                disamb_rows.append({
                    "model": model_name,
                    "type": d_type,
                    "metric": metric,
                    "score": value,
                    "percentage": value * 100,
                })
    if disamb_rows:
        comparison_tables["disambiguation_type_pass_scores"] = pd.DataFrame(disamb_rows)

    return comparison_tables


def analyze_task_level_performance(
    all_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze performance at the task level across all models."""

    # Collect task-level results for each model
    task_results = defaultdict(dict)  # task_id -> {model: success_rate}
    task_metadata = {}  # task_id -> task info (actions, etc.)

    for model_name, results in all_results.items():
        # Get organized data to extract task-level performance
        data = []
        # We need to reconstruct the data or get it from somewhere
        # For now, let's extract from the action_count_analysis and other sources

        # We need to get back to the original organized data
        # Let's add a method to store this info
        pass

    # Since we don't have direct access to task-level data in the current structure,
    # let's modify the analyze_single_model to include task-level results
    return {}


def analyze_task_level_performance_from_data(
    all_organized_data: Dict[str, Dict[int, Dict[int, Dict]]],
) -> Dict[str, Any]:
    """Analyze performance at the task level across all models."""

    # Collect all unique task IDs across all models
    all_task_ids = set()
    for model_data in all_organized_data.values():
        all_task_ids.update(model_data.keys())

    all_task_ids = sorted(all_task_ids)

    # Collect task-level results for each model
    task_results = {}  # task_id -> {model: (success, num_actions, task_info)}

    for task_id in all_task_ids:
        task_results[task_id] = {}

        for model_name, organized_data in all_organized_data.items():
            if task_id in organized_data and 0 in organized_data[task_id]:
                # Get trial 0 (first trial) results
                entry = organized_data[task_id][0]
                success = entry["reward"] == 1.0
                
                # Skip tasks with errors (no task metadata available)
                if has_task_error(entry):
                    continue
                
                # Safely access task metadata
                try:
                    num_actions = len(entry["info"]["task"]["actions"])
                    task_type = entry["info"]["task"].get("task_type", "unknown")
                    has_disambiguation = "disambiguation_element_user" in entry["info"]["task"]
                except (KeyError, TypeError):
                    # Task metadata not available, skip this task
                    continue

                task_results[task_id][model_name] = {
                    "success": success,
                    "num_actions": num_actions,
                    "task_info": {
                        "task_type": task_type,
                        "has_disambiguation": has_disambiguation,
                    },
                }

    # Analyze patterns
    unsolved_tasks = []  # Tasks no model solved
    mixed_results_tasks = []  # Tasks some models solved, others didn't
    universally_solved_tasks = []  # Tasks all models solved

    model_names = list(all_organized_data.keys())

    for task_id, model_results in task_results.items():
        if not model_results:  # No model attempted this task
            continue

        successes = [result["success"] for result in model_results.values()]
        success_count = sum(successes)
        total_models = len(successes)

        if success_count == 0:
            # No model solved this task
            task_info = list(model_results.values())[0]  # Get info from any model
            unsolved_tasks.append(
                {
                    "task_id": task_id,
                    "num_actions": task_info["num_actions"],
                    "task_type": task_info["task_info"]["task_type"],
                    "has_disambiguation": task_info["task_info"]["has_disambiguation"],
                    "attempted_by": list(model_results.keys()),
                }
            )
        elif success_count == total_models:
            # All models solved this task
            task_info = list(model_results.values())[0]
            universally_solved_tasks.append(
                {
                    "task_id": task_id,
                    "num_actions": task_info["num_actions"],
                    "task_type": task_info["task_info"]["task_type"],
                    "has_disambiguation": task_info["task_info"]["has_disambiguation"],
                }
            )
        else:
            # Mixed results - some models solved, others didn't
            successful_models = [
                model for model, result in model_results.items() if result["success"]
            ]
            failed_models = [
                model
                for model, result in model_results.items()
                if not result["success"]
            ]
            task_info = list(model_results.values())[0]

            mixed_results_tasks.append(
                {
                    "task_id": task_id,
                    "num_actions": task_info["num_actions"],
                    "task_type": task_info["task_info"]["task_type"],
                    "has_disambiguation": task_info["task_info"]["has_disambiguation"],
                    "successful_models": successful_models,
                    "failed_models": failed_models,
                    "success_rate": success_count / total_models,
                }
            )

    # Sort mixed results by success rate to see most controversial tasks
    mixed_results_tasks.sort(key=lambda x: x["success_rate"])

    # Analyze task characteristics
    def analyze_task_characteristics(tasks, label):
        if not tasks:
            return {}

        action_counts = [task["num_actions"] for task in tasks]
        task_types = [task["task_type"] for task in tasks]
        disambiguation_count = sum(1 for task in tasks if task["has_disambiguation"])

        return {
            "count": len(tasks),
            "avg_actions": np.mean(action_counts) if action_counts else 0,
            "action_range": (
                (min(action_counts), max(action_counts)) if action_counts else (0, 0)
            ),
            "task_types": dict(Counter(task_types)),
            "disambiguation_percentage": (
                disambiguation_count / len(tasks) * 100 if tasks else 0
            ),
        }

    return {
        "unsolved_tasks": unsolved_tasks,
        "mixed_results_tasks": mixed_results_tasks,
        "universally_solved_tasks": universally_solved_tasks,
        "statistics": {
            "unsolved": analyze_task_characteristics(unsolved_tasks, "unsolved"),
            "mixed": analyze_task_characteristics(mixed_results_tasks, "mixed"),
            "universal": analyze_task_characteristics(
                universally_solved_tasks, "universal"
            ),
        },
        "total_tasks": len(task_results),
        "model_names": model_names,
    }


def create_visualizations(all_results: Dict[str, Dict[str, Any]], output_dir: Path):
    """Create comparison visualizations."""
    # Create visualizations subdirectory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    plt.style.use("default")
    fig_size = (12, 8)

    # 1. Performance comparison - Pass^1 and higher k metrics if available
    plt.figure(figsize=fig_size)
    models = list(all_results.keys())

    # Get max trials across all models to determine what metrics to show
    max_trials_across_models = max(
        results["max_trials"] for results in all_results.values()
    )
    highest_k = max_trials_across_models

    if max_trials_across_models == 1:
        # Only Pass^1 available (which equals Pass@1)
        pass_1_scores = [all_results[model]["pass_power_k_scores"].get("Pass^1", 0) for model in models]

        plt.bar(range(len(models)), pass_1_scores, alpha=0.8, color="steelblue")
        plt.xlabel("Model")
        plt.ylabel("Success Rate")
        plt.title("Model Performance Comparison: Pass^1 (Pass Rate)")
        plt.xticks(range(len(models)), models, rotation=45, ha="right")

    else:
        # Show Pass@k vs Pass^k for the highest available k
        pass_at_k_scores = [
            all_results[model]["pass_at_k_scores"].get(f"Pass@{highest_k}", 0)
            for model in models
        ]
        pass_power_k_scores = [
            all_results[model]["pass_power_k_scores"].get(f"Pass^{highest_k}", 0) for model in models
        ]

        x = np.arange(len(models))
        width = 0.35

        plt.bar(
            x - width / 2,
            pass_at_k_scores,
            width,
            label=f"Pass@{highest_k} (any trial succeeds)",
            alpha=0.8,
        )
        plt.bar(
            x + width / 2,
            pass_power_k_scores,
            width,
            label=f"Pass^{highest_k} (all trials succeed)",
            alpha=0.8,
        )

        plt.xlabel("Model")
        plt.ylabel("Success Rate")
        plt.title(f"Model Performance Comparison: Pass@{highest_k} vs Pass^{highest_k}")
        plt.xticks(x, models, rotation=45, ha="right")
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "visualizations" / "pass_scores_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Cost vs Performance scatter plot
    plt.figure(figsize=fig_size)
    # Use total cost (user + agent) per task to avoid skew when one of the components is missing
    costs = []
    for model in models:
        md = all_results[model]["metadata"]
        total_cost_per_task = md.get("total_cost_per_task")
        if total_cost_per_task is None:
            total_cost_per_task = md.get("total_user_cost_per_task", 0) + md.get(
                "total_agent_cost_per_task", 0
            )
        costs.append(total_cost_per_task)

    # Use Pass^{highest_k} scores for performance scaling
    performance = [
        all_results[model]["pass_power_k_scores"].get(f"Pass^{highest_k}", 0) for model in models
    ]

    plt.scatter(costs, performance, s=100, alpha=0.7)
    for i, model in enumerate(models):
        plt.annotate(
            model,
            (costs[i], performance[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.xlabel("Cost per Task ($)")
    plt.ylabel(f"Pass^{highest_k} Success Rate")
    plt.title(f"Cost vs Performance Trade-off (Pass^{highest_k})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "visualizations" / "cost_vs_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Latency vs Performance
    plt.figure(figsize=fig_size)
    latencies = [
        all_results[model]["metadata"]["avg_llm_latency_per_step_ms"]
        for model in models
    ]

    plt.scatter(latencies, performance, s=100, alpha=0.7, color="orange")
    for i, model in enumerate(models):
        plt.annotate(
            model,
            (latencies[i], performance[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.xlabel("Avg LLM Latency per Step (ms)")
    plt.ylabel(f"Pass^{highest_k} Success Rate")
    plt.title(f"Latency vs Performance Trade-off (Pass^{highest_k})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "visualizations" / "latency_vs_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Success rate by action count (line plot)
    plt.figure(figsize=fig_size)

    all_action_counts = set()
    for results in all_results.values():
        all_action_counts.update(results["action_count_analysis"].keys())

    action_counts = sorted(all_action_counts)

    for model_name, results in all_results.items():
        rates = []
        for count in action_counts:
            if count in results["action_count_analysis"]:
                rate = results["action_count_analysis"][count]["pass_rate"]
            else:
                rate = 0
            rates.append(rate)

        plt.plot(action_counts, rates, marker="o", label=model_name, linewidth=2)

    plt.xlabel("Number of Ground Truth Actions")
    plt.ylabel("Pass Rate")
    plt.title("Success Rate by Task Complexity (Number of Actions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "visualizations" / "success_by_action_count.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 5. Component distribution heatmap (for the first model as reference)
    if all_results:
        first_model = list(all_results.keys())[0]
        component_data = all_results[first_model]["reward_component_analysis"]

        # Create heatmap data
        components = sorted(
            [comp for comp in component_data.keys() if comp.startswith("r_")]
        )
        heatmap_data = []

        for component in components:
            values = component_data[component]
            for value, stats in values.items():
                heatmap_data.append(
                    {
                        "component": component,
                        "value": str(value),
                        "percentage": stats["distribution_rate"] * 100,
                    }
                )

        if heatmap_data:
            df_heatmap = (
                pd.DataFrame(heatmap_data)
                .pivot(index="component", columns="value", values="percentage")
                .fillna(0)
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                df_heatmap,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                cbar_kws={"label": "Percentage of Tasks"},
            )
            plt.title(f"Reward Component Distribution ({first_model})")
            plt.tight_layout()
            plt.savefig(
                output_dir / "visualizations" / "component_distribution_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # 8. Component correlation heatmap (for the first model as reference)
    if all_results:
        first_model = list(all_results.keys())[0]
        correlation_data = all_results[first_model]["reward_component_correlations"]

        if correlation_data["correlation_matrix"]:
            correlation_df = pd.DataFrame(correlation_data["correlation_matrix"])

            plt.figure(figsize=(12, 10))
            mask = np.triu(
                np.ones_like(correlation_df, dtype=bool)
            )  # Mask upper triangle
            sns.heatmap(
                correlation_df,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                mask=mask,
                square=True,
                cbar_kws={"label": "Correlation"},
            )
            plt.title(f"Reward Component Failure Correlations ({first_model})")
            plt.tight_layout()
            plt.savefig(
                output_dir / "visualizations" / "component_correlation_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # 6. Cost efficiency (performance per dollar)
    plt.figure(figsize=fig_size)
    cost_efficiency = [
        (performance[i] / costs[i]) if costs[i] and costs[i] > 0 else 0
        for i in range(len(models))
    ]

    bars = plt.bar(range(len(models)), cost_efficiency, alpha=0.8, color="green")
    plt.xlabel("Model")
    plt.ylabel(f"Success Rate per $ Cost (Pass^{highest_k} per $)")
    plt.title(f"Cost Efficiency (Pass^{highest_k} Success Rate per Dollar)")
    plt.xticks(range(len(models)), models, rotation=45, ha="right")

    # Add value labels on bars
    for i, (bar, eff) in enumerate(zip(bars, cost_efficiency)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{eff:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "visualizations" / "cost_efficiency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 7. Multi-metric radar chart (if we have multiple models)
    if len(models) > 1:
        plt.figure(figsize=(10, 10))

        # Metrics for radar chart (normalized to 0-1)
        metrics = [
            f"Pass^{highest_k}",
            f"Cost Efficiency\n(Pass^{highest_k} per $)",
            "Speed\n(1 = fastest)",
            "Consistency\n(Pass^k / Pass@k)",
        ]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        # Pre-compute best values for ratio-style normalization
        # Cost efficiency: normalize by best (max) value so 1 = best, others in (0,1]
        ce_max = max(cost_efficiency) if cost_efficiency else 0
        # Speed: use latency; normalize as best_latency / latency so 1 = fastest
        lat_min = min(latencies) if latencies else 0
        lat_max = max(latencies) if latencies else 0

        # Create a single polar axis
        ax = plt.subplot(111, projection="polar")

        for i, model in enumerate(models):
            # Normalize metrics to 0-1 scale
            pass_power_k_norm = performance[
                i
            ]  # Use Pass^{highest_k} performance directly (already in [0,1])

            if ce_max > 0:
                cost_eff_norm = min(cost_efficiency[i] / ce_max, 1.0)
            else:
                cost_eff_norm = 1.0

            if lat_min > 0 and latencies[i] > 0:
                speed_norm = min(lat_min / latencies[i], 1.0)
            else:
                speed_norm = 1.0

            # Consistency: ratio of Pass^k to Pass@k (only meaningful when k > 1)
            if max_trials_across_models > 1:
                highest_k_val = max_trials_across_models
                pass_at_highest = all_results[model]["pass_at_k_scores"].get(
                    f"Pass@{highest_k_val}", 0
                )
                pass_power_highest = all_results[model]["pass_power_k_scores"].get(f"Pass^{highest_k_val}", 0)
                consistency_norm = (
                    pass_power_highest / max(0.01, pass_at_highest) if pass_at_highest > 0 else 0
                )
            else:
                consistency_norm = 1.0  # Perfect consistency when only 1 trial

            values = [pass_power_k_norm, cost_eff_norm, speed_norm, consistency_norm]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors[i])
            # Draw lines only; no filled area as requested

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])  # Explicit radial ticks
        fastest_s = lat_min / 1000.0 if lat_min else 0
        slowest_s = lat_max / 1000.0 if lat_max else 0
        ax.set_title(
            "Multi-Metric Model Comparison\n"
            f"Speed normalized to fastest; Fastest: {fastest_s:.2f}s, Slowest: {slowest_s:.2f}s; Cost Efficiency: ratio to best",
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(
            output_dir / "visualizations" / "radar_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 9. Pass^k vs Pass@k curves across k values
    if max_trials_across_models > 1:
        plt.figure(figsize=fig_size)
        
        # Get all k values available
        k_values = list(range(1, max_trials_across_models + 1))
        
        # Use a colormap for distinguishing models
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            pass_power_scores = []
            pass_at_scores = []
            
            for k in k_values:
                pass_power = all_results[model]["pass_power_k_scores"].get(f"Pass^{k}", None)
                pass_at = all_results[model]["pass_at_k_scores"].get(f"Pass@{k}", None)
                
                pass_power_scores.append(pass_power if pass_power is not None else 0)
                pass_at_scores.append(pass_at if pass_at is not None else 0)
            
            # Plot Pass^k (solid) and Pass@k (dashed) in the same color
            plt.plot(k_values, pass_power_scores, '-', color=colors[i], 
                    linewidth=2.5, label=f"{model} (Pass^k)", marker='o', markersize=6)
            plt.plot(k_values, pass_at_scores, '--', color=colors[i], 
                    linewidth=2.5, label=f"{model} (Pass@k)", marker='s', markersize=6)
        
        plt.xlabel("k (Number of Trials)", fontsize=12)
        plt.ylabel("Pass Score", fontsize=12)
        plt.title("Pass^k vs Pass@k: Performance Across Multiple Trials", fontsize=14)
        plt.legend(loc='best', fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        plt.xticks(k_values)
        plt.tight_layout()
        plt.savefig(
            output_dir / "visualizations" / "pass_curves_by_k.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"📊 Visualizations saved to visualizations/:")
    print("   - pass_scores_comparison.png")
    print("   - cost_vs_performance.png")
    print("   - latency_vs_performance.png")
    print("   - success_by_action_count.png")
    print("   - component_distribution_heatmap.png")
    print("   - component_correlation_heatmap.png")
    print("   - cost_efficiency.png")
    if len(models) > 1:
        print("   - radar_comparison.png")
        if max_trials_across_models > 1:
            print("   - pass_curves_by_k.png")


def create_task_level_visualizations(task_analysis: Dict[str, Any], output_dir: Path):
    """Create visualizations for task-level analysis."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    fig_size = (12, 8)

    # 1. Task difficulty distribution (pie chart)
    plt.figure(figsize=(10, 8))

    labels = ["Unsolved by all", "Mixed results", "Solved by all"]
    sizes = [
        len(task_analysis["unsolved_tasks"]),
        len(task_analysis["mixed_results_tasks"]),
        len(task_analysis["universally_solved_tasks"]),
    ]
    colors = ["#ff6b6b", "#ffa726", "#66bb6a"]
    explode = (0.05, 0.05, 0.05)

    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    plt.title("Task Difficulty Distribution Across All Models")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(
        viz_dir / "task_difficulty_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Task complexity vs success rate (for mixed results tasks)
    if task_analysis["mixed_results_tasks"]:
        plt.figure(figsize=fig_size)

        mixed_tasks = task_analysis["mixed_results_tasks"]
        action_counts = [task["num_actions"] for task in mixed_tasks]
        success_rates = [task["success_rate"] for task in mixed_tasks]

        plt.scatter(action_counts, success_rates, alpha=0.6, s=60)

        # Add trend line
        if len(action_counts) > 1:
            z = np.polyfit(action_counts, success_rates, 1)
            p = np.poly1d(z)
            plt.plot(sorted(action_counts), p(sorted(action_counts)), "r--", alpha=0.8)

        plt.xlabel("Number of Actions (Task Complexity)")
        plt.ylabel("Success Rate Across Models")
        plt.title("Task Complexity vs Success Rate (Mixed Results Tasks)")
        plt.grid(True, alpha=0.3)

        # Add some task ID annotations for outliers
        for i, task in enumerate(
            mixed_tasks[:5]
        ):  # Annotate first 5 (lowest success rates)
            plt.annotate(
                f"T{task['task_id']}",
                (action_counts[i], success_rates[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(
            viz_dir / "task_complexity_vs_success.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 3. Model performance heatmap (task success matrix)
    if task_analysis["mixed_results_tasks"] and len(task_analysis["model_names"]) > 1:
        # Create a matrix: rows = controversial tasks, columns = models
        controversial_tasks = task_analysis["mixed_results_tasks"][
            :20
        ]  # Top 20 most controversial
        models = task_analysis["model_names"]

        if controversial_tasks:
            # Build success matrix
            matrix_data = []
            task_labels = []

            for task in controversial_tasks:
                row = []
                task_labels.append(f"T{task['task_id']}")

                for model in models:
                    if model in task["successful_models"]:
                        row.append(1)  # Success
                    elif model in task["failed_models"]:
                        row.append(0)  # Failure
                    else:
                        row.append(0.5)  # Not attempted (shouldn't happen)
                matrix_data.append(row)

            plt.figure(
                figsize=(
                    max(8, len(models) * 1.5),
                    max(6, len(controversial_tasks) * 0.4),
                )
            )

            # Create heatmap
            im = plt.imshow(matrix_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

            # Set labels
            plt.xticks(range(len(models)), models, rotation=45, ha="right")
            plt.yticks(range(len(task_labels)), task_labels)
            plt.xlabel("Models")
            plt.ylabel("Tasks (Most Controversial)")
            plt.title("Task Success Matrix: Controversial Tasks Across Models")

            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label("Success (1) / Failure (0)")

            # Add grid
            for i in range(len(task_labels)):
                for j in range(len(models)):
                    value = matrix_data[i][j]
                    color = "white" if value < 0.5 else "black"
                    plt.text(
                        j,
                        i,
                        "✓" if value == 1 else "✗",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

            plt.tight_layout()
            plt.savefig(
                viz_dir / "task_success_matrix.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    # 4. Action count distribution by task category
    plt.figure(figsize=fig_size)

    categories = ["Unsolved", "Mixed Results", "Universal Success"]
    task_groups = [
        task_analysis["unsolved_tasks"],
        task_analysis["mixed_results_tasks"],
        task_analysis["universally_solved_tasks"],
    ]

    action_data = []
    labels = []

    for i, (category, tasks) in enumerate(zip(categories, task_groups)):
        if tasks:
            action_counts = [task["num_actions"] for task in tasks]
            action_data.append(action_counts)
            labels.append(f"{category}\n(n={len(tasks)})")

    if action_data:
        plt.boxplot(action_data, labels=labels)
        plt.ylabel("Number of Actions")
        plt.title("Task Complexity Distribution by Success Category")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            viz_dir / "action_count_by_category.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"\n📊 Task-level visualizations saved to visualizations/:")
    print("   - task_difficulty_distribution.png")
    if task_analysis["mixed_results_tasks"]:
        print("   - task_complexity_vs_success.png")
    if task_analysis["mixed_results_tasks"] and len(task_analysis["model_names"]) > 1:
        print("   - task_success_matrix.png")
    print("   - action_count_by_category.png")


def print_analysis_results(
    pass_power_k_scores: Dict[str, float],
    pass_at_k_scores: Dict[str, float],
    rdi: float,
    error_analysis: Dict[str, Any],
    action_count_analysis: Dict[int, Dict[str, Any]],
    persona_analysis: Dict[str, Any],
    hallucination_type_pass_scores: Dict[str, Dict[str, Dict[str, float]]] | None,
    disambiguation_type_pass_scores: Dict[str, Dict[str, Dict[str, float]]] | None,
    reward_component_analysis: Dict[str, Dict[str, Any]],
    reward_component_correlations: Dict[str, Any],
    metadata: Dict[str, Any],
    success_count_categorization: Dict[str, Any] | None = None,
):
    """Print comprehensive analysis results."""

    print("=" * 80)
    print("TAU-BENCH RESULTS ANALYSIS")
    print("=" * 80)
    
    # Error analysis
    if error_analysis["total_error_occurrences"] > 0:
        print("\n⚠️  ERROR TASKS (Execution Failures)")
        print("-" * 50)
        print(f"Total error occurrences:     {error_analysis['total_error_occurrences']:4d}")
        print(f"Unique tasks with errors:    {error_analysis['unique_tasks_with_errors']:4d}")
        print(f"Errors in trial 0 (Pass^1):  {error_analysis['trial_0_errors']:4d}")
        if len(error_analysis['error_task_ids']) <= 20:
            print(f"Task IDs with errors: {error_analysis['error_task_ids']}")
        else:
            print(f"First 20 task IDs with errors: {error_analysis['error_task_ids'][:20]}")
            print(f"... and {len(error_analysis['error_task_ids']) - 20} more")
        print("\nNote: Error tasks are counted as failures (reward=0) in all metrics.")

    # Pass^k scores
    print("\n📊 PASS^K SCORES (All k trials must succeed)")
    print("-" * 50)
    for metric, score in sorted(pass_power_k_scores.items()):
        print(f"{metric:>10}: {score:6.3f} ({score*100:5.1f}%)")

    # Pass@k scores
    print("\n📊 PASS@K SCORES (At least one of k trials must succeed)")
    print("-" * 50)
    for metric, score in sorted(pass_at_k_scores.items()):
        print(f"{metric:>10}: {score:6.3f} ({score*100:5.1f}%)")
    
    # RDI
    print("\n📊 RETRY DEPENDENCY INDEX (RDI)")
    print("-" * 50)
    print(f"RDI: {rdi:6.3f} (Sum of gaps between Pass@k and Pass^k)")
    print(f"Interpretation: {'High inconsistency - benefits greatly from retries' if rdi > 0.5 else 'Low inconsistency - relatively consistent performance' if rdi > 0.1 else 'Very consistent performance'}")

    # Pass^1 by action count
    print("\n📊 PASS^1 RATE BY NUMBER OF GROUND TRUTH ACTIONS")
    print("-" * 50)
    for num_actions in sorted(action_count_analysis.keys()):
        stats = action_count_analysis[num_actions]
        print(
            f"{num_actions:2d} actions: {stats['pass_rate']:6.3f} ({stats['pass_rate']*100:5.1f}%) "
            f"[{stats['successful']:3d}/{stats['total']:3d}]"
        )

    # Pass^1 by user persona
    print("\n👤 PASS^1 RATE BY USER PERSONA")
    print("-" * 50)
    
    # By conversation style
    print("\nBy Conversation Style:")
    if persona_analysis["by_conversation_style"]:
        for style in sorted(persona_analysis["by_conversation_style"].keys()):
            stats = persona_analysis["by_conversation_style"][style]
            print(
                f"  {style:15s}: {stats['pass_rate']:6.3f} ({stats['pass_rate']*100:5.1f}%) "
                f"[{stats['successful']:3d}/{stats['total']:3d}]"
            )
    else:
        print("  No data available")
    
    # By technological proficiency
    print("\nBy Technological Proficiency:")
    if persona_analysis["by_tech_proficiency"]:
        for tech in sorted(persona_analysis["by_tech_proficiency"].keys()):
            stats = persona_analysis["by_tech_proficiency"][tech]
            print(
                f"  {tech:15s}: {stats['pass_rate']:6.3f} ({stats['pass_rate']*100:5.1f}%) "
                f"[{stats['successful']:3d}/{stats['total']:3d}]"
            )
    else:
        print("  No data available")
    
    # By combination (top combinations only)
    print("\nBy Combination (Conversation Style + Tech Proficiency):")
    if persona_analysis["by_combination"]:
        # Sort by total count to show most common combinations
        sorted_combos = sorted(
            persona_analysis["by_combination"].items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )
        for combo, stats in sorted_combos[:10]:  # Show top 10 combinations
            print(
                f"  {combo:40s}: {stats['pass_rate']:6.3f} ({stats['pass_rate']*100:5.1f}%) "
                f"[{stats['successful']:3d}/{stats['total']:3d}]"
            )
    else:
        print("  No data available")

    # Per-subtype pass scores: Hallucination and Disambiguation
    if hallucination_type_pass_scores:
        print("\n🧪 PASS SCORES BY HALLUCINATION TYPE")
        print("-" * 50)
        for h_type, scores in hallucination_type_pass_scores.items():
            pp = scores.get("pass_power_k_scores", {})
            pa = scores.get("pass_at_k_scores", {})
            if not pp and not pa:
                continue
            print(f"\n{h_type}:")
            for metric, score in sorted(pp.items()):
                print(f"  {metric:>10}: {score:6.3f}")
            for metric, score in sorted(pa.items()):
                print(f"  {metric:>10}: {score:6.3f}")

    if disambiguation_type_pass_scores:
        print("\n🧩 PASS SCORES BY DISAMBIGUATION TYPE")
        print("-" * 50)
        for d_type, scores in disambiguation_type_pass_scores.items():
            pp = scores.get("pass_power_k_scores", {})
            pa = scores.get("pass_at_k_scores", {})
            if not pp and not pa:
                continue
            print(f"\n{d_type}:")
            for metric, score in sorted(pp.items()):
                print(f"  {metric:>10}: {score:6.3f}")
            for metric, score in sorted(pa.items()):
                print(f"  {metric:>10}: {score:6.3f}")

    # Pass^1 by reward components (distribution)
    print("\n📊 REWARD COMPONENT VALUE DISTRIBUTION")
    print("-" * 50)
    for component in sorted(reward_component_analysis.keys()):
        print(f"\n{component}:")
        for value in sorted(reward_component_analysis[component].keys()):
            stats = reward_component_analysis[component][value]
            print(
                f"  {value:>12}: {stats['distribution_rate']:6.3f} ({stats['distribution_rate']*100:5.1f}%) "
                f"[{stats['count']:3d}/{stats['total_tasks']:3d}]"
            )

    # Reward component correlations
    print("\n🔗 REWARD COMPONENT FAILURE CORRELATIONS")
    print("-" * 50)

    if reward_component_correlations["interesting_patterns"]:
        print("High co-failure patterns (>70% when one fails, both fail):")
        for pattern in reward_component_correlations["interesting_patterns"][
            :5
        ]:  # Show top 5
            print(
                f"  {pattern['primary_failure']} → {pattern['secondary_failure']}: "
                f"{pattern['co_failure_rate']*100:5.1f}% co-failure rate "
                f"(r={pattern['correlation']:5.2f}, n={pattern['sample_size']})"
            )
    else:
        print("No strong co-failure patterns detected (>70% threshold)")

    # Show overall component success rates for context
    if reward_component_correlations["component_names"]:
        print(
            f"\nComponent overview ({reward_component_correlations['total_tasks']} tasks):"
        )
        for component in sorted(reward_component_correlations["component_names"]):
            if component in reward_component_analysis:
                success_stats = reward_component_analysis[component].get("1.0", {})
                if success_stats:
                    success_rate = success_stats["distribution_rate"]
                    print(f"  {component}: {success_rate*100:5.1f}% success rate")

    # Metadata
    print("\n💰 COST AND LATENCY METADATA")
    print("-" * 50)
    print(
        f"Total user cost (trial 1):     ${metadata.get('total_user_cost_all_trial1', 0):8.4f}"
    )
    print(
        f"User cost per task:            ${metadata.get('total_user_cost_per_task', 0):8.4f}"
    )
    print(
        f"Total agent cost (trial 1):    ${metadata.get('total_agent_cost_all_trial1', 0):8.4f}"
    )
    print(
        f"Agent cost per task:           ${metadata.get('total_agent_cost_per_task', 0):8.4f}"
    )
    print(
        f"LLM latency per task:          {metadata.get('total_llm_latency_per_task_ms', 0):8.1f} ms"
    )
    print(
        f"Avg LLM latency per step:      {metadata.get('avg_llm_latency_per_step_ms', 0):8.1f} ms"
    )
    print(
        f"Avg LLM latency per turn:      {metadata.get('avg_llm_latency_per_turn_ms', 0):8.1f} ms"
    )
    print(f"Number of tasks analyzed:      {metadata.get('num_tasks_analyzed', 0):8d}")

    # Success count categorization (Pass^k vs Pass@k divergence explanation)
    if success_count_categorization:
        dist = success_count_categorization["distribution"]
        summary = success_count_categorization["summary"]
        print("\n🔍 SUCCESS COUNT DISTRIBUTION (for divergence Pass^k vs Pass@k)")
        print("-" * 50)
        print(
            f"Total tasks: {summary['total_tasks']} | Always failed: {summary['always_failed_tasks']} ({summary['always_failed_pct']:.1f}%) | "
            f"Always succeeded: {summary['always_succeeded_tasks']} ({summary['always_succeeded_pct']:.1f}%) | Partial success: {summary['partial_success_tasks']} ({summary['partial_success_pct']:.1f}%)"
        )
        print("\nSuccess Count -> #Tasks (% of total)")
        for sc in sorted(dist.keys()):
            info = dist[sc]
            print(
                f"  {sc:2d} successes: {info['task_count']:4d} ({info['percentage']:5.1f}%)"
            )
        # Interpretation guidance
        if summary['partial_success_tasks'] > 0:
            print(
                "\nInterpretation: A high partial-success mass indicates instability—many tasks succeed at least once (boosting Pass@k) but not consistently across trials (hurting Pass^k)."
            )
        else:
            print(
                "\nInterpretation: Low or zero partial-success tasks means divergence is mainly due to outright failures rather than instability across retries."
            )


def save_analysis_to_csv(
    pass_power_k_scores: Dict[str, float],
    pass_at_k_scores: Dict[str, float],
    rdi: float,
    error_analysis: Dict[str, Any],
    action_count_analysis: Dict[int, Dict[str, Any]],
    persona_analysis: Dict[str, Any],
    hallucination_type_pass_scores: Dict[str, Dict[str, Dict[str, float]]] | None,
    disambiguation_type_pass_scores: Dict[str, Dict[str, Dict[str, float]]] | None,
    reward_component_analysis: Dict[str, Dict[str, Any]],
    reward_component_correlations: Dict[str, Any],
    metadata: Dict[str, Any],
    success_count_categorization: Dict[str, Any] | None,
    output_path: str,
):
    """Save analysis results to CSV files."""

    output_dir = Path(output_path).parent
    individual_dir = output_dir / "individual_results"
    individual_dir.mkdir(exist_ok=True)
    base_name = Path(output_path).stem
    
    # Error analysis
    if error_analysis["total_error_occurrences"] > 0:
        error_df = pd.DataFrame([{
            "total_error_occurrences": error_analysis["total_error_occurrences"],
            "unique_tasks_with_errors": error_analysis["unique_tasks_with_errors"],
            "trial_0_errors": error_analysis["trial_0_errors"],
            "error_task_ids": ",".join(map(str, error_analysis["error_task_ids"])),
        }])
        error_df.to_csv(f"{individual_dir}/{base_name}_error_analysis.csv", index=False)

    # Pass scores (including RDI)
    pass_scores_data = [
        {"metric": k, "score": v, "percentage": v * 100}
        for k, v in {**pass_power_k_scores, **pass_at_k_scores}.items()
    ]
    pass_scores_data.append({"metric": "RDI", "score": rdi, "percentage": rdi * 100})
    pass_scores_df = pd.DataFrame(pass_scores_data)
    pass_scores_df.to_csv(f"{individual_dir}/{base_name}_pass_scores.csv", index=False)

    # Action count analysis
    action_df = pd.DataFrame(
        [
            {
                "num_actions": num_actions,
                "pass_rate": stats["pass_rate"],
                "pass_percentage": stats["pass_rate"] * 100,
                "successful": stats["successful"],
                "total": stats["total"],
            }
            for num_actions, stats in action_count_analysis.items()
        ]
    )
    action_df.to_csv(
        f"{individual_dir}/{base_name}_action_count_analysis.csv", index=False
    )

    # Per-subtype pass scores CSVs
    if hallucination_type_pass_scores:
        rows = []
        for h_type, scores in hallucination_type_pass_scores.items():
            for metric, value in {**scores.get("pass_power_k_scores", {}), **scores.get("pass_at_k_scores", {})}.items():
                rows.append({"type": h_type, "metric": metric, "score": value, "percentage": value * 100})
        if rows:
            pd.DataFrame(rows).to_csv(
                f"{individual_dir}/{base_name}_hallucination_type_pass_scores.csv",
                index=False,
            )

    if disambiguation_type_pass_scores:
        rows = []
        for d_type, scores in disambiguation_type_pass_scores.items():
            for metric, value in {**scores.get("pass_power_k_scores", {}), **scores.get("pass_at_k_scores", {})}.items():
                rows.append({"type": d_type, "metric": metric, "score": value, "percentage": value * 100})
        if rows:
            pd.DataFrame(rows).to_csv(
                f"{individual_dir}/{base_name}_disambiguation_type_pass_scores.csv",
                index=False,
            )

    # Persona analysis
    # Save by conversation style
    if persona_analysis["by_conversation_style"]:
        style_df = pd.DataFrame(
            [
                {
                    "conversation_style": style,
                    "pass_rate": stats["pass_rate"],
                    "pass_percentage": stats["pass_rate"] * 100,
                    "successful": stats["successful"],
                    "total": stats["total"],
                }
                for style, stats in persona_analysis["by_conversation_style"].items()
            ]
        )
        style_df.to_csv(
            f"{individual_dir}/{base_name}_persona_conversation_style.csv", index=False
        )
    
    # Save by tech proficiency
    if persona_analysis["by_tech_proficiency"]:
        tech_df = pd.DataFrame(
            [
                {
                    "tech_proficiency": tech,
                    "pass_rate": stats["pass_rate"],
                    "pass_percentage": stats["pass_rate"] * 100,
                    "successful": stats["successful"],
                    "total": stats["total"],
                }
                for tech, stats in persona_analysis["by_tech_proficiency"].items()
            ]
        )
        tech_df.to_csv(
            f"{individual_dir}/{base_name}_persona_tech_proficiency.csv", index=False
        )
    
    # Save by combination
    if persona_analysis["by_combination"]:
        combo_df = pd.DataFrame(
            [
                {
                    "combination": combo,
                    "pass_rate": stats["pass_rate"],
                    "pass_percentage": stats["pass_rate"] * 100,
                    "successful": stats["successful"],
                    "total": stats["total"],
                }
                for combo, stats in persona_analysis["by_combination"].items()
            ]
        )
        combo_df.to_csv(
            f"{individual_dir}/{base_name}_persona_combination.csv", index=False
        )

    # Reward component analysis
    component_rows = []
    for component, value_stats in reward_component_analysis.items():
        for value, stats in value_stats.items():
            component_rows.append(
                {
                    "component": component,
                    "value": value,
                    "distribution_rate": stats["distribution_rate"],
                    "distribution_percentage": stats["distribution_rate"] * 100,
                    "count": stats["count"],
                    "total_tasks": stats["total_tasks"],
                }
            )

    component_df = pd.DataFrame(component_rows)
    component_df.to_csv(
        f"{individual_dir}/{base_name}_reward_component_distribution.csv", index=False
    )

    # Metadata
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(f"{individual_dir}/{base_name}_metadata.csv", index=False)

    # Correlation analysis
    if reward_component_correlations["correlation_matrix"]:
        # Save correlation matrix
        correlation_df = pd.DataFrame(
            reward_component_correlations["correlation_matrix"]
        )
        correlation_df.to_csv(f"{individual_dir}/{base_name}_correlation_matrix.csv")

        # Save interesting patterns
        if reward_component_correlations["interesting_patterns"]:
            patterns_df = pd.DataFrame(
                reward_component_correlations["interesting_patterns"]
            )
            patterns_df.to_csv(
                f"{individual_dir}/{base_name}_failure_patterns.csv", index=False
            )

    print(f"\n💾 Individual results saved to individual_results/:")
    if error_analysis["total_error_occurrences"] > 0:
        print(f"   - {base_name}_error_analysis.csv")
    print(f"   - {base_name}_pass_scores.csv")
    print(f"   - {base_name}_action_count_analysis.csv")
    if persona_analysis["by_conversation_style"]:
        print(f"   - {base_name}_persona_conversation_style.csv")
    if persona_analysis["by_tech_proficiency"]:
        print(f"   - {base_name}_persona_tech_proficiency.csv")
    if persona_analysis["by_combination"]:
        print(f"   - {base_name}_persona_combination.csv")
    if hallucination_type_pass_scores and any(hallucination_type_pass_scores.values()):
        print(f"   - {base_name}_hallucination_type_pass_scores.csv")
    if disambiguation_type_pass_scores and any(disambiguation_type_pass_scores.values()):
        print(f"   - {base_name}_disambiguation_type_pass_scores.csv")
    print(f"   - {base_name}_reward_component_distribution.csv")
    print(f"   - {base_name}_correlation_matrix.csv")
    print(f"   - {base_name}_failure_patterns.csv")
    print(f"   - {base_name}_metadata.csv")
    # Success count distribution
    if success_count_categorization:
        dist_rows = []
        for sc, info in success_count_categorization["distribution"].items():
            dist_rows.append(
                {
                    "success_count": sc,
                    "task_count": info["task_count"],
                    "percentage": info["percentage"],
                    # Store all task ids (may create large cells for many tasks)
                    "sample_task_ids": ",".join(map(str, info["task_ids"])),
                }
            )
        dist_df = pd.DataFrame(dist_rows)
        dist_df.to_csv(
            f"{individual_dir}/{base_name}_success_count_distribution.csv",
            index=False,
        )
        summary_df = pd.DataFrame([success_count_categorization["summary"]])
        summary_df.to_csv(
            f"{individual_dir}/{base_name}_success_count_summary.csv", index=False
        )
        print(f"   - {base_name}_success_count_distribution.csv")
        print(f"   - {base_name}_success_count_summary.csv")


def print_single_model_results(model_name: str, results: Dict[str, Any]):
    """Print analysis results for a single model."""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    print_analysis_results(
        results["pass_power_k_scores"],
        results["pass_at_k_scores"],
        results["rdi"],
        results["error_analysis"],
        results["action_count_analysis"],
        results["persona_analysis"],
        results.get("hallucination_type_pass_scores"),
        results.get("disambiguation_type_pass_scores"),
        results["reward_component_analysis"],
        results["reward_component_correlations"],
        results["metadata"],
        results.get("success_count_categorization"),
    )


def print_comparison_results(comparison_tables: Dict[str, pd.DataFrame]):
    """Print comparison tables across models."""
    print(f"\n{'='*80}")
    print("MULTI-MODEL COMPARISON")
    print(f"{'='*80}")

    # Pass scores comparison (including RDI)
    print("\n📊 PASS SCORES COMPARISON (including RDI)")
    print("-" * 50)
    pass_scores_df = comparison_tables["pass_scores"]
    print(pass_scores_df.round(3))

    # Metadata comparison
    print("\n💰 COST AND LATENCY COMPARISON")
    print("-" * 50)
    metadata_df = comparison_tables["metadata"]
    display_cols = [
        "total_user_cost_per_task",
        "total_agent_cost_per_task",
        "avg_llm_latency_per_turn_ms",
        "num_tasks_analyzed",
    ]
    print(metadata_df[display_cols].round(4))

    # Action summary
    print("\n📊 OVERALL PERFORMANCE SUMMARY")
    print("-" * 50)
    action_df = comparison_tables["action_summary"]
    print(action_df.round(3))
    
    # Persona analysis comparison - by conversation style
    if "persona_by_style" in comparison_tables:
        print("\n👤 PERSONA ANALYSIS: BY CONVERSATION STYLE")
        print("-" * 50)
        persona_style_df = comparison_tables["persona_by_style"]
        # Pivot to show pass_rate by model and conversation style
        style_pivot = persona_style_df.pivot(
            index="conversation_style", columns="model", values="pass_rate"
        )
        print(style_pivot.round(3))
        
        # Also show aggregated stats (average across all models)
        print("\nAverage Pass Rate by Conversation Style (across all models):")
        for style in persona_style_df["conversation_style"].unique():
            style_data = persona_style_df[persona_style_df["conversation_style"] == style]
            avg_rate = style_data["pass_rate"].mean()
            total_successful = style_data["successful"].sum()
            total_tasks = style_data["total"].sum()
            overall_rate = total_successful / total_tasks if total_tasks > 0 else 0
            print(f"  {style:15s}: {overall_rate:6.3f} ({overall_rate*100:5.1f}%) [{total_successful:4d}/{total_tasks:4d}]")
    
    # Persona analysis comparison - by tech proficiency
    if "persona_by_tech" in comparison_tables:
        print("\n👤 PERSONA ANALYSIS: BY TECHNOLOGICAL PROFICIENCY")
        print("-" * 50)
        persona_tech_df = comparison_tables["persona_by_tech"]
        # Pivot to show pass_rate by model and tech proficiency
        tech_pivot = persona_tech_df.pivot(
            index="tech_proficiency", columns="model", values="pass_rate"
        )
        print(tech_pivot.round(3))
        
        # Also show aggregated stats (average across all models)
        print("\nAverage Pass Rate by Tech Proficiency (across all models):")
        for tech in persona_tech_df["tech_proficiency"].unique():
            tech_data = persona_tech_df[persona_tech_df["tech_proficiency"] == tech]
            avg_rate = tech_data["pass_rate"].mean()
            total_successful = tech_data["successful"].sum()
            total_tasks = tech_data["total"].sum()
            overall_rate = total_successful / total_tasks if total_tasks > 0 else 0
            print(f"  {tech:15s}: {overall_rate:6.3f} ({overall_rate*100:5.1f}%) [{total_successful:4d}/{total_tasks:4d}]")

    # Hallucination type pass scores comparison
    if "hallucination_type_pass_scores" in comparison_tables:
        print("\n🧪 HALLUCINATION TYPE PASS SCORES (comparison)")
        print("-" * 50)
        df = comparison_tables["hallucination_type_pass_scores"]
        pivot = df.pivot_table(index=["type", "metric"], columns="model", values="score")
        print(pivot.round(3))
        
        # Show averaged scores across all models
        print("\nAverage Pass Scores by Hallucination Type (across all models):")
        for h_type in df["type"].unique():
            type_data = df[df["type"] == h_type]
            print(f"\n  {h_type}:")
            for metric in sorted(type_data["metric"].unique()):
                metric_data = type_data[type_data["metric"] == metric]
                avg_score = metric_data["score"].mean()
                print(f"    {metric:>10}: {avg_score:6.3f} ({avg_score*100:5.1f}%)")

    # Disambiguation type pass scores comparison
    if "disambiguation_type_pass_scores" in comparison_tables:
        print("\n🧩 DISAMBIGUATION TYPE PASS SCORES (comparison)")
        print("-" * 50)
        df = comparison_tables["disambiguation_type_pass_scores"]
        pivot = df.pivot_table(index=["type", "metric"], columns="model", values="score")
        print(pivot.round(3))
        
        # Show averaged scores across all models
        print("\nAverage Pass Scores by Disambiguation Type (across all models):")
        for d_type in df["type"].unique():
            type_data = df[df["type"] == d_type]
            print(f"\n  {d_type}:")
            for metric in sorted(type_data["metric"].unique()):
                metric_data = type_data[type_data["metric"] == metric]
                avg_score = metric_data["score"].mean()
                print(f"    {metric:>10}: {avg_score:6.3f} ({avg_score*100:5.1f}%)")


def print_task_level_results(task_analysis: Dict[str, Any]):
    """Print task-level analysis results."""
    print(f"\n{'='*80}")
    print("TASK-LEVEL PERFORMANCE ANALYSIS")
    print(f"{'='*80}")

    total_tasks = task_analysis["total_tasks"]
    unsolved = task_analysis["unsolved_tasks"]
    mixed = task_analysis["mixed_results_tasks"]
    universal = task_analysis["universally_solved_tasks"]

    print(f"\n📋 TASK DISTRIBUTION ({total_tasks} total tasks)")
    print("-" * 50)
    print(
        f"❌ Unsolved by all models:     {len(unsolved):3d} ({len(unsolved)/total_tasks*100:5.1f}%)"
    )
    print(
        f"⚡ Mixed results across models: {len(mixed):3d} ({len(mixed)/total_tasks*100:5.1f}%)"
    )
    print(
        f"✅ Solved by all models:       {len(universal):3d} ({len(universal)/total_tasks*100:5.1f}%)"
    )

    # Unsolved tasks analysis
    if unsolved:
        print(f"\n🚫 TASKS NO MODEL SOLVED ({len(unsolved)} tasks)")
        print("-" * 50)
        stats = task_analysis["statistics"]["unsolved"]
        print(f"Average actions: {stats['avg_actions']:.1f}")
        print(f"Action range: {stats['action_range'][0]}-{stats['action_range'][1]}")
        print(f"With disambiguation: {stats['disambiguation_percentage']:.1f}%")

        if len(unsolved) <= 10:
            print("Task IDs:", [task["task_id"] for task in unsolved])
        else:
            print(f"First 10 Task IDs: {[task['task_id'] for task in unsolved[:10]]}")
            print(f"... and {len(unsolved)-10} more")

    # Mixed results analysis
    if mixed:
        print(f"\n⚡ TASKS WITH MIXED RESULTS ({len(mixed)} tasks)")
        print("-" * 50)
        stats = task_analysis["statistics"]["mixed"]
        print(f"Average actions: {stats['avg_actions']:.1f}")
        print(f"Action range: {stats['action_range'][0]}-{stats['action_range'][1]}")
        print(f"With disambiguation: {stats['disambiguation_percentage']:.1f}%")

        print(f"\nMost controversial tasks (lowest success rates):")
        for task in mixed[:5]:
            successful = ", ".join(task["successful_models"])
            failed = ", ".join(task["failed_models"])
            print(
                f"  Task {task['task_id']:3d}: {task['success_rate']*100:4.0f}% success | "
                f"✅ {successful} | ❌ {failed}"
            )

    # Universal success analysis
    if universal:
        print(f"\n✅ UNIVERSALLY SOLVED TASKS ({len(universal)} tasks)")
        print("-" * 50)
        stats = task_analysis["statistics"]["universal"]
        print(f"Average actions: {stats['avg_actions']:.1f}")
        print(f"Action range: {stats['action_range'][0]}-{stats['action_range'][1]}")
        print(f"With disambiguation: {stats['disambiguation_percentage']:.1f}%")


def save_task_level_results(
    task_analysis: Dict[str, Any], output_dir: Path, base_name: str
):
    """Save task-level analysis results to CSV files."""
    comparison_dir = output_dir / "comparison_tables"
    comparison_dir.mkdir(exist_ok=True)

    # Save unsolved tasks
    if task_analysis["unsolved_tasks"]:
        unsolved_df = pd.DataFrame(task_analysis["unsolved_tasks"])
        unsolved_df.to_csv(
            comparison_dir / f"{base_name}_unsolved_tasks.csv", index=False
        )

    # Save mixed results tasks
    if task_analysis["mixed_results_tasks"]:
        mixed_df = pd.DataFrame(task_analysis["mixed_results_tasks"])
        # Convert lists to strings for CSV
        mixed_df["successful_models"] = mixed_df["successful_models"].apply(
            lambda x: "; ".join(x)
        )
        mixed_df["failed_models"] = mixed_df["failed_models"].apply(
            lambda x: "; ".join(x)
        )
        mixed_df.to_csv(
            comparison_dir / f"{base_name}_mixed_results_tasks.csv", index=False
        )

    # Save universally solved tasks
    if task_analysis["universally_solved_tasks"]:
        universal_df = pd.DataFrame(task_analysis["universally_solved_tasks"])
        universal_df.to_csv(
            comparison_dir / f"{base_name}_universally_solved_tasks.csv", index=False
        )

    # Save summary statistics
    stats_rows = []
    for category, stats in task_analysis["statistics"].items():
        if stats:  # Only include non-empty categories
            stats_row = {"category": category, **stats}
            # Convert task_types dict to string
            if "task_types" in stats_row:
                stats_row["task_types"] = str(stats_row["task_types"])
            stats_rows.append(stats_row)

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(
            comparison_dir / f"{base_name}_task_statistics.csv", index=False
        )

    print(f"\n💾 Task-level results saved to comparison_tables/:")
    print(f"   - {base_name}_unsolved_tasks.csv")
    print(f"   - {base_name}_mixed_results_tasks.csv")
    print(f"   - {base_name}_universally_solved_tasks.csv")
    print(f"   - {base_name}_task_statistics.csv")


def save_comparison_results(
    comparison_tables: Dict[str, pd.DataFrame], output_dir: Path, base_name: str
):
    """Save comparison results to CSV files."""
    comparison_dir = output_dir / "comparison_tables"
    comparison_dir.mkdir(exist_ok=True)

    for table_name, df in comparison_tables.items():
        output_file = comparison_dir / f"{base_name}_comparison_{table_name}.csv"
        df.to_csv(output_file)

    print(f"\n💾 Comparison results saved to comparison_tables/:")
    for table_name in comparison_tables.keys():
        print(f"   - {base_name}_comparison_{table_name}.csv")


def save_results_json(
    all_results: Dict[str, Dict[str, Any]],
    all_organized_data: Dict[str, Dict[int, Dict[int, Dict]]],
    task_analysis: Dict[str, Any] | None,
    comparison_tables: Dict[str, pd.DataFrame] | None,
    exclude_task_ids: List[int],
    output_dir: Path,
):
    """Save all analysis results to JSON for later visualization."""
    
    # Convert DataFrames to dict for JSON serialization
    comparison_tables_dict = None
    if comparison_tables:
        comparison_tables_dict = {
            name: df.to_dict(orient="index") for name, df in comparison_tables.items()
        }
    
    # Prepare the complete results structure
    results_data = {
        "all_results": all_results,
        "task_analysis": task_analysis,
        "comparison_tables": comparison_tables_dict,
        "excluded_task_ids": exclude_task_ids,
    }
    
    # Save to JSON
    json_file = output_dir / "analysis_results.json"
    with open(json_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n💾 Complete analysis results saved to:")
    print(f"   - analysis_results.json (use with generate_figures.py)")
    if exclude_task_ids:
        print(f"   - Excluded {len(exclude_task_ids)} task(s) from analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tau-bench results and save metrics. Use generate_figures.py to create visualizations."
    )
    parser.add_argument(
        "input_files", nargs="+", help="Path(s) to JSON results file(s)"
    )
    parser.add_argument(
        "--output", "-o", help="Output directory (default: analysis_results)"
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    parser.add_argument(
        "--individual-only",
        action="store_true",
        help="Only analyze individual models, skip comparison",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=str,
        help="Comma-separated list of task IDs to exclude from analysis (e.g., '1,5,23')",
    )

    args = parser.parse_args()
    
    # Parse excluded task IDs
    exclude_task_ids = []
    if args.exclude_tasks:
        try:
            exclude_task_ids = [int(tid.strip()) for tid in args.exclude_tasks.split(",")]
            print(f"\n⚠️  Excluding {len(exclude_task_ids)} task(s) from analysis: {exclude_task_ids}")
        except ValueError as e:
            print(f"Error: Invalid task ID in --exclude-tasks. Must be comma-separated integers.")
            sys.exit(1)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("analysis_results")

    output_dir.mkdir(exist_ok=True)

    # Load and analyze all models
    all_results = {}
    all_organized_data = {}  # Store organized data for task-level analysis

    print(f"\n🔍 Processing {len(args.input_files)} result file(s)...")
    print("You will be prompted to enter a model name for each file.\n")

    for i, input_file in enumerate(args.input_files, 1):
        print(f"📁 File {i}/{len(args.input_files)}: {input_file}")
        data, model_name = load_results(input_file)

        # Check for duplicate model names
        if model_name in all_results:
            print(
                f"⚠️  Warning: Model name '{model_name}' already exists. Please use a unique name."
            )
            while True:
                model_name = input(f"Please enter a different model name: ").strip()
                if model_name and model_name not in all_results:
                    break
                if model_name in all_results:
                    print(
                        f"'{model_name}' is already used. Please try a different name."
                    )
                else:
                    print("Model name cannot be empty. Please try again.")

        print(f"🧠 Analyzing model: {model_name}")

        # Store organized data for task-level analysis
        organized_data = organize_data_by_task_and_trial(data, exclude_task_ids)
        all_organized_data[model_name] = organized_data

        results = analyze_single_model(data, model_name, exclude_task_ids)
        all_results[model_name] = results

        # Print individual results
        print_single_model_results(model_name, results)

        # Save individual results if requested
        if not args.no_csv:
            model_safe_name = model_name.replace(".", "_").replace("-", "_")
            save_analysis_to_csv(
                results["pass_power_k_scores"],
                results["pass_at_k_scores"],
                results["rdi"],
                results["error_analysis"],
                results["action_count_analysis"],
                results["persona_analysis"],
                results.get("hallucination_type_pass_scores"),
                results.get("disambiguation_type_pass_scores"),
                results["reward_component_analysis"],
                results["reward_component_correlations"],
                results["metadata"],
                results.get("success_count_categorization"),
                str(output_dir / f"{model_safe_name}_analysis.csv"),
            )

    # Multi-model comparison (if more than one model)
    comparison_tables = None
    task_analysis = None
    
    if len(all_results) > 1 and not args.individual_only:
        comparison_tables = create_comparison_tables(all_results)
        print_comparison_results(comparison_tables)

        # Task-level analysis
        print(f"\n🎯 Analyzing task-level performance patterns...")
        task_analysis = analyze_task_level_performance_from_data(all_organized_data)
        print_task_level_results(task_analysis)

        # Save comparison results
        if not args.no_csv:
            save_comparison_results(comparison_tables, output_dir, "multi_model")
            save_task_level_results(task_analysis, output_dir, "multi_model")

    elif len(all_results) == 1:
        print(f"\nOnly one model analyzed. Use multiple input files for comparison.")
    
    # Save complete results to JSON for visualization
    save_results_json(all_results, all_organized_data, task_analysis, comparison_tables, exclude_task_ids, output_dir)

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"📊 To generate visualizations, run: python3 generate_figures.py {output_dir}")


if __name__ == "__main__":
    main()
