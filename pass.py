"""
Bouncer System Status Monitor

This script makes repeated HTTP GET requests to monitor a bouncer system's status.
It parses JSON responses containing admission/rejection counts and person attributes.

Usage:
1. Install dependencies: pip install -r requirements.txt
2. Update the bouncer_api_url in the main section
3. Run: python pass.py

Expected JSON response format:
{
    "status": "running",
    "admittedCount": 0,
    "rejectedCount": 1,
    "nextPerson": {
        "personIndex": 1,
        "attributes": {
            "well_dressed": false,
            "young": true
        }
    }
}
"""

import requests
import time
import json
from typing import Optional, List, Dict

try:
    from pytorch_policy import (
        ScenarioConfig,
        train_policy,
        load_policy,
        policy_decide,
    )
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Berghain bouncer decision loop
def bouncer_decision_loop(base_url, game_id, num_requests=20, delay=1, model_path: Optional[str] = None, attribute_ids: Optional[List[str]] = None, max_total_encounters: int = 20000, min_counts: Optional[Dict[str, int]] = None, capacity: int = 1000):
    """
    Makes repeated HTTP GET requests to the Berghain challenge API with dynamic decisions

    Args:
        base_url (str): Base URL of the API
        game_id (str): Game ID for the challenge
        num_requests (int): Number of requests to make (default: 20)
        delay (float): Delay between requests in seconds (default: 1)
    """
    current_person_index = 0
    accept_decision = False  # Default for first request

    # Optional: load a trained PyTorch policy if available
    policy_model = None
    deferred_model_load = False
    if TORCH_AVAILABLE and model_path is not None:
        if attribute_ids is not None:
            try:
                policy_model = load_policy(model_path, num_features=len(attribute_ids))
                print("Loaded PyTorch policy for decisions.")
            except Exception as e:
                print(f"Could not load policy: {e}. Falling back to rule-based decisions.")
        else:
            # Load after we discover attribute_ids from first response
            deferred_model_load = True

    for i in range(num_requests):
        try:
            # Build the URL with current person index
            # Per API: first person (index 0) may omit 'accept'
            if i == 0 and current_person_index == 0:
                url = f"{base_url}?gameId={game_id}&personIndex={current_person_index}"
            else:
                url = f"{base_url}?gameId={game_id}&personIndex={current_person_index}&accept={str(accept_decision).lower()}"

            print(f"\nRequest {i+1}/{num_requests}")
            print(f"URL: {url}")
            print(f"Person Index: {current_person_index}, Accept: {accept_decision}")

            response = requests.get(url)
            print(f"HTTP Status Code: {response.status_code}")

            # Parse JSON response
            try:
                data = response.json()

                # Extract bouncer system information
                status = data.get('status', 'unknown')
                admitted = data.get('admittedCount', 0)
                rejected = data.get('rejectedCount', 0)

                # Compute acceptance rate and total encounters
                total_encounters = admitted + rejected
                acc_rate = (admitted / total_encounters) if total_encounters > 0 else 0.0
                # Print summary at start of each request
                print(f"📊 SUMMARY: Admitted: {admitted} | Rejected: {rejected} | Total: {total_encounters} | AccRate: {acc_rate:.3f} | Current Person: {current_person_index}")
                print(f"System Status: {status}")

                # Adaptive policy threshold controller to keep acceptance within [0.40, 0.75]
                if not hasattr(bouncer_decision_loop, "policy_tau"):
                    bouncer_decision_loop.policy_tau = 0.5
                tau = bouncer_decision_loop.policy_tau
                if total_encounters >= 10:  # wait a bit before controlling
                    if acc_rate > 0.75:
                        tau = min(0.999, tau + 0.05)
                    elif acc_rate < 0.40:
                        tau = max(0.05, tau - 0.05)
                bouncer_decision_loop.policy_tau = tau
                print(f"  Controller threshold tau={tau:.3f}")

                # Stop if game is completed/failed or encounter cap reached
                if status in ("completed", "failed"):
                    print(f"Game status '{status}' reached. Stopping loop.")
                    break
                if admitted >= capacity:
                    print(f"Venue full at capacity {capacity}. Stopping loop.")
                    break
                if total_encounters >= max_total_encounters:
                    print(f"Reached encounter cap {max_total_encounters}. Stopping loop.")
                    break

                # Check next person information
                next_person = data.get('nextPerson')
                if next_person:
                    person_index = next_person.get('personIndex', current_person_index)
                    attributes = next_person.get('attributes', {})

                    # Initialize attribute_ids from first response if not provided
                    if attribute_ids is None:
                        attribute_ids = sorted(list(attributes.keys()))
                        if TORCH_AVAILABLE and model_path is not None and deferred_model_load:
                            try:
                                policy_model = load_policy(model_path, num_features=len(attribute_ids))
                                print("Loaded PyTorch policy after discovering attributes.")
                            except Exception as e:
                                print(f"Could not load policy after discovery: {e}. Using rule-based decisions.")

                    # Convenience lookups for fallback rule
                    well_dressed = bool(attributes.get('well_dressed', False))
                    young = bool(attributes.get('young', False))

                    print(f"Next Person (Index: {person_index}):")
                    print(f"  Well dressed: {well_dressed}")
                    print(f"  Young: {young}")

                    # Maintain per-attribute accepted counts locally
                    if not hasattr(bouncer_decision_loop, "accepted_by_attr"):
                        bouncer_decision_loop.accepted_by_attr = {a: 0 for a in (attribute_ids or [])}
                    accepted_by_attr = bouncer_decision_loop.accepted_by_attr

                    # Feasibility-aware guard: ensure we keep enough slots to meet minima
                    if True:
                        forced_reject = False
                        if min_counts and attribute_ids:
                            slots_remaining = capacity - admitted
                            potential_slots_after = max(0, slots_remaining - 1)
                            needed_after_accept = []
                            for a in attribute_ids:
                                need = max(0, min_counts.get(a, 0) - accepted_by_attr.get(a, 0) - (1 if attributes.get(a, False) else 0))
                                needed_after_accept.append(need)
                            min_slots_required = max(needed_after_accept) if needed_after_accept else 0
                            if min_slots_required > potential_slots_after:
                                forced_reject = True
                            # Hard capacity safety: never accept when no slots remain
                            if slots_remaining <= 0:
                                forced_reject = True

                        if forced_reject:
                            accept_decision = False
                            print("  → Forced REJECT to preserve feasibility of constraints (guard)")
                        else:
                            # If all quotas satisfied, accept to fill remaining capacity
                            quotas_met = True
                            if min_counts and attribute_ids:
                                for a in attribute_ids:
                                    if accepted_by_attr.get(a, 0) < min_counts.get(a, 0):
                                        quotas_met = False
                                        break
                            if quotas_met:
                                accept_decision = True
                                print("  → Decision: ACCEPT (all quotas met; filling capacity)")
                            else:
                                # If acceptance is too high, reject non-helpful candidates to conserve capacity
                                if acc_rate > 0.75 and min_counts and attribute_ids:
                                    helps_any_deficit = False
                                    for a in attribute_ids:
                                        if accepted_by_attr.get(a, 0) < min_counts.get(a, 0) and attributes.get(a, False):
                                            helps_any_deficit = True
                                            break
                                    if not helps_any_deficit:
                                        accept_decision = False
                                        print("  → Decision: REJECT (acc_rate>0.75 and candidate non-helpful)")
                                        # Skip model decision
                                        goto_model_decision = False
                                    else:
                                        goto_model_decision = True
                                else:
                                    goto_model_decision = True

                                # Otherwise, use model with adaptive threshold; if not, heuristic fallback
                                if policy_model is not None and attribute_ids is not None:
                                    # Deficit-aware boost: lower threshold slightly if candidate helps any deficit
                                    threshold_boost = 0.0
                                    if min_counts and attribute_ids:
                                        for a in attribute_ids:
                                            if accepted_by_attr.get(a, 0) < min_counts.get(a, 0) and attributes.get(a, False):
                                                threshold_boost = -0.05
                                                break
                                    dyn_threshold = max(0.05, min(0.999, bouncer_decision_loop.policy_tau + threshold_boost))
                                    if goto_model_decision:
                                        accept_decision = policy_decide(policy_model, attributes, attribute_ids, threshold=dyn_threshold)
                                    print(f"  → Decision from policy: {'ACCEPT' if accept_decision else 'REJECT'}")
                                else:
                                    # Simple heuristic: accept if acceptance below lower bound, otherwise reject
                                    if acc_rate < 0.40:
                                        accept_decision = True
                                        print("  → Decision: ACCEPT (fallback, rate below lower bound)")
                                    else:
                                        accept_decision = False
                                        print("  → Decision: REJECT (fallback)")

                    # Update accepted-by-attribute counts if we decided to accept
                    if accept_decision and attribute_ids:
                        for a in attribute_ids:
                            if attributes.get(a, False):
                                accepted_by_attr[a] = accepted_by_attr.get(a, 0) + 1
                else:
                    print("No next person information available")
                    # Default decision when no next person data
                    accept_decision = False

                # Move to API-reported next person index (more robust than +1)
                next_person = data.get('nextPerson')
                if next_person and 'personIndex' in next_person:
                    current_person_index = int(next_person['personIndex'])
                else:
                    current_person_index = current_person_index + 1
                print(f"  → Next person index: {current_person_index}")

                # Print final summary with updated counts
                print(f"-Running Tally: {accepted_by_attr}")
                print(f"✅ FINAL: Admitted: {admitted} | Rejected: {rejected} | Next Person: {current_person_index}")
                print("-" * 80)

            except json.JSONDecodeError:
                print(f"Response: {response.text[:300]}...")
                print("-" * 80)

            if i < num_requests - 1:  # Don't delay after last request
                time.sleep(delay)

        except requests.RequestException as e:
            print(f"Error on request {i+1}: {e}")
            continue

if __name__ == "__main__":
    # Berghain challenge API configuration
    base_url = "https://berghain.challenges.listenlabs.ai/decide-and-next"
    game_id = "5317b713-cb8e-4af5-864a-4545bf38252d"

    # Make requests to the bouncer challenge API
    bouncer_decision_loop(base_url, game_id, num_requests=1000, delay=0.01)
