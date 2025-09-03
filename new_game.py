import argparse
import os
import importlib.util
import requests
from pytorch_policy import ScenarioConfig, train_policy, save_policy


API_BASE = "https://berghain.challenges.listenlabs.ai"


def create_new_game(scenario: int, player_id: str):
    url = f"{API_BASE}/new-game?scenario={scenario}&playerId={player_id}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def extract_config(game_info) -> ScenarioConfig:
    constraints = game_info["constraints"]
    attr_stats = game_info["attributeStatistics"]

    attribute_ids = sorted(list(attr_stats["relativeFrequencies"].keys()))
    min_counts = {c["attribute"]: int(c["minCount"]) for c in constraints}
    rel_freq = {a: float(attr_stats["relativeFrequencies"][a]) for a in attribute_ids}
    correlations = {a: {b: float(attr_stats["correlations"][a][b]) for b in attribute_ids} for a in attribute_ids}

    return ScenarioConfig(
        attribute_ids=attribute_ids,
        min_counts=min_counts,
        rel_freq=rel_freq,
        correlations=correlations,
        capacity=1000,
    )


def load_bouncer_loop():
    """Dynamically load bouncer_decision_loop from pass.py (pass is a Python keyword)."""
    here = os.path.dirname(__file__)
    module_path = os.path.join(here, "pass.py")
    spec = importlib.util.spec_from_file_location("bouncer_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Unable to load pass.py"
    spec.loader.exec_module(mod)
    return mod.bouncer_decision_loop


def main():
    parser = argparse.ArgumentParser(description="Create a new game, train a policy, and run it.")
    parser.add_argument("--scenario", type=int, default=1, help="Scenario (1, 2, or 3)")
    parser.add_argument("--player", required=True, help="playerId UUID")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch", type=int, default=4096, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model", default="policy.pt", help="Path to save trained model")
    parser.add_argument("--requests", type=int, default=1200, help="Max requests to issue in run loop (upper bound)")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between requests during run loop")
    parser.add_argument("--cap", type=int, default=20000, help="Hard cap on total encounters (API limit)")
    args = parser.parse_args()

    # 1) Create new game
    game_info = create_new_game(args.scenario, args.player)
    game_id = game_info["gameId"]
    print(f"Created game: {game_id}")

    # 2) Build config and train policy
    config = extract_config(game_info)
    print("Training policy from game constraints/statistics...")
    policy, lambdas = train_policy(config, steps=args.steps, batch_size=args.batch, lr=args.lr)
    save_policy(policy, args.model)
    print(f"Saved trained policy to {args.model}")
    print("Lambdas:", lambdas)

    # 3) Run the game with the trained model
    decide_url = f"{API_BASE}/decide-and-next"
    print("Starting run loop...")
    bouncer_decision_loop = load_bouncer_loop()
    bouncer_decision_loop(
        base_url=decide_url,
        game_id=game_id,
        num_requests=args.requests,
        delay=args.delay,
        model_path=args.model,
        attribute_ids=config.attribute_ids,
        max_total_encounters=args.cap,
        min_counts=config.min_counts,
        capacity=config.capacity,
    )


if __name__ == "__main__":
    main()


