import argparse
from pytorch_policy import ScenarioConfig, train_policy, save_policy


def main():
    parser = argparse.ArgumentParser(description="Train and save a Berghain policy once.")
    parser.add_argument("--out", default="policy.pt", help="Path to save trained policy (default: policy.pt)")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps (default: 2000)")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size (default: 4096)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--capacity", type=int, default=1000, help="Venue capacity N (default: 1000)")
    parser.add_argument("--min_young", type=int, default=600, help="Min count for attribute 'young'")
    parser.add_argument("--min_well_dressed", type=int, default=600, help="Min count for 'well_dressed'")
    args = parser.parse_args()

    # Defaults from your scenario example
    attribute_ids = ["young", "well_dressed"]
    rel_freq = {"young": 0.3225, "well_dressed": 0.3225}
    correlations = {
        "young": {"young": 1.0, "well_dressed": 0.18304299322062992},
        "well_dressed": {"young": 0.18304299322062992, "well_dressed": 1.0},
    }

    min_counts = {"young": args.min_young, "well_dressed": args.min_well_dressed}

    config = ScenarioConfig(
        attribute_ids=attribute_ids,
        min_counts=min_counts,
        rel_freq=rel_freq,
        correlations=correlations,
        capacity=args.capacity,
    )

    print("Training policy...")
    policy, lambdas = train_policy(config, steps=args.steps, batch_size=args.batch, lr=args.lr)
    save_policy(policy, args.out)
    print(f"Saved trained policy to {args.out}")
    print("Lambdas (constraint pressures):", lambdas)


if __name__ == "__main__":
    main()


