from omegaconf import DictConfig

from ..utils import SetUp


def pipeline(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    recommendation_manager = setup.get_manager(manager_type="recommendation")
    report_manager = setup.get_manager(manager_type="report")

    print("Recipe AI Demo")

    while True:
        option = None
        lab_id_mode = str(config.input_mode.lab_id)
        ingredients_mode = str(config.input_mode.ingredients)
        while option not in {lab_id_mode, ingredients_mode}:
            print("Select input mode:")
            print(f"  {lab_id_mode}) Lab number based recommendation")
            print(f"  {ingredients_mode}) Ingredients based recommendation")
            print("Warning: please input mode for (or 'q', 'quit', 'exit' to quit): ")
            option = input(f"Enter {lab_id_mode} or {ingredients_mode}: ").strip()

            if option.lower() in {"q", "quit", "exit"}:
                print("Exiting. Bye!")
                exit()

        print(
            f"Recommendation mode: {'Lab' if option == lab_id_mode else 'Ingredients'}"
        )

        if option == lab_id_mode:
            lab_id = input(
                "Enter the lab number to recommend for (or 'q', 'quit', 'exit' to quit): "
            ).strip()

            if not lab_id:
                print("Warning: Enter the lab number to recommend for.")
                continue

            if lab_id.lower() in {"q", "quit", "exit"}:
                print("Exiting. Bye!")
                break

            query_value = lab_id
            query_type = config.input_mode.lab_id
        elif option == ingredients_mode:
            print(
                "Enter ingredients separated by commas (A, B, C) or newlines. (or 'q', 'quit', 'exit' to quit):"
            )
            ing_line = input("Ingredients: ").strip()
            if not ing_line:
                print(
                    "Warning: please input ingredients (or 'q', 'quit', 'exit' to quit)."
                )
                continue
            if ing_line.lower() in {"q", "quit", "exit"}:
                print("Exiting. Bye!")
                break

            ing_line = ing_line.replace("\n", ",")
            parts = [p.strip() for p in ing_line.split(",") if p.strip()]
            query_value = "|".join(parts)
            query_type = config.input_mode.ingredients

        category_value = input(
            "Enter a category to restrict by (press Enter to 'all' to include all categories, or 'q', 'quit', 'exit' to quit): "
        ).strip()

        if category_value.lower() in {"q", "quit", "exit"}:
            print("Exiting. Bye!")
            break
        if category_value.lower() == "all":
            category_value = None

        recommendations = recommendation_manager.recommend(
            input_value=query_value,
            input_type=query_type,
            category_value=category_value,
        )
        print("\nSummary of AI recommendations")
        print(recommendations)

        print("\nWhat would you like to do next?")
        print("  1) Generate report for these recommendations")
        print("  2) Start over and enter new input")
        print("  3) Quit")

        choice = (
            input(
                "Enter 1 to generate report, 2 to start over (or 'q', 'quit', 'exit' to quit): "
            )
            .strip()
            .lower()
        )

        if choice in {"q", "quit", "exit"}:
            print("Exiting. Bye!")
            break
        elif choice == "1":
            report = report_manager.generate(recommendations=recommendations)
            print("\nSummary of AI report")
            print(report)
            continue
        elif choice == "2":
            continue
        else:
            print("Invalid choice. Starting over.")
            continue
