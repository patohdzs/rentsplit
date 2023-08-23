from typing import Optional

import pandas as pd
import typer
from rich import print
from rich.table import Table


def run():
    typer.run(main)


def main():
    # Get number of rooms
    num_rooms = typer.prompt("Enter the number of rooms", type=int)

    # Get total rent
    rent = typer.prompt("Enter the total rent amount", type=float)

    # Get flatmate names
    names = _get_names(num_rooms)

    # Get valuations
    valuations = _get_valuations(names)

    # Confirm valuations
    confirm = _check_valuations(valuations, rent)
    if not confirm:
        print("wawa")


def _get_names(n: int) -> list[str]:
    return [
        typer.prompt(f"Enter the name of flatmate {i+1}", type=str) for i in range(n)
    ]


def _get_valuations(names: list[str]) -> pd.DataFrame:
    print("Now lets start the bidding process...")
    valuations = [
        [
            typer.prompt(f"{name}, how much do you value room {i+1}?", type=float)
            for i, _ in enumerate(names)
        ]
        for name in names
    ]
    return pd.DataFrame(
        valuations, index=names, columns=[f"room {i+1}" for i, _ in enumerate(names)]
    )


def _check_valuations(df: pd.DataFrame, rent: int) -> bool:
    print("Checking valuations...")
    sum_to_rent = (df.sum(axis=1) == rent).all()
    return sum_to_rent


def _df_to_table(
    df: pd.DataFrame,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    table = Table()

    if show_index:
        index_name = str(index_name) if index_name else ""
        table.add_column(index_name)

    for column in df.columns:
        table.add_column(str(column))

    for index, value_list in enumerate(df.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        table.add_row(*row)

    return table


if __name__ == "__main__":
    typer.run(main)
