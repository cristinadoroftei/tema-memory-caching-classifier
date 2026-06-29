"""
Tool implementations — funcțiile efective ale tool-urilor.

Convenție: toate tools primesc params cu `input_dfs` (lista de DataFrames) + parametri specifici.
"""
import pandas as pd

from .registry import register_tool
from .params import JoinDataParams, FilterDataParams


@register_tool
def join_data(params: JoinDataParams) -> pd.DataFrame:
    """
    Combină două DataFrames pe baza unei chei comune (join).
    Suportă inner, left, right, outer join.

    TODO: Implementează folosind pandas merge().

    Args:
        params.input_dfs: [left_df, right_df]
        params.left_key: coloana cheie din primul DataFrame
        params.right_key: coloana cheie din al doilea DataFrame
        params.how: tipul de join

    Returns:
        DataFrame rezultat după join
    """
    left_df = params.input_dfs[0]
    right_df = params.input_dfs[1]
    return pd.merge(left_df, right_df, left_on=params.left_key, right_on=params.right_key, how=params.how)


@register_tool
def filter_data(params: FilterDataParams) -> pd.DataFrame:
    """
    Filtrează un DataFrame pe baza unei condiții.
    Suportă operatori: ==, !=, >, <, >=, <=, contains.

    TODO: Implementează folosind pandas boolean indexing.

    Args:
        params.input_dfs: [df]
        params.column: coloana pe care se aplică filtrul
        params.operator: operatorul de comparație
        params.value: valoarea pentru comparație

    Returns:
        DataFrame filtrat
    """
    import operator as op

    df = params.input_dfs[0]
    col = df[params.column]
    value = params.value

    if params.operator == "contains":
        mask = col.astype(str).str.contains(str(value), case=False, na=False)
    else:
        # Conversie numerică pentru comparații
        if params.operator in (">", "<", ">=", "<="):
            try:
                value = float(value)
                col = pd.to_numeric(col, errors="coerce")
            except ValueError:
                pass

        ops = {"==": op.eq, "!=": op.ne, ">": op.gt, "<": op.lt, ">=": op.ge, "<=": op.le}
        mask = ops[params.operator](col, value)

    return df[mask]
