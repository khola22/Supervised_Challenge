# This files contains functions used across multiple notebooks

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import defaultdict
import math  
import ast
from collections.abc import Iterable


def plot_missing_values(datasets):
    """Plot percent of missing values per column for multiple datasets."""
    # plt.figure(figsize=(14, 10))
    # number of figures depends on number of datasets
    n_datasets = len(datasets)
    plt.figure(figsize=(14, 5 * ((n_datasets + 1) // 2)))
    for i, (name, df) in enumerate(datasets.items(), 1):
        plt.subplot(2, 2, i)
        # percent of missing values by column (0-100)
        na_pct = df.isna().mean() * 100
        na_pct = na_pct[na_pct > 0].sort_values(ascending=False)
        if na_pct.empty:
            plt.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=12)
            plt.title(f'{name}: 0 missing columns')
            plt.xlabel('')
            plt.yticks([])
        else:
            sns.barplot(x=na_pct.values, y=na_pct.index, palette='viridis')
            plt.xlabel('Percent missing (%)')
            plt.title(f'{name}: Missing values by column')
            # annotate bars with percent values
            for j, v in enumerate(na_pct.values):
                plt.text(v + 0.5, j, f'{v:.1f}%', va='center')
    plt.suptitle('Percent Missing Values per Column', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def replace_titles_with_ids(df, column, genres_id):
    title_to_id = dict(zip(genres_id['genre_title'], genres_id['genre_id']))
    df[column] = df[column].map(title_to_id)
    return df

def replace_ids_with_titles(df, column, genres_id):
    id_to_title = dict(zip(genres_id['genre_id'], genres_id['genre_title']))
    df[column] = df[column].map(id_to_title)
    return df

def build_genre_hierarchy(genres):
    """
    Builds a genre hierarchy from the genres DataFrame.

    Expects columns:
      - 'genre_id'
      - 'genre_parent_id'
      - 'genre_title'

    Returns:
      - parent_to_children: dict[parent_id] -> list[child_id]
      - mapped_parent_to_children: dict[parent_title] -> list[child_title]
      - id_to_title: dict[genre_id] -> genre_title
      - root_ids: list of root genre_ids (no parent)
      - root_titles: list of root genre_titles
    """
    # parent_id -> list of child_ids
    parent_to_children = defaultdict(list)
    id_to_title = dict(zip(genres['genre_id'], genres['genre_title']))

    # fill parent_to_children (by ids)
    for _, row in genres.iterrows():
        gid = row['genre_id']
        pid = row['genre_parent_id']
        # treat None / NaN as "no parent" (root)
        if pid is None or (isinstance(pid, float) and math.isnan(pid)):
            continue
        parent_to_children[pid].append(gid)

    # roots = ids that never appear as a child
    all_ids = set(genres['genre_id'])
    all_child_ids = set(genres['genre_parent_id'].dropna())
    root_ids = list(all_ids - all_child_ids)
    root_titles = [id_to_title[rid] for rid in root_ids]

    # same mapping, but with titles instead of ids
    mapped_parent_to_children = {}
    for pid, children in parent_to_children.items():
        parent_title = id_to_title.get(pid, f"Unknown({pid})")
        child_titles = [id_to_title.get(cid, f"Unknown({cid})") for cid in children]
        mapped_parent_to_children[parent_title] = child_titles

    return parent_to_children, mapped_parent_to_children, id_to_title, root_ids, root_titles

