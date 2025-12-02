# %%
import json
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

import src.config as config
from src.utils.path import get_project_root

# %%
CURRENT_DATE = "2025-11-28"
PREV_DATE = "2025-09-05"
START_DATE = "2021-01-01"


# %%
def compare_files_by_date(
    stock: str,
    file_base_name: str,
    current_date: str,
    prev_date: str,
    start_date: str,
    data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare deux fichiers JSON pour un stock donné sur une plage de dates.

    Args:
        stock: Symbol du stock
        file_base_name: Nom de base du fichier (ex: "_historical_price_full")
        current_date: Date du fichier actuel
        prev_date: Date du fichier précédent
        start_date: Date de début pour le filtrage
        data_path: Chemin du dossier contenant les données

    Returns:
        Tuple contenant (differences, comparison complète)
    """
    current_file = data_path / f"{stock}_{current_date}{file_base_name}.json"
    prev_file = data_path / f"{stock}_{prev_date}{file_base_name}.json"

    with open(current_file, "r") as f:
        current_json = json.load(f)
    with open(prev_file, "r") as f:
        prev_json = json.load(f)

    df_current = pd.DataFrame(current_json)
    df_prev = pd.DataFrame(prev_json)

    # Convertir les colonnes 'date' en datetime
    df_current["date"] = pd.to_datetime(df_current["date"])
    df_prev["date"] = pd.to_datetime(df_prev["date"])

    # Filtrer pour garder seulement les dates <= prev_date
    prev_date_obj = pd.to_datetime(prev_date)
    start_date_obj = pd.to_datetime(start_date)

    df_current_filtered = df_current[
        (df_current["date"] <= prev_date_obj) & (df_current["date"] >= start_date_obj)
    ].copy()

    df_prev_filtered = df_prev[
        (df_prev["date"] <= prev_date_obj) & (df_prev["date"] >= start_date_obj)
    ].copy()

    if "ratingScore" in df_prev_filtered.columns:
        df_prev_filtered["overallScore"] = df_prev_filtered["ratingScore"]
    if "ratingDetailsDCFScore" in df_prev_filtered.columns:
        df_prev_filtered["discountedCashFlowScore"] = df_prev_filtered[
            "ratingDetailsDCFScore"
        ]
    if "ratingDetailsROEScore" in df_prev_filtered.columns:
        df_prev_filtered["returnOnEquityScore"] = df_prev_filtered[
            "ratingDetailsROEScore"
        ]
    if "ratingDetailsROAScore" in df_prev_filtered.columns:
        df_prev_filtered["returnOnAssetsScore"] = df_prev_filtered[
            "ratingDetailsROAScore"
        ]
    if "ratingDetailsPEScore" in df_prev_filtered.columns:
        df_prev_filtered["priceEarningsScore"] = df_prev_filtered[
            "ratingDetailsPEScore"
        ]
    if "ratingDetailsPBScore" in df_prev_filtered.columns:
        df_prev_filtered["priceBookScore"] = df_prev_filtered["ratingDetailsPBScore"]

    comparison = pd.merge(
        df_current_filtered, df_prev_filtered, on="date", how="outer", indicator=True
    )

    # Retourner les dates avec des différences
    differences = comparison[comparison["_merge"] != "both"]
    return differences, comparison


# %%
def check_differences_for_all_stocks(file_base_name: str) -> None:
    """
    Vérifie les différences pour tous les stocks configurés.

    Args:
        file_base_name: Nom de base du fichier (ex: "_historical_price_full")
    """
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Afficher les options pandas pour un meilleur affichage
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    for stock in tqdm(config.TRADE_STOCKS):
        differences, comparison = compare_files_by_date(
            stock=stock,
            file_base_name=file_base_name,
            current_date=CURRENT_DATE,
            prev_date=PREV_DATE,
            start_date=START_DATE,
            data_path=data_path,
        )

        if len(differences) > 0:
            min_date = differences["date"].min()
            max_date = differences["date"].max()
            total_lines = len(comparison)
            num_differences = len(differences)
            print(f"\n=== {stock} - Dates avec différences ===")
            print(
                f"Range: {min_date.strftime('%Y-%m-%d')} à {max_date.strftime('%Y-%m-%d')}"
            )
            print(f"Différences: {num_differences}/{total_lines} lignes")
            # print(differences[["date", "_merge"]])


# %%
def compare_general_files_by_date(
    file_name: str,
    current_date: str,
    prev_date: str,
    start_date: str,
    data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare deux fichiers JSON généraux (non liés aux stocks) sur une plage de dates.

    Args:
        file_name: Nom du fichier (ex: "_treasury_rates")
        current_date: Date du fichier actuel
        prev_date: Date du fichier précédent
        start_date: Date de début pour le filtrage
        data_path: Chemin du dossier contenant les données

    Returns:
        Tuple contenant (differences, comparison complète)
    """
    current_file = data_path / f"{current_date}{file_name}.json"
    prev_file = data_path / f"{prev_date}{file_name}.json"

    with open(current_file, "r") as f:
        current_json = json.load(f)
    with open(prev_file, "r") as f:
        prev_json = json.load(f)

    df_current = pd.DataFrame(current_json)
    df_prev = pd.DataFrame(prev_json)

    # Convertir les colonnes 'date' en datetime
    df_current["date"] = pd.to_datetime(df_current["date"])
    df_prev["date"] = pd.to_datetime(df_prev["date"])

    # Filtrer pour garder seulement les dates <= prev_date
    prev_date_obj = pd.to_datetime(prev_date)
    start_date_obj = pd.to_datetime(start_date)
    df_current_filtered = df_current[
        (df_current["date"] <= prev_date_obj) & (df_current["date"] >= start_date_obj)
    ]
    df_prev_filtered = df_prev[
        (df_prev["date"] <= prev_date_obj) & (df_prev["date"] >= start_date_obj)
    ]

    comparison = pd.merge(
        df_current_filtered, df_prev_filtered, on="date", how="outer", indicator=True
    )

    # Retourner les dates avec des différences
    differences = comparison[comparison["_merge"] != "both"]
    return differences, comparison


def check_differences_for_general_file(file_name: str) -> None:
    """
    Vérifie les différences pour un fichier général (non lié aux stocks).

    Args:
        file_name: Nom du fichier (ex: "__treasury_rates")
    """
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Afficher les options pandas pour un meilleur affichage
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    differences, comparison = compare_general_files_by_date(
        file_name=file_name,
        current_date=CURRENT_DATE,
        prev_date=PREV_DATE,
        start_date=START_DATE,
        data_path=data_path,
    )

    if len(differences) > 0:
        min_date = differences["date"].min()
        max_date = differences["date"].max()
        total_lines = len(comparison)
        num_differences = len(differences)
        print(f"\n=== {file_name} - Dates avec différences ===")
        print(
            f"Range: {min_date.strftime('%Y-%m-%d')} à {max_date.strftime('%Y-%m-%d')}"
        )
        print(f"Différences: {num_differences}/{total_lines} lignes")
        # print(differences[["date", "_merge"]])


def compare_dict_files_by_date(
    file_name: str,
    current_date: str,
    prev_date: str,
    start_date: str,
    data_path: Path,
    threshold_percent: float = 3.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare deux fichiers JSON au format dictionnaire (dates comme clés) sur une plage de dates.

    Args:
        file_name: Nom du fichier (ex: "_economic_indicators")
        current_date: Date du fichier actuel
        prev_date: Date du fichier précédent
        start_date: Date de début pour le filtrage
        data_path: Chemin du dossier contenant les données
        threshold_percent: Seuil de changement en pourcentage pour reporter une différence (default: 1.0)

    Returns:
        Tuple contenant (differences, comparison complète)
    """
    current_file = data_path / f"{current_date}{file_name}.json"
    prev_file = data_path / f"{prev_date}{file_name}.json"

    with open(current_file, "r") as f:
        current_json = json.load(f)
    with open(prev_file, "r") as f:
        prev_json = json.load(f)

    # Convertir les dictionnaires en DataFrames en transposant
    df_current = pd.DataFrame.from_dict(current_json, orient="index")
    df_current.index = pd.to_datetime(df_current.index)
    df_current = df_current.sort_index()

    df_prev = pd.DataFrame.from_dict(prev_json, orient="index")
    df_prev.index = pd.to_datetime(df_prev.index)
    df_prev = df_prev.sort_index()

    # Filtrer pour garder seulement les dates <= prev_date
    prev_date_obj = pd.to_datetime(prev_date)
    start_date_obj = pd.to_datetime(start_date)

    df_current_filtered = df_current[
        (df_current.index <= prev_date_obj) & (df_current.index >= start_date_obj)
    ]
    df_prev_filtered = df_prev[
        (df_prev.index <= prev_date_obj) & (df_prev.index >= start_date_obj)
    ]

    # Réinitialiser l'index pour la fusion
    df_current_filtered = df_current_filtered.reset_index().rename(
        columns={"index": "date"}
    )
    df_prev_filtered = df_prev_filtered.reset_index().rename(columns={"index": "date"})

    comparison = pd.merge(
        df_current_filtered,
        df_prev_filtered,
        on="date",
        how="outer",
        indicator=True,
        suffixes=("_current", "_prev"),
    )

    # Trouver les différences de valeurs pour les dates présentes dans les deux fichiers
    both_dates = comparison[comparison["_merge"] == "both"]
    differences_list = []

    for idx, row in both_dates.iterrows():
        date = row["date"]
        # Comparer les colonnes (exclure 'date' et '_merge')
        cols_to_compare = [col for col in df_current_filtered.columns if col != "date"]

        for col in cols_to_compare:
            col_current = f"{col}_current"
            col_prev = f"{col}_prev"

            if col_current in row.index and col_prev in row.index:
                val_current = row[col_current]
                val_prev = row[col_prev]

                # Vérifier si les valeurs sont différentes (en tenant compte des NaN)
                if pd.isna(val_current) and pd.isna(val_prev):
                    continue
                elif pd.isna(val_current) or pd.isna(val_prev):
                    differences_list.append(
                        {
                            "date": date,
                            "column": col,
                            "current": val_current,
                            "prev": val_prev,
                            "change_percent": None,
                        }
                    )
                elif val_current != val_prev:
                    # Calculer le changement en pourcentage
                    change_percent = None
                    if isinstance(val_current, (int, float)) and isinstance(
                        val_prev, (int, float)
                    ):
                        if val_prev != 0:
                            change_percent = abs(
                                (val_current - val_prev) / val_prev * 100
                            )
                        else:
                            # Si val_prev est 0, considérer le changement comme significatif
                            change_percent = float("inf")

                        # Appliquer le seuil uniquement pour les valeurs numériques
                        if (
                            change_percent is not None
                            and change_percent < threshold_percent
                        ):
                            continue

                    differences_list.append(
                        {
                            "date": date,
                            "column": col,
                            "current": val_current,
                            "prev": val_prev,
                            "change_percent": change_percent,
                        }
                    )

    # Ajouter les dates présentes uniquement dans un fichier
    only_current = comparison[comparison["_merge"] == "left_only"]
    only_prev = comparison[comparison["_merge"] == "right_only"]

    differences = pd.DataFrame(differences_list)
    if len(only_current) > 0 or len(only_prev) > 0:
        # Ajouter les informations sur les dates manquantes
        for idx, row in only_current.iterrows():
            differences_list.append(
                {
                    "date": row["date"],
                    "column": "ALL",
                    "current": "Present",
                    "prev": "Missing",
                    "change_percent": None,
                }
            )
        for idx, row in only_prev.iterrows():
            differences_list.append(
                {
                    "date": row["date"],
                    "column": "ALL",
                    "current": "Missing",
                    "prev": "Present",
                    "change_percent": None,
                }
            )
        differences = pd.DataFrame(differences_list)

    return differences, comparison


def check_differences_for_dict_file(
    file_name: str, threshold_percent: float = 3.1
) -> None:
    """
    Vérifie les différences pour un fichier au format dictionnaire avec dates comme clés.

    Args:
        file_name: Nom du fichier (ex: "_economic_indicators")
        threshold_percent: Seuil de changement en pourcentage pour reporter une différence (default: 1.0)
    """
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Afficher les options pandas pour un meilleur affichage
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    differences, comparison = compare_dict_files_by_date(
        file_name=file_name,
        current_date=CURRENT_DATE,
        prev_date=PREV_DATE,
        start_date=START_DATE,
        data_path=data_path,
        threshold_percent=threshold_percent,
    )

    if len(differences) > 0:
        print(f"\n=== {file_name} - Différences > {threshold_percent}% détectées ===")
        print(f"Nombre de différences: {len(differences)}")
        print("\nDétail des différences:")
        print(differences.to_string())
    else:
        print(
            f"\n=== {file_name} - Aucune différence > {threshold_percent}% détectée ==="
        )


def compare_news_files_by_date(
    stock: str,
    current_date: str,
    prev_date: str,
    data_path: Path,
    similarity_threshold: float = 95.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare deux fichiers JSON de news pour un stock donné.

    Args:
        stock: Symbol du stock
        current_date: Date du fichier actuel
        prev_date: Date du fichier précédent
        data_path: Chemin du dossier contenant les données
        similarity_threshold: Seuil de similarité (0-100) pour considérer les textes comme identiques (default: 95.0)

    Returns:
        Tuple contenant (differences, comparison complète)
    """
    current_file = data_path / f"{stock}_{current_date}_stock_news.json"
    prev_file = data_path / f"{stock}_{prev_date}_stock_news.json"

    with open(current_file, "r") as f:
        current_json = json.load(f)
    with open(prev_file, "r") as f:
        prev_json = json.load(f)

    df_current = pd.DataFrame(current_json)
    df_prev = pd.DataFrame(prev_json)

    # Convertir publishedDate en datetime
    df_current["publishedDate"] = pd.to_datetime(df_current["publishedDate"])
    df_prev["publishedDate"] = pd.to_datetime(df_prev["publishedDate"])

    # Filtrer pour garder seulement les dates <= prev_date
    prev_date_obj = pd.to_datetime(prev_date)
    df_current_filtered = df_current[df_current["publishedDate"] <= prev_date_obj]
    df_prev_filtered = df_prev[df_prev["publishedDate"] <= prev_date_obj]

    # Merger sur publishedDate
    comparison = pd.merge(
        df_current_filtered,
        df_prev_filtered,
        on="publishedDate",
        how="outer",
        indicator=True,
        suffixes=("_current", "_prev"),
    )

    # Trouver les différences
    differences_list = []

    # Dates présentes dans les deux fichiers
    both_dates = comparison[comparison["_merge"] == "both"]
    for idx, row in both_dates.iterrows():
        published_date = row["publishedDate"]

        # Comparer title avec rapidfuzz
        if "title_current" in row.index and "title_prev" in row.index:
            title_current = row["title_current"]
            title_prev = row["title_prev"]
            if pd.notna(title_current) and pd.notna(title_prev):
                similarity = fuzz.ratio(str(title_current), str(title_prev))
                if similarity < similarity_threshold:
                    differences_list.append(
                        {
                            "publishedDate": published_date,
                            "field": "title",
                            "current": title_current,
                            "prev": title_prev,
                            "similarity": f"{similarity:.1f}%",
                        }
                    )
            elif title_current != title_prev:  # Un des deux est NaN
                differences_list.append(
                    {
                        "publishedDate": published_date,
                        "field": "title",
                        "current": title_current,
                        "prev": title_prev,
                        "similarity": "N/A",
                    }
                )

        # Comparer text avec rapidfuzz
        if "text_current" in row.index and "text_prev" in row.index:
            text_current = row["text_current"]
            text_prev = row["text_prev"]
            if pd.notna(text_current) and pd.notna(text_prev):
                similarity = fuzz.ratio(str(text_current), str(text_prev))
                if similarity < similarity_threshold:
                    differences_list.append(
                        {
                            "publishedDate": published_date,
                            "field": "text",
                            "current": text_current,
                            "prev": text_prev,
                            "similarity": f"{similarity:.1f}%",
                        }
                    )
            elif text_current != text_prev:  # Un des deux est NaN
                differences_list.append(
                    {
                        "publishedDate": published_date,
                        "field": "text",
                        "current": text_current,
                        "prev": text_prev,
                        "similarity": "N/A",
                    }
                )

    # Ajouter les dates présentes uniquement dans un fichier
    # only_current = comparison[comparison["_merge"] == "left_only"]
    # only_prev = comparison[comparison["_merge"] == "right_only"]

    # for idx, row in only_current.iterrows():
    #     differences_list.append(
    #         {
    #             "publishedDate": row["publishedDate"],
    #             "field": "ALL",
    #             "current": "Present",
    #             "prev": "Missing",
    #             "similarity": "N/A",
    #         }
    #     )
    # for idx, row in only_prev.iterrows():
    #     differences_list.append(
    #         {
    #             "publishedDate": row["publishedDate"],
    #             "field": "ALL",
    #             "current": "Missing",
    #             "prev": "Present",
    #             "similarity": "N/A",
    #         }
    #     )

    differences = pd.DataFrame(differences_list)
    return differences, comparison


def check_differences_for_news(similarity_threshold: float = 95.0) -> None:
    """
    Vérifie les différences dans les fichiers de news pour tous les stocks configurés.

    Args:
        similarity_threshold: Seuil de similarité (0-100) pour considérer les textes comme identiques (default: 95.0)
    """
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Afficher les options pandas pour un meilleur affichage
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # Dictionnaires pour accumuler les erreurs et le total par site
    errors_by_site = {}
    total_news_by_site = {}

    for stock in tqdm(config.TRADE_STOCKS):
        try:
            differences, comparison = compare_news_files_by_date(
                stock=stock,
                current_date=CURRENT_DATE,
                prev_date=PREV_DATE,
                data_path=data_path,
                similarity_threshold=similarity_threshold,
            )

            # Calculer le nombre total de news dans la comparaison
            total_news = len(comparison)
            num_differences = len(differences)

            # Compter le total de news par site
            for idx, row in comparison.iterrows():
                site = row.get("site_current") or row.get("site_prev")
                if pd.notna(site):
                    total_news_by_site[site] = total_news_by_site.get(site, 0) + 1

            if num_differences > 0:
                # Calculer le range des dates avec différences
                min_date = differences["publishedDate"].min()
                max_date = differences["publishedDate"].max()

                print(
                    f"\n=== {stock} - Différences dans les news (seuil: {similarity_threshold}%) ==="
                )
                print(f"Différences: {num_differences} sur {total_news} news")
                print(
                    f"Range: {min_date.strftime('%Y-%m-%d %H:%M:%S')} à {max_date.strftime('%Y-%m-%d %H:%M:%S')}"
                )

                # Compter les erreurs par site pour ce stock
                for idx, row in differences.iterrows():
                    published_date = row["publishedDate"]
                    # Trouver le site correspondant dans comparison
                    matching_rows = comparison[
                        comparison["publishedDate"] == published_date
                    ]
                    if len(matching_rows) > 0:
                        site = matching_rows.iloc[0].get(
                            "site_current"
                        ) or matching_rows.iloc[0].get("site_prev")
                        if pd.notna(site):
                            errors_by_site[site] = errors_by_site.get(site, 0) + 1

        except FileNotFoundError:
            print(f"\n=== {stock} - Fichiers de news non trouvés ===")

    # Afficher l'histogramme global des erreurs par site
    if errors_by_site:
        print("\n" + "=" * 100)
        print("HISTOGRAMME GLOBAL DES ERREURS PAR SITE DE NEWS")
        print("=" * 100)

        # Trier par nombre d'erreurs décroissant
        sorted_errors = sorted(errors_by_site.items(), key=lambda x: x[1], reverse=True)

        total_errors = sum(errors_by_site.values())
        total_all_news = sum(total_news_by_site.values())
        print(f"\nTotal d'erreurs: {total_errors} / {total_all_news} news\n")

        for site, error_count in sorted_errors:
            news_count = total_news_by_site.get(site, 0)
            error_rate = (error_count / news_count * 100) if news_count > 0 else 0
            percentage = (error_count / total_errors) * 100
            bar_length = int(percentage / 2)  # Échelle: 1 caractère = 2%
            bar = "█" * bar_length
            print(
                f"{site:30} | {bar} {error_count:4d}/{news_count:4d} ({error_rate:5.1f}% erreurs | {percentage:5.1f}% du total)"
            )

        print("=" * 100)

        # Afficher les sites avec moins de 10% d'erreurs
        print("\n" + "=" * 100)
        print("SITES AVEC MOINS DE 10% D'ERREURS SUR LEURS PUBLICATIONS")
        print("=" * 100 + "\n")

        low_error_sites = []
        for site in total_news_by_site.keys():
            error_count = errors_by_site.get(site, 0)
            news_count = total_news_by_site[site]
            error_rate = (error_count / news_count * 100) if news_count > 0 else 0

            if error_rate < 10.0:
                low_error_sites.append((site, error_count, news_count, error_rate))

        # Trier par taux d'erreur croissant
        low_error_sites.sort(key=lambda x: x[3])

        if low_error_sites:
            for site, error_count, news_count, error_rate in low_error_sites:
                print(
                    f"{site:30} | {error_count:4d}/{news_count:4d} ({error_rate:5.2f}% erreurs)"
                )
            print(f"\nTotal: {len(low_error_sites)} sites avec < 10% d'erreurs")
        else:
            print("Aucun site avec moins de 10% d'erreurs trouvé")

        print("=" * 100)

        # Générer et afficher la liste Python des sites avec moins de 5% d'erreurs
        print("\n" + "=" * 100)
        print(
            "LISTE PYTHON DES SITES AVEC MOINS DE 5% D'ERREURS ET AU MOINS 0.1% DES NEWS TOTALES"
        )
        print("=" * 100 + "\n")

        very_low_error_sites = []
        for site in total_news_by_site.keys():
            error_count = errors_by_site.get(site, 0)
            news_count = total_news_by_site[site]
            error_rate = (error_count / news_count * 100) if news_count > 0 else 0
            news_percentage = (
                (news_count / total_all_news * 100) if total_all_news > 0 else 0
            )

            if error_rate < 5.0 and news_percentage >= 0.1:
                very_low_error_sites.append(
                    (site, error_count, news_count, error_rate, news_percentage)
                )

        # Trier par ordre alphabétique
        very_low_error_sites.sort(key=lambda x: x[0])

        if very_low_error_sites:
            print("RELIABLE_NEWS_SITES = [")
            for (
                site,
                error_count,
                news_count,
                error_rate,
                news_percentage,
            ) in very_low_error_sites:
                print(
                    f'    "{site}",  # {error_count}/{news_count} ({error_rate:.2f}% erreurs, {news_percentage:.1f}% des news)'
                )
            print("]")
            print(
                f"\nTotal: {len(very_low_error_sites)} sites avec < 5% d'erreurs et >= 1% des news"
            )
        else:
            print("RELIABLE_NEWS_SITES = []")
            print("\nAucun site trouvé avec < 5% d'erreurs ET >= 1% des news totales")

        print("=" * 100)


# # %%
# check_differences_for_all_stocks("_historical_price_full")

# # %%
# check_differences_for_all_stocks("_earnings")

# %%
check_differences_for_all_stocks("_ratings")

# %%
check_differences_for_all_stocks("_analyst_stock_recommendations")

# # %%
# check_differences_for_general_file("_treasury_rates")

# %%
check_differences_for_dict_file("_economic_indicators")

# %%
check_differences_for_news()

# %%
