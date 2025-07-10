import pandas as pd
from bs4 import BeautifulSoup
import requests

def read_GMT(filepath):
    pathway_list = []
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines:
        pathway, link, genes = line.split("\t", 2)
        genes = genes.split("\t")
        pathway_list.append((pathway, link, genes))
    df = pd.DataFrame(pathway_list, columns=["Pathway", "Link", "Genes"])
    df = df.set_index("Pathway", drop=True)
    return df

def fetch_msigdb_geneset(geneset_name, species="human"):
    url = f"https://www.gsea-msigdb.org/gsea/msigdb/{species}/geneset/{geneset_name}.html"

    res = requests.get(url)
    soup = BeautifulSoup(res.content, "html.parser")

    table = soup.select_one("#geneListing > table")
    rows = table.find_all("tr")[1:]  # 跳過 header

    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            source_id = cols[0].text.strip()
            entrez_id = cols[1].text.strip()
            gene_symbol = cols[2].text.strip()
            description = cols[3].text.strip()
            data.append({
                "source_id": source_id,
                "entrez_id": entrez_id,
                "gene_symbol": gene_symbol,
                "description": description
            })

    df = pd.DataFrame(data)
    return df
