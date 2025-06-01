import requests
import pandas as pd
from typing import List, Dict, Optional

def get_ncbi_gene_ID(gene_symbol: str) -> Optional[str]:
    """
    使用 NCBI Entrez API 查詢基因符號並返回相應的 Gene ID。

    :param gene_symbol: 要查詢的基因符號（例如 'TP53'）
    :return: 第一個匹配的基因 ID（如無結果，返回 None）
    """
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "gene",
        "term": f"{gene_symbol}[gene] AND human[orgn]",
        "retmode": "json"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # 如果請求失敗，拋出異常
        data = response.json()
        gene_ids = data.get("esearchresult", {}).get("idlist", [])
        return gene_ids[0] if gene_ids else None
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving Gene ID for {gene_symbol}: {e}")
        return None

def get_ncbi_gene_IDs(gene_symbols: List[str]) -> Dict[str, Optional[str]]:
    """
    批量查詢基因符號，返回其對應的 Gene IDs。

    :param gene_symbols: 基因符號列表
    :return: 字典，鍵為基因符號，值為 Gene ID 或 None
    """
    gene_ids = {}
    for symbol in gene_symbols:
        gene_id = get_ncbi_gene_ID(symbol)
        gene_ids[symbol] = gene_id
        print(f"Retrieved Gene ID for {symbol}: {gene_id}")
    return gene_ids

def get_gene_info_json(gene_ids: Dict[str, Optional[str]]) -> Dict[str, dict]:
    """
    根據多個 Gene IDs 批量查詢詳細的基因信息。

    :param gene_ids: 字典，鍵為基因符號，值為 Gene ID
    :return: 字典，包含每個基因的詳細信息
    """
    # 提取有效的 Gene IDs
    valid_gene_ids = [gene_id for gene_id in gene_ids.values() if gene_id]
    
    if not valid_gene_ids:
        print("No valid Gene IDs found.")
        return {}
    
    # 拼接所有 Gene IDs 用逗號分隔
    joined_ids = ",".join(valid_gene_ids)
    
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "gene", "id": joined_ids, "retmode": "json"}
    
    try:
        # 發送一次請求
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # 處理返回的數據
        result = data.get("result", {})
        
        # 移除不相關的鍵（例如 'uids'）
        result.pop("uids", None)
        
        # 將結果對應回基因符號
        gene_info = {}
        for symbol, gene_id in gene_ids.items():
            if gene_id and str(gene_id) in result:
                gene_info[symbol] = result[str(gene_id)]
            else:
                gene_info[symbol] = None
        
        print(f"Retrieved details for {len(valid_gene_ids)} genes.")
        return gene_info
    
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving gene details: {e}")
        return {}


def parse_gene_info(gene_info: Dict[str, dict]) -> Dict[str, Optional[str]]:
    """
    提取基因的摘要信息。

    :param gene_info: 每個基因的詳細信息字典
    :return: 字典，包含每個基因的摘要信息
    """
    summaries = {}
    for symbol, info in gene_info.items():
        summaries[symbol] = info.get("summary", None)
    return summaries

def get_gene_description_df(gene_symbols: list[str]):

    IDs = get_ncbi_gene_IDs(gene_symbols)
    gene_info = get_gene_info_json(IDs)
    parsed_info = parse_gene_info(gene_info)
    gene_description_df = pd.DataFrame.from_dict(parsed_info, 
                                                 orient='index', 
                                                 columns=['Description'])
    gene_description_df.reset_index(inplace=True)
    gene_description_df.rename(columns={"index": "Gene"}, inplace=True)

    return gene_description_df

