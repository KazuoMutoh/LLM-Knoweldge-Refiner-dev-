# pip install networkit
from typing import List, Tuple, Hashable, Dict, Set
import networkit as nk

Triple = Tuple[Hashable, Hashable, Hashable]

def extract_k_hop_enclosing_subgraph(
    all_triples: List[Triple],
    target_triple: Triple,
    k: int = 2,
    directed: bool = False,
    remove_target: bool = False,
) -> List[Triple]:
    """Return k-hop enclosing subgraph triples around a target triple using NetworKit.

    Args:
        triples: 全トリプル (h, r, t) のリスト。h, r, t は任意のハッシュ可能オブジェクト。
        target: 対象トリプル (h, r, t)。
        k: hop 数（>=0）。
        directed: True なら距離は有向グラフ、False なら無向距離。
        remove_target: True なら結果から厳密な (h, r, t) を除外。

    Returns:
        k-hop enclosing subgraph 内のトリプル (h', r', t') のリスト。
    """
    
    th, tr, tt = target_triple

    # --- 1) エンティティを整数IDへ圧縮 ---
    ent2id: Dict[Hashable, int] = {}
    id2ent: List[Hashable] = []

    def _eid(x: Hashable) -> int:
        if x in ent2id:
            return ent2id[x]
        i = len(id2ent)
        ent2id[x] = i
        id2ent.append(x)
        return i

    # まず全頂点を登録＆辺のためのID列を準備
    edges_int: List[Tuple[int, int]] = []
    for h, r, t in all_triples:
        u = _eid(h)
        v = _eid(t)
        edges_int.append((u, v))

    if th not in ent2id or tt not in ent2id:
        return []

    src = ent2id[th]
    dst = ent2id[tt]

    # --- 2) NetworKit グラフ構築（無向/有向を切替） ---
    G = nk.Graph(n=len(id2ent), weighted=False, directed=directed)  # C++実装で高速 :contentReference[oaicite:2]{index=2}
    # 多重辺は距離に影響しないので一度だけ張れば十分
    seen = set()
    for u, v in edges_int:
        if directed:
            if (u, v) not in seen:
                G.addEdge(u, v)
                seen.add((u, v))
        else:
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) not in seen:
                G.addEdge(a, b)
                seen.add((a, b))

    # --- 3) BFS で ≤k-hop 範囲のノード集合（h側・t側） ---
    # NetworKit の BFS は最短路距離を返せます（C++実装） :contentReference[oaicite:3]{index=3}
    bfs_h = nk.distance.BFS(G, src)
    bfs_h.run()
    dist_h = bfs_h.getDistances()  # Python list/NumPy対応（新バージョンで高速） :contentReference[oaicite:4]{index=4}

    bfs_t = nk.distance.BFS(G, dst)
    bfs_t.run()
    dist_t = bfs_t.getDistances()

    # dist が float('inf') の場合は到達不可
    def nodes_within_k(dist_list) -> Set[int]:
        out = set()
        for i, d in enumerate(dist_list):
            if d <= k:
                out.add(i)
        return out

    v_h = nodes_within_k(dist_h)
    v_t = nodes_within_k(dist_t)
    v_sub = v_h & v_t
    if not v_sub:
        return []

    # --- 4) 誘導トリプルの抽出（元の入力を O(|V_sub|) セット照合でフィルタ） ---
    nodes_keep = {id2ent[i] for i in v_sub}
    out: List[Triple] = []
    skip = (th, tr, tt) if remove_target else None
    for h, r, t in all_triples:
        if h in nodes_keep and t in nodes_keep:
            if skip is not None and (h, r, t) == skip:
                continue
            out.append((h, r, t))
    return out
