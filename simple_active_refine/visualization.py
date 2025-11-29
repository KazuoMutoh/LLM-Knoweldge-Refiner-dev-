from typing import Dict, List, Tuple, Optional
from pyvis.network import Network

Triple = Tuple[str, str, str]

def visualize_triples(
    grouped_triples: Dict[str, List[Triple]],
    output_html: str = "triples_graph.html",
    title: str = "Knowledge Graph (Triples)",
    width: str = "100%",
    height: str = "720px",
    directed: bool = True,
    physics: bool = True,
    edge_color: Optional[str] = None,
    node_shape: str = "dot",
    node_size: int = 18,
) -> Network:
    r"""可視化関数：色ごとに与えられたトリプル群を単一のグラフに描画する。

    数学的概要
    ----------
    入力は色 $c \in \mathcal{C}$ ごとに与えられたトリプル集合
    $\mathcal{T}_c \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$。
    全体集合 $\mathcal{T}=\bigcup_{c \in \mathcal{C}} \mathcal{T}_c$ の
    エンティティ集合を $\mathcal{V}=\{h,t \mid (h,r,t)\in\mathcal{T}\}$ とする。
    ノード色は写像 $C:\mathcal{V}\to\mathcal{C}$ を次で定義する：
    同一ノードが複数色に出現する場合、最初に観測された色を優先する
    （安定なトポロジ表示のため）。エッジは $(h,r,t)\in\mathcal{T}$ に対し
    $h\to t$ を張り、ラベルに $r$ を付与する。

    Args:
        grouped_triples (Dict[str, List[Triple]]): 
            キーがノード色（例: "#1f77b4", "orange" など）、値が
            その色で表示したいトリプルのリスト (head, relation, tail)。
        output_html (str, optional): 出力HTMLファイル名。デフォルトは "triples_graph.html"。
        title (str, optional): グラフタイトル。デフォルトは "Knowledge Graph (Triples)"。
        width (str, optional): ビュー幅（例 "100%", "1200px"）。デフォルト "100%"。
        height (str, optional): ビュー高さ。デフォルト "720px"。
        directed (bool, optional): 有向グラフとして描画するか。デフォルト True。
        physics (bool, optional): 物理シミュレーション有効化。デフォルト True。
        edge_color (Optional[str], optional): エッジの色を固定したい場合に指定。
            None の場合、各トリプルの属するグループ色を使用。
        node_shape (str, optional): ノード形状 (pyvis の shape)。デフォルト "dot"。
        node_size (int, optional): ノードサイズ。デフォルト 18。

    Returns:
        Network: pyvis の Network オブジェクト（HTMLは `output_html` に保存済み）。

    Notes:
        - 同一ノードが複数の色グループに出現した場合、最初に出現したグループの色を保持します。
        - 重複エッジや自己ループは pyvis 上では表示可能ですが、見やすさのため重複は抑制します。
        - 関係名（relation）はエッジラベルとして描画します。
    """
    # Network 準備
    net = Network(width=width, height=height, directed=directed, notebook=False)
    net.set_options("""
    {
    "nodes": {
        "font": { "size": 14 },
        "size": 7              
    },
    "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.9 } },
        "smooth": { "enabled": true }
    },
    "physics": {
        "enabled": true,
        "stabilization": { "iterations": 250 },
        "barnesHut": {
        "gravitationalConstant": -2000,   
        "springLength": 500,              
        "springConstant": 0.005
        }
    },
    "interaction": { "hover": true, "tooltipDelay": 120 }
    }
    """)

    # ノードとエッジの管理（重複追加防止）
    node_seen = {}  # node_id -> color
    edge_seen = set()  # (h, r, t)

    # 追加ヘルパ
    def _ensure_node(node_id: str, color: str):
        if node_id not in node_seen:
            node_seen[node_id] = color
            net.add_node(
                n_id=node_id,
                label=node_id,
                color=color,
                shape=node_shape,
                size=node_size,
                title=node_id,
            )

    # グループごとに処理
    for color, triples in grouped_triples.items():
        for h, r, t in triples:
            # ノードの色は最初に観測した色を固定
            if h not in node_seen:
                _ensure_node(h, color)
            if t not in node_seen:
                _ensure_node(t, color)

            # エッジ追加（重複抑止）
            key = (h, r, t)
            if key in edge_seen:
                continue
            edge_seen.add(key)

            net.add_edge(
                source=h,
                to=t,
                label=r,
                title=f"{h}  -[{r}]->  {t}",
                color=(edge_color if edge_color is not None else color),
            )

    # タイトルのための小さなハック（pyvis はタイトル欄がないのでHTMLに埋める）
    net.heading = title

    # HTML 出力
    net.write_html(output_html, open_browser=False, notebook=False) # 内部で write_html を呼ぶ
    return net
