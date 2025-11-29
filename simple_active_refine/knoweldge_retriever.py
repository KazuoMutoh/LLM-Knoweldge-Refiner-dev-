"""
hornルールで記述された知識取得ルールに基づき外部情報源から知識を取得するモジュール
"""
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import openai

from simple_active_refine.amie import AmieRules
from simple_active_refine.util import get_logger
from settings import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
logger = get_logger('KnowledgeRetriever')

class Entity(BaseModel):
    id: str  # Unique entity identifier
    label: str
    description_short: str
    description: str | None = None
    source: str | None = None

class Relation(BaseModel):
    id: str  # Unique relation identifier
    label: str
    description_short: str
    description: str | None = None
    source: str | None = None

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    source:str | None = None

class Entities(BaseModel):
    entities: List[Entity]

class RetrievedKnowledge(BaseModel):
    triples: List[Triple]
    entities: List[Entity]

class TextAttributedKnoweldgeGraph:
    """
    テキスト情報を持つ知識グラフ。
    
    機能:
    - train.txtなどからの初期化
    - 永続化（pickle/JSON）によるキャッシュ
    - ベクトル/キーワード/ハイブリッド検索
    - Entity/Tripleインスタンスの管理
    """

    def __init__(self, dir_triples: str, cache_dir: Optional[str] = None, embedding_model: str = "text-embedding-3-small"):
        """
        Args:
            dir_triples: トリプルデータのディレクトリ
            cache_dir: キャッシュディレクトリ（Noneの場合はdir_triples/.cache）
            embedding_model: OpenAI埋め込みモデル名
        """
        self.dir_triples = dir_triples
        self.cache_dir = cache_dir or os.path.join(dir_triples, ".cache")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # キャッシュファイルのパス
        self.kg_cache_path = os.path.join(self.cache_dir, "kg_data.pkl")
        self.chroma_persist_dir = os.path.join(self.cache_dir, "chroma_db")
        
        # 埋め込みモデルの初期化
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # ChromaDBクライアントの初期化
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        
        # キャッシュがあれば読み込み、なければ初期化
        if os.path.exists(self.kg_cache_path):
            logger.info(f"Loading knowledge graph from cache: {self.kg_cache_path}")
            self._load_from_cache()
        else:
            logger.info(f"Initializing knowledge graph from {dir_triples}")
            self._initialize_from_files()
            self._save_to_cache()
        
        # ベクトルDBの初期化
        self._initialize_vector_db()
        
        # BM25インデックスの初期化
        self._initialize_bm25_index()

    def _initialize_from_files(self):
        """ファイルから知識グラフを初期化"""
        self.triples = {}
        for data_type in ['train', 'valid', 'test']:
            file_path = os.path.join(self.dir_triples, f"{data_type}.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as fin:
                    lines = fin.readlines()
                self.triples[data_type] = [tuple(line.strip().split('\t')) for line in lines]
            else:
                logger.info(f'File {file_path} does not exist. Initializing empty list for {data_type} triples.')
                self.triples[data_type] = []
                
        # エンティティテキスト情報の読み込み
        if os.path.exists(os.path.join(self.dir_triples, 'entity2text.txt')):
            with open(os.path.join(self.dir_triples, 'entity2text.txt'), 'r', encoding='utf-8') as fin:
                lines = fin.readlines()
            self.entity_texts = {}
            for line in lines:
                parts = line.strip().split('\t')
                entity = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                self.entity_texts[entity] = {'id': entity, 'label': entity, 'description_short': text, 'description': ""}
        else:
            self.entity_texts = {}
        
        # リレーションテキスト情報の読み込み
        if os.path.exists(os.path.join(self.dir_triples, 'relation2text.txt')):
            with open(os.path.join(self.dir_triples, 'relation2text.txt'), 'r', encoding='utf-8') as fin:
                lines = fin.readlines()
            self.relation_texts = {}
            for line in lines:
                parts = line.strip().split('\t')
                relation = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                self.relation_texts[relation] = {'label': relation, 'description': text}
        else:
            self.relation_texts = {}

        # エンティティ長文説明の読み込み
        if os.path.exists(os.path.join(self.dir_triples, 'entity2textlong.txt')):
            with open(os.path.join(self.dir_triples, 'entity2textlong.txt'), 'r', encoding='utf-8') as fin:
                lines = fin.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                entity = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                if entity in self.entity_texts:
                    self.entity_texts[entity]['description'] = text
                else:
                    self.entity_texts[entity] = {'id': entity, 'label': entity, 'description_short': "", 'description': text}

    def _save_to_cache(self):
        """知識グラフをキャッシュに保存"""
        cache_data = {
            'triples': self.triples,
            'entity_texts': self.entity_texts,
            'relation_texts': self.relation_texts
        }
        with open(self.kg_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Knowledge graph saved to cache: {self.kg_cache_path}")

    def _load_from_cache(self):
        """キャッシュから知識グラフを読み込み"""
        with open(self.kg_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        self.triples = cache_data['triples']
        self.entity_texts = cache_data['entity_texts']
        self.relation_texts = cache_data['relation_texts']
        logger.info(f"Knowledge graph loaded from cache: {self.kg_cache_path}")

    def _initialize_vector_db(self):
        """ベクトルDBの初期化または読み込み"""
        collection_name = "entities"
        
        try:
            # 既存のコレクションを取得
            self.entity_collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Loaded existing vector DB collection: {collection_name}")
        except Exception:
            # コレクションが存在しない場合は作成
            logger.info(f"Creating new vector DB collection: {collection_name}")
            self.entity_collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # エンティティを埋め込みと共に追加
            if self.entity_texts:
                entity_ids = []
                documents = []
                metadatas = []
                
                logger.info(f"Preparing {len(self.entity_texts)} entities for embedding...")
                for entity_id, entity_data in self.entity_texts.items():
                    entity_ids.append(entity_id)
                    # 検索用テキスト: description_short + description (ラベルは除外)
                    search_text = f"{entity_data.get('description_short', '')} {entity_data.get('description', '')}"
                    documents.append(search_text.strip())
                    metadatas.append({
                        'label': entity_data.get('label', ''),
                        'description_short': entity_data.get('description_short', ''),
                        'description': entity_data.get('description', '')
                    })
                
                # 埋め込みを生成（バッチ処理 + プログレスバー）
                logger.info("Generating embeddings (this may take a while)...")
                batch_size = 100  # OpenAI API制限を考慮したバッチサイズ
                embeddings_list = []
                
                for i in tqdm(range(0, len(documents), batch_size), 
                             desc="Embedding batches",
                             unit="batch"):
                    batch_docs = documents[i:i + batch_size]
                    batch_embeddings = self.embeddings.embed_documents(batch_docs)
                    embeddings_list.extend(batch_embeddings)
                
                # コレクションに追加
                logger.info("Adding embeddings to vector database...")
                self.entity_collection.add(
                    ids=entity_ids,
                    embeddings=embeddings_list,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(entity_ids)} entities to vector DB")

    def _initialize_bm25_index(self):
        """BM25インデックスの初期化"""
        # 各エンティティの検索テキストをトークン化
        self.entity_id_list = list(self.entity_texts.keys())
        
        if not self.entity_id_list:
            # エンティティが空の場合は空のBM25インデックスを作成
            logger.warning("No entities found, initializing empty BM25 index")
            self.bm25 = None
            return
            
        tokenized_corpus = []
        
        for entity_id in self.entity_id_list:
            entity_data = self.entity_texts[entity_id]
            search_text = f"{entity_data.get('description_short', '')} {entity_data.get('description', '')}"
            # 簡易トークン化（スペースで分割）
            tokens = search_text.lower().split()
            if not tokens:  # 空のトークンリストの場合はダミートークンを追加
                tokens = [entity_id.lower()]
            tokenized_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Initialized BM25 index with {len(self.entity_id_list)} entities")

    def search_entities_by_text(
        self, 
        query: str, 
        method: str = "hybrid", 
        top_k: int = 10,
        vector_weight: float = 0.5,
        return_scores: bool = False,
        min_score: float = 0.0
    ) -> Union[List[Entity], List[Tuple[Entity, float]]]:
        """
        テキストクエリでエンティティを検索
        
        Args:
            query: 検索クエリ
            method: 検索方法 ("vector", "keyword", "hybrid")
            top_k: 返す結果の数
            vector_weight: ハイブリッド検索時のベクトル検索の重み（0-1）
            return_scores: Trueの場合、(Entity, score)のタプルのリストを返す
            min_score: 最小スコア閾値（この値未満の結果は除外）
            
        Returns:
            検索結果のEntityリストまたは(Entity, score)のタプルのリスト
        """
        if method == "vector":
            return self._vector_search(query, top_k, return_scores, min_score)
        elif method == "keyword":
            return self._keyword_search(query, top_k, return_scores, min_score)
        elif method == "hybrid":
            return self._hybrid_search(query, top_k, vector_weight, return_scores, min_score)
        else:
            raise ValueError(f"Unknown search method: {method}")

    def _vector_search(self, query: str, top_k: int, return_scores: bool = False, min_score: float = 0.0) -> Union[List[Entity], List[Tuple[Entity, float]]]:
        """ベクトル検索"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.entity_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2  # 余裕を持って多めに取得（min_scoreフィルタリングのため）
        )
        
        entities_with_scores = []
        if results['ids'] and len(results['ids']) > 0:
            ids = results['ids'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)
            distances = results['distances'][0] if results['distances'] else [0.0] * len(ids)
            
            for entity_id, metadata, distance in zip(ids, metadatas, distances):
                # cosine距離を類似度スコアに変換（1 - distance、1が最高スコア）
                score = 1.0 - distance
                
                # スコアフィルタリング
                if score < min_score:
                    continue
                    
                entity_data = self.entity_texts.get(entity_id, {})
                entity = Entity(
                    id=entity_id,
                    label=entity_id,  # labelはidと同じ値を使用
                    description_short=entity_data.get('description_short', ''),
                    description=entity_data.get('description', '')
                )
                entities_with_scores.append((entity, score))
        
        # top_k個に制限
        entities_with_scores = entities_with_scores[:top_k]
        
        if return_scores:
            return entities_with_scores
        else:
            return [entity for entity, score in entities_with_scores]

    def _keyword_search(self, query: str, top_k: int, return_scores: bool = False, min_score: float = 0.0) -> Union[List[Entity], List[Tuple[Entity, float]]]:
        """キーワード検索（BM25）"""
        if self.bm25 is None or not self.entity_id_list:
            # BM25インデックスがない場合は空の結果を返す
            return [] if not return_scores else []
            
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # スコアを正規化（0-1の範囲に）
        max_score = float(max(scores)) if len(scores) > 0 else 1.0
        normalized_scores = [float(score) / max_score if max_score > 0 else 0.0 for score in scores]
        
        # スコアと一緒にソート
        scored_indices = [(i, normalized_scores[i]) for i in range(len(normalized_scores))]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        entities_with_scores = []
        for idx, score in scored_indices:
            # スコアフィルタリング
            if float(score) < min_score:
                continue
                
            entity_id = self.entity_id_list[idx]
            entity_data = self.entity_texts.get(entity_id, {})
            entity = Entity(
                id=entity_id,
                label=entity_id,  # labelはidと同じ値を使用
                description_short=entity_data.get('description_short', ''),
                description=entity_data.get('description', '')
            )
            entities_with_scores.append((entity, float(score)))
            
            # top_k個に制限
            if len(entities_with_scores) >= top_k:
                break
        
        if return_scores:
            return entities_with_scores
        else:
            return [entity for entity, score in entities_with_scores]

    def _hybrid_search(self, query: str, top_k: int, vector_weight: float, return_scores: bool = False, min_score: float = 0.0) -> Union[List[Entity], List[Tuple[Entity, float]]]:
        """ハイブリッド検索（ベクトル + キーワード）"""
        # ベクトル検索の結果（スコア付き）
        vector_results = self._vector_search(query, top_k * 2, return_scores=True, min_score=0.0)
        vector_scores = {entity.id: score for entity, score in vector_results}
        
        # キーワード検索の結果（スコア付き）
        keyword_results = self._keyword_search(query, top_k * 2, return_scores=True, min_score=0.0)
        keyword_scores = {entity.id: score for entity, score in keyword_results}
        
        # スコアを結合
        all_entity_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        hybrid_scores = {}
        for entity_id in all_entity_ids:
            v_score = vector_scores.get(entity_id, 0.0)
            k_score = keyword_scores.get(entity_id, 0.0)
            hybrid_scores[entity_id] = vector_weight * v_score + (1 - vector_weight) * k_score
        
        # スコアでソートしてフィルタリング
        scored_entities = []
        for entity_id, score in hybrid_scores.items():
            # スコアフィルタリング
            if score < min_score:
                continue
                
            entity_data = self.entity_texts.get(entity_id, {})
            entity = Entity(
                id=entity_id,
                label=entity_id,  # labelはidと同じ値を使用
                description_short=entity_data.get('description_short', ''),
                description=entity_data.get('description', '')
            )
            scored_entities.append((entity, score))
        
        # スコアでソートして上位top_k件を取得
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        scored_entities = scored_entities[:top_k]
        
        if return_scores:
            return scored_entities
        else:
            return [entity for entity, score in scored_entities]

    def search_similar_entities(
        self, 
        entity: Entity, 
        top_k: int = 10, 
        method: str = "hybrid", 
        return_scores: bool = False, 
        min_score: float = 0.0
    ) -> Union[List[Entity], List[Tuple[Entity, float]]]:
        """
        与えられたEntityに類似するエンティティを検索
        
        Args:
            entity: 検索クエリとなるEntity
            top_k: 返す結果の数
            method: 検索方法 ("vector", "keyword", "hybrid")
            return_scores: Trueの場合、(Entity, score)のタプルのリストを返す
            min_score: 最小スコア閾値（この値未満の結果は除外）
            
        Returns:
            類似エンティティのリスト（クエリエンティティ自身は除外）またはスコア付きのタプルのリスト
        """
        # エンティティ情報から検索クエリを構築（ラベルは除外）
        query = f"{entity.description_short} {entity.description or ''}"
        
        # 検索実行（余裕を持って多めに取得）
        results = self.search_entities_by_text(
            query, 
            method=method, 
            top_k=top_k + 5,  # 自身を除外するために余裕を持って取得
            return_scores=return_scores, 
            min_score=min_score
        )
        
        # クエリエンティティ自身を結果から除外
        if return_scores:
            # スコア付きの場合
            filtered_results = [(e, score) for e, score in results if e.id != entity.id]
            return filtered_results[:top_k]
        else:
            # スコアなしの場合
            filtered_results = [e for e in results if e.id != entity.id]
            return filtered_results[:top_k]

    def add_entities(self, entities: List[Entity]):
        """
        エンティティリストを知識グラフに追加
        
        Args:
            entities: 追加するEntityのリスト
        """
        if not entities:
            return
        
        logger.info(f"Adding {len(entities)} entities with embeddings...")
        
        # バッチ処理用のデータを準備
        entity_ids = []
        documents = []
        metadatas = []
        
        for entity in entities:
            # entity_textsに追加（labelはidと同じ値を使用）
            self.entity_texts[entity.id] = {
                'id': entity.id,
                'label': entity.id,
                'description_short': entity.description_short,
                'description': entity.description or ''
            }
            
            # ベクトルDB追加用データを準備（ラベルは除外）
            search_text = f"{entity.description_short} {entity.description or ''}"
            entity_ids.append(entity.id)
            documents.append(search_text)
            metadatas.append({
                'label': entity.id,  # labelはidと同じ値を使用
                'description_short': entity.description_short,
                'description': entity.description or ''
            })
        
        # 埋め込みを生成（バッチ処理 + プログレスバー）
        batch_size = 100
        embeddings_list = []
        
        if len(documents) > batch_size:
            # 大量のエンティティの場合はプログレスバー表示
            for i in tqdm(range(0, len(documents), batch_size), 
                         desc="Embedding entities",
                         unit="batch"):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_docs)
                embeddings_list.extend(batch_embeddings)
        else:
            # 少数の場合はプログレスバーなし
            embeddings_list = self.embeddings.embed_documents(documents)
        
        # ベクトルDBに追加
        self.entity_collection.add(
            ids=entity_ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        # BM25インデックスを再構築
        self._initialize_bm25_index()
        
        # キャッシュを更新
        self._save_to_cache()
        
        logger.info(f"Added {len(entities)} entities to knowledge graph")

    def get_all_entities(self) -> List[Entity]:
        """
        すべてのエンティティをEntityインスタンスとして返す
        
        Returns:
            全EntityのList
        """
        entities = []
        for entity_id, entity_data in self.entity_texts.items():
            entities.append(Entity(
                id=entity_id,
                label=entity_id,  # labelはidと同じ値を使用
                description_short=entity_data.get('description_short', ''),
                description=entity_data.get('description', '')
            ))
        return entities

    def get_all_triples(self, data_type: Optional[str] = None) -> List[Triple]:
        """
        トリプルをTripleインスタンスとして返す
        
        Args:
            data_type: 'train', 'valid', 'test' のいずれか。Noneの場合は全て
            
        Returns:
            TripleのList
        """
        triples = []
        
        if data_type:
            triple_tuples = self.triples.get(data_type, [])
            for subj, pred, obj in triple_tuples:
                triples.append(Triple(subject=subj, predicate=pred, object=obj))
        else:
            # すべてのデータタイプから取得
            for dt in ['train', 'valid', 'test']:
                triple_tuples = self.triples.get(dt, [])
                for subj, pred, obj in triple_tuples:
                    triples.append(Triple(subject=subj, predicate=pred, object=obj, source=dt))
        
        return triples

    def add_triples(self, triples: List[Triple], data_type: str = 'train'):
        """
        トリプルを知識グラフに追加
        
        Args:
            triples: 追加するTripleのリスト
            data_type: 追加先のデータタイプ ('train', 'valid', 'test')
        """
        if data_type not in self.triples:
            self.triples[data_type] = []
        
        for triple in triples:
            triple_tuple = (triple.subject, triple.predicate, triple.object)
            if triple_tuple not in self.triples[data_type]:
                self.triples[data_type].append(triple_tuple)
        
        # キャッシュを更新
        self._save_to_cache()
        
        logger.info(f"Added {len(triples)} triples to {data_type} dataset")

    def add_retrieved_knowledge(self, retrieved_knowledge: RetrievedKnowledge, data_type: str = 'train'):
        """
        RetrievedKnowledgeインスタンスを知識グラフに追加
        
        エンティティとトリプルの両方を一括で追加します。
        
        Args:
            retrieved_knowledge: RetrievedKnowledgeインスタンス
            data_type: トリプルの追加先データタイプ ('train', 'valid', 'test')
        """
        # エンティティを追加
        if retrieved_knowledge.entities:
            self.add_entities(retrieved_knowledge.entities)
            logger.info(f"Added {len(retrieved_knowledge.entities)} entities from RetrievedKnowledge")
        
        # トリプルを追加
        if retrieved_knowledge.triples:
            self.add_triples(retrieved_knowledge.triples, data_type=data_type)
            logger.info(f"Added {len(retrieved_knowledge.triples)} triples from RetrievedKnowledge")

    def add_retrieved_knowledge_list(self, retrieved_knowledge_list: List[RetrievedKnowledge], data_type: str = 'train'):
        """
        RetrievedKnowledgeインスタンスのリストを知識グラフに追加
        
        複数のRetrievedKnowledgeを一括で追加します。
        
        Args:
            retrieved_knowledge_list: RetrievedKnowledgeインスタンスのリスト
            data_type: トリプルの追加先データタイプ ('train', 'valid', 'test')
        """
        total_entities = 0
        total_triples = 0
        
        for retrieved_knowledge in retrieved_knowledge_list:
            if retrieved_knowledge.entities:
                total_entities += len(retrieved_knowledge.entities)
            if retrieved_knowledge.triples:
                total_triples += len(retrieved_knowledge.triples)
            
            self.add_retrieved_knowledge(retrieved_knowledge, data_type=data_type)
        
        logger.info(f"Added total of {total_entities} entities and {total_triples} triples from {len(retrieved_knowledge_list)} RetrievedKnowledge instances")

    def save_to_files(self):
        """知識グラフをファイルに保存（元のフォーマット）"""
        # トリプルの保存
        for data_type, triple_list in self.triples.items():
            file_path = os.path.join(self.dir_triples, f"{data_type}.txt")
            with open(file_path, 'w', encoding='utf-8') as fout:
                for subj, pred, obj in triple_list:
                    fout.write(f"{subj}\t{pred}\t{obj}\n")
        
        # エンティティテキストの保存
        entity2text_path = os.path.join(self.dir_triples, 'entities2text.txt')
        with open(entity2text_path, 'w', encoding='utf-8') as fout:
            for entity_id, entity_data in self.entity_texts.items():
                short_desc = entity_data.get('description_short', '')
                fout.write(f"{entity_id}\t{short_desc}\n")
        
        # エンティティ長文説明の保存
        entity2textlong_path = os.path.join(self.dir_triples, 'entity2textlong.txt')
        with open(entity2textlong_path, 'w', encoding='utf-8') as fout:
            for entity_id, entity_data in self.entity_texts.items():
                long_desc = entity_data.get('description', '')
                if long_desc:
                    fout.write(f"{entity_id}\t{long_desc}\n")
        
        logger.info(f"Knowledge graph saved to files in {self.dir_triples}")

class LLMKnowledgeRetriever:

    def __init__(self, 
                 kg:TextAttributedKnoweldgeGraph=None,
                 llm_model:str='gpt-4o', 
                 use_web_search:bool=True):
        """
        LLMを用いて知識を取得するリトリーバークラス

        Args:
            kg: TextAttributedKnoweldgeGraphインスタンス
            llm_model (str, optional): 使用するLLMモデル. Defaults to 'gpt-4o'.
            use_web_search (bool, optional): web search toolを使用するか. Defaults to True.
        """
        self.kg = kg
        self.llm_model = llm_model
        self.use_web_search = use_web_search
        
        # OpenAIクライアントを初期化
        import openai
        self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # フォールバック用のLangChain LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        
        logger.info(f"LLM initialized with model: {llm_model}, web_search: {use_web_search}")
    
    def retrieve_knowledge_for_entity(self, entity: Entity, list_relations: List[Relation]) -> RetrievedKnowledge:
        """
        entityに関する情報をwebから取得します
        
        処理の手順:
        1. LLMに、entityのdescriptionを参考に、list_relationsより、entityがheadもしくはtailになりうるrelationを1個以上選択させます
           すなわち、entity, relation, ?もしくは、?, relation, entityとなるような組み合わせを見つけさせます
        2. 1で選択したrelationのそれぞれに対して、?にあたる情報をwebから探し、Entityクラスのインスタンスretrieved_entityとして情報を保持します
        3. entity, relation, retrieved_entityとentityの情報をRetrievedKnowledgeのインスタンスに格納して返します
        
        Args:
            entity: 検索対象のEntity
            list_relations: 候補となるRelationのリスト
            
        Returns:
            RetrievedKnowledge: 取得された知識（トリプルとエンティティ）
        """
        logger.info(f"Retrieving knowledge for entity: {entity.id} ({entity.label})")
        
        # ステップ1: LLMにrelationを選択させる
        relation_descriptions = "\n".join([
            f"- {rel.id}: {rel.description_short} ({rel.description or 'No detailed description'})"
            for rel in list_relations
        ])
        
        relation_selection_prompt = f"""
You are a knowledge graph expert. Your task is to identify which relations are relevant for a given entity.

ENTITY INFORMATION:
- ID: {entity.id}
- Label: {entity.label}
- Description (short): {entity.description_short}
- Description (long): {entity.description or 'N/A'}

AVAILABLE RELATIONS:
{relation_descriptions}

TASK: Select 1 or more relations from the list above where this entity could be either the HEAD (subject) or TAIL (object) of a triple.
- For HEAD: the pattern would be (entity, relation, ?)
- For TAIL: the pattern would be (?, relation, entity)

Consider the entity's description carefully and select only relations that make semantic sense.

OUTPUT: Return a JSON object with this exact format:
{{
    "selected_relations": [
        {{
            "relation_id": "relation_id_from_list",
            "position": "head" or "tail",
            "reasoning": "Brief explanation why this relation is relevant"
        }}
    ]
}}

IMPORTANT:
- Select at least 1 relation, but no more than 5
- Only use relation IDs from the provided list
- Be specific about whether the entity is head or tail
- Return only valid JSON, no additional text
"""

        try:
            # LLMでrelationを選択
            logger.info("Requesting LLM to select relevant relations")
            llm_response = self.llm.invoke(relation_selection_prompt)
            response_text = llm_response.content
            
            logger.debug(f"Relation selection response: {response_text[:300]}...")
            
            # JSONをパース
            import json
            import re
            
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.warning("No valid JSON found in relation selection response")
                return RetrievedKnowledge(triples=[], entities=[])
            
            json_str = json_match.group(0)
            selection_result = json.loads(json_str)
            selected_relations = selection_result.get('selected_relations', [])
            
            if not selected_relations:
                logger.warning("No relations selected by LLM")
                return RetrievedKnowledge(triples=[], entities=[])
            
            logger.info(f"LLM selected {len(selected_relations)} relations")
            
            # ステップ2 & 3: 各relationについてwebから情報を取得
            all_triples = []
            all_entities = []
            entity_counter = 1
            
            for rel_info in selected_relations:
                relation_id = rel_info.get('relation_id')
                position = rel_info.get('position')
                reasoning = rel_info.get('reasoning', 'N/A')
                
                logger.info(f"Processing relation: {relation_id} (position: {position})")
                logger.debug(f"Reasoning: {reasoning}")
                
                # relationの詳細情報を取得
                relation = next((r for r in list_relations if r.id == relation_id), None)
                if not relation:
                    logger.warning(f"Relation {relation_id} not found in list_relations")
                    continue
                
                # トリプルパターンの説明
                if position == "head":
                    triple_pattern = f"({entity.label}, {relation.label}, ?)"
                    search_instruction = f"Find entities that could be the OBJECT (tail) in the relation '{relation.label}' with subject '{entity.label}'"
                else:  # tail
                    triple_pattern = f"(?, {relation.label}, {entity.label})"
                    search_instruction = f"Find entities that could be the SUBJECT (head) in the relation '{relation.label}' with object '{entity.label}'"
                
                # webから情報を検索
                knowledge_search_prompt = f"""
You are a knowledge retrieval expert. Your task is to find real-world entities that complete a knowledge graph triple.

ENTITY INFORMATION:
- Label: {entity.label}
- Description (short): {entity.description_short}
- Description (long): {entity.description or 'N/A'}

RELATION INFORMATION:
- Label: {relation.label}
- Description (short): {relation.description_short}
- Description (long): {relation.description or 'N/A'}

TRIPLE PATTERN: {triple_pattern}

TASK: {search_instruction}

Find 1-3 real-world entities that would complete this triple pattern based on factual information.

OUTPUT: Return a JSON object with this exact format:
{{
    "found_entities": [
        {{
            "label": "Entity name or title exactly as it appears (like rdfs:label, e.g., 'St Andrews', 'University of St Andrews')",
            "description_short": "Entity name or title exactly as it appears (like rdfs:label, e.g., 'St Andrews', 'University of St Andrews')",
            "description": "Detailed description (2-3 sentences) or null",
            "source": "Complete URL source (e.g., https://en.wikipedia.org/wiki/...)"
        }}
    ]
}}

REQUIREMENTS:
1. Provide 1-3 entities that accurately complete the triple pattern
2. Use REAL entity names, not placeholders like [Person Name] or [City Name]
3. Provide factual, verifiable information
4. Include COMPLETE URL sources (must start with http:// or https://)
5. Ensure descriptions are informative and specific
6. Return only valid JSON, no additional text
"""

                try:
                    # Web検索を使用する場合はOpenAI responses APIを使用
                    if self.use_web_search:
                        logger.info(f"Using web search for relation {relation_id}")
                        try:
                            response = self.openai_client.responses.create(
                                model=self.llm_model,
                                tools=[{"type": "web_search_preview"}],
                                tool_choice={"type": "web_search_preview"},
                                input=knowledge_search_prompt,
                                store=True
                            )
                            
                            if response.status == "completed":
                                search_response_text = response.output_text
                                logger.info(f"Web search response received for relation {relation_id}")
                            else:
                                logger.warning(f"Web search failed for relation {relation_id}, falling back to standard LLM")
                                llm_resp = self.llm.invoke(knowledge_search_prompt)
                                search_response_text = llm_resp.content
                        except Exception as web_err:
                            logger.warning(f"Web search error for relation {relation_id}: {web_err}, falling back to standard LLM")
                            llm_resp = self.llm.invoke(knowledge_search_prompt)
                            search_response_text = llm_resp.content
                    else:
                        # Web検索を使用しない場合は標準LLMを使用
                        llm_resp = self.llm.invoke(knowledge_search_prompt)
                        search_response_text = llm_resp.content
                    
                    logger.debug(f"Knowledge search response: {search_response_text[:300]}...")
                    
                    # JSONをパース
                    json_match = re.search(r'\{[\s\S]*\}', search_response_text)
                    if not json_match:
                        logger.warning(f"No valid JSON found in knowledge search response for relation {relation_id}")
                        continue
                    
                    json_str = json_match.group(0)
                    search_result = json.loads(json_str)
                    found_entities = search_result.get('found_entities', [])
                    
                    logger.info(f"Found {len(found_entities)} entities for relation {relation_id}")
                    
                    # 各エンティティについてトリプルを作成
                    for found_entity_data in found_entities:
                        # 新しいエンティティを作成
                        new_entity_id = f"e{entity_counter}"
                        entity_counter += 1
                        
                        retrieved_entity = Entity(
                            id=new_entity_id,
                            label=new_entity_id,
                            description_short=found_entity_data.get('description_short', ''),
                            description=found_entity_data.get('description'),
                            source=found_entity_data.get('source', '')
                        )
                        all_entities.append(retrieved_entity)
                        
                        # トリプルを作成
                        if position == "head":
                            # (entity, relation, retrieved_entity)
                            triple = Triple(
                                subject=entity.id,
                                predicate=relation.id,
                                object=new_entity_id,
                                source=found_entity_data.get('source', '')
                            )
                        else:  # tail
                            # (retrieved_entity, relation, entity)
                            triple = Triple(
                                subject=new_entity_id,
                                predicate=relation.id,
                                object=entity.id,
                                source=found_entity_data.get('source', '')
                            )
                        all_triples.append(triple)
                        
                        logger.info(f"Created triple: ({triple.subject}, {triple.predicate}, {triple.object})")
                
                except Exception as e:
                    logger.error(f"Error retrieving knowledge for relation {relation_id}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # RetrievedKnowledgeを作成して返す
            retrieved_knowledge = RetrievedKnowledge(triples=all_triples, entities=all_entities)
            logger.info(f"Retrieved {len(all_triples)} triples and {len(all_entities)} entities for entity {entity.id}")
            
            return retrieved_knowledge
            
        except Exception as e:
            logger.error(f"Error in retrieve_knowledge_for_entity for entity {entity.id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return RetrievedKnowledge(triples=[], entities=[])
        
    def retrieve_knowledge(self, target_triples:List[Tuple], rules:AmieRules) -> RetrievedKnowledge:
        """
        rulesに基づき知識を取得する
        """
        rules = rules.rules
        list_retrieved_knowledge = []
        for triple in tqdm(target_triples,desc="Retrieving knowledge for target triples", unit="triple"):
            for rule in rules:
                    
                head_pattern = rule.head

                dict_head = {}
                # Entity情報をKGから取得
                entity_s = self.kg.entity_texts.get(triple[0], {})
                dict_head[head_pattern.s] = {
                    'label': triple[0],
                    'description_short': entity_s.get('description_short', ''),
                    'description': entity_s.get('description', '')
                }

                relation_p = self.kg.relation_texts.get(triple[1], {})
                dict_head[head_pattern.p] = {
                    'label': triple[1], 
                    'description_short': relation_p.get('label', triple[1]),
                    'description': relation_p.get('description', '')
                }
                
                entity_o = self.kg.entity_texts.get(triple[2], {})
                dict_head[head_pattern.o] = {
                    'label': triple[2], 
                    'description_short': entity_o.get('description_short', ''),
                    'description': entity_o.get('description', '')
                } 
                
                descriptions_for_target_triple = ''.join([
                    f"{dict_head[key]['label']}: **{dict_head[key]['description_short']}**\n{dict_head[key]['description']}" 
                    for key in dict_head.keys()
                ])


                body_str = ' , '.join([tp.to_tsv() for tp in rule.body])
                head_str = f'{rule.head.to_tsv()}'
                descriptions_for_rule = f'{body_str}  =>  {head_str}\n'
                        
                prompt = f"""
                You are a knowledge graph completion expert. Your task is to find supporting information for a target triple by retrieving triples that match the body patterns of a Horn rule.

                TARGET TRIPLE (this is the HEAD of the Horn rule):
                Subject: {triple[0]} 
                Predicate: {triple[1]}
                Object: {triple[2]} 

                HORN RULE BODY PATTERNS (find information matching these patterns):
                {body_str}

                TASK: The target triple represents the conclusion (head) of a Horn rule. You need to find real-world information that matches the BODY PATTERNS: "{body_str}". These body patterns, when satisfied, would logically support the target triple.

                Variable Mapping for this rule:
                - ?a = {triple[0]} ({dict_head[head_pattern.s]['description_short']} + '\n\n' + {dict_head[head_pattern.s]['description']})
                - ?b = {triple[2]} ({dict_head[head_pattern.o]['description_short']} + '\n\n' + {dict_head[head_pattern.o]['description']})
                - ?c, ?d, etc. = new entity(s) to be discovered through web search

                Instructions:
                1. Focus on finding triples that match the specific patterns in "{body_str}"
                2. Each pattern in the body should be instantiated with REAL entities/relationships, not placeholders
                3. The entities in these patterns should be related to the target triple entities
                4. Use specific entity names, locations, and descriptions based on actual facts
                5. Do NOT create the target triple itself - only find supporting evidence from the body patterns
                6. Avoid generic placeholders like [Person's Name], [City Name], etc. - use actual names

                OUTPUT: Return valid JSON matching this schema:
                {{
                    "triples": [
                        {{
                            "subject": "entity_id",
                            "predicate": "relationship", 
                            "object": "entity_id",
                            "source": "source_info"
                        }}
                    ],
                    "entities": [
                        {{
                            "id": "e1",
                            "label": "Entity proper name",
                            "description_short": "Entity proper name (same as label, like rdfs:label)",
                            "description": "Detailed description or null",
                            "source": "source_info"
                        }}
                    ]
                }}

                EXAMPLE OUTPUT:
                For a rule "?a /people/person/place_of_birth ?c, ?c /location/location/contains ?b => ?a /people/person/nationality ?b" with target triple "Barack Obama nationality USA":
                Where ?a=/m/02mjmr (Barack Obama), ?b=/m/09c7w0 (USA), ?c=new entity to be found (Hawaii)
                {{
                    "triples": [
                        {{
                            "subject": "/m/02mjmr",
                            "predicate": "/people/person/place_of_birth",
                            "object": "e1",
                            "source": "https://en.wikipedia.org/wiki/Barack_Obama"
                        }},
                        {{
                            "subject": "e1", 
                            "predicate": "/location/location/contains",
                            "object": "/m/09c7w0",
                            "source": "https://www.census.gov/topics/geography/about/state-local-gov.html"
                        }}
                    ],
                    "entities": [
                        {{
                            "id": "e1",
                            "label": "Hawaii",
                            "description_short": "Hawaii", 
                            "description": "Hawaii is the 50th U.S. state, located in the Pacific Ocean. It consists of volcanic islands and is known for its tropical climate.",
                            "source": "https://en.wikipedia.org/wiki/Hawaii"
                        }}
                    ]
                }}

                Requirements:
                1. Find 1-3 triples that match the body patterns: {body_str}
                2. For existing entities (?a, ?b from target triple), use their actual IDs (e.g., /m/02mjmr, /m/09c7w0)
                3. For NEW entities (?c, ?d, etc. discovered during search), assign sequential IDs: e1, e2, e3, etc.
                4. Set label and description_short to the entity's proper name (NOT id)
                5. Do NOT include the target triple itself
                6. Ensure the found triples correspond to the body patterns of the Horn rule
                7. Provide source information as COMPLETE URLs (e.g., https://en.wikipedia.org/wiki/Hawaii, https://www.census.gov/...)
                8. Use REAL entity names and descriptions, avoid generic placeholders like [Person's Name] or [City Name]
                9. Return only valid JSON, no explanations
                """

                try:
                    logger.info(f"Sending prompt to LLM for triple {triple}")
                    logger.debug(f"Prompt: {prompt}")  # First 200 chars
                    
                    # Web検索を使用する場合はOpenAI responses APIを使用
                    if self.use_web_search:
                        logger.warning(f"Using web search for knowledge retrieval - ensure accurate URL sources are provided in response for triple {triple}")
                        try:
                            response = self.openai_client.responses.create(
                                model=self.llm_model,
                                tools=[{"type": "web_search_preview"}],
                                tool_choice={"type": "web_search_preview"},
                                input=prompt,
                                store=True
                            )
                            
                            if response.status == "completed":
                                response_text = response.output_text
                                logger.info(f"Web search response received for triple {triple} - verifying URL sources in response")
                            else:
                                logger.warning(f"Web search failed for triple {triple}, falling back to standard LLM without web sources")
                                # フォールバック: 標準LLMを使用
                                llm_response = self.llm.invoke(prompt)
                                response_text = llm_response.content
                        except Exception as web_err:
                            logger.warning(f"Web search error for triple {triple}: {web_err}, falling back to standard LLM without web sources")
                            # フォールバック: 標準LLMを使用
                            llm_response = self.llm.invoke(prompt)
                            response_text = llm_response.content
                    else:
                        # Web検索を使用しない場合は標準LLMを使用
                        llm_response = self.llm.invoke(prompt)
                        response_text = llm_response.content
                    
                    logger.debug(f"Raw LLM response: {response_text[:300]}...")
                    
                    # JSONパーシングでRetrievedKnowledgeを作成
                    try:
                        import json
                        import re
                        
                        # JSON部分を抽出（マークダウンブロックや余分なテキストを除去）
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        if json_match:
                            json_str = json_match.group(0)
                            parsed_data = json.loads(json_str)
                            
                            # RetrievedKnowledgeオブジェクトを作成
                            triples = []
                            entities = []
                            
                            # トリプルをパーシング
                            for t in parsed_data.get('triples', []):
                                triples.append(Triple(
                                    subject=t.get('subject', ''),
                                    predicate=t.get('predicate', ''),
                                    object=t.get('object', ''),
                                    source=t.get('source', '')
                                ))
                            
                            # エンティティをパーシング
                            for e in parsed_data.get('entities', []):
                                entities.append(Entity(
                                    id=e.get('id', ''),
                                    label=e.get('id', ''),  # labelをidと同じ値に設定
                                    description_short=e.get('description_short', ''),
                                    description=e.get('description'),
                                    source=e.get('source', '')
                                ))
                            
                            retrieved_knowledge = RetrievedKnowledge(triples=triples, entities=entities)
                            list_retrieved_knowledge.append(retrieved_knowledge)
                            logger.info(f"Successfully parsed knowledge for triple {triple}: {len(triples)} triples, {len(entities)} entities")
                            #logger.info(retrieved_knowledge)
                        else:
                            logger.warning(f"No valid JSON found in response for triple {triple}")
                            empty_knowledge = RetrievedKnowledge(triples=[], entities=[])
                            list_retrieved_knowledge.append(empty_knowledge)
                            
                    except json.JSONDecodeError as json_err:
                        logger.error(f"JSON parsing error for triple {triple}: {json_err}")
                        logger.debug(f"Failed to parse: {response_text}")
                        empty_knowledge = RetrievedKnowledge(triples=[], entities=[])
                        list_retrieved_knowledge.append(empty_knowledge)
                        
                except Exception as e:
                    logger.error(f"Error retrieving knowledge for triple {triple}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # エラー時は空のRetrievedKnowledgeを追加
                    empty_knowledge = RetrievedKnowledge(triples=[], entities=[])
                    list_retrieved_knowledge.append(empty_knowledge)

        return list_retrieved_knowledge


class KnowledgeRefiner:
    """
    TextAttributedKnowledgeGraphクラスのインスタンスとして表現される知識グラフをRefineします。
    """
    
    def __init__(self, kg: TextAttributedKnoweldgeGraph):
        """
        Args:
            kg: Refineの対象となる知識グラフ
        """
        self.kg = kg
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0.1)
        logger.info("KnowledgeRefiner initialized")
    
    def find_same_entity(self, entity: Entity, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Entity]:
        """
        entityと同じkg内のentityを探しList[Entity]として返す。
        
        処理の手順:
        1. TextAttributedKnowledgeGraphのsearch_similar_entitiesで類似のentityを検索する
        2. 検索されたentityの内、引数で与えられたentityと同一のものがあるかLLMに判断させる
        
        Args:
            entity: 検索対象のEntity
            top_k: search_similar_entitiesで取得する候補数
            similarity_threshold: 類似度の閾値（この値以上のものを候補として取得）
        
        Returns:
            同一と判断されたEntityのリスト（見つからない場合は空のリスト）
        """
        logger.info(f"Finding same entity for: {entity.id} ({entity.label})")
        
        # ステップ1: search_similar_entitiesで類似エンティティを検索
        similar_entities = self.kg.search_similar_entities(
            entity=entity,
            top_k=top_k,
            method="hybrid",
            return_scores=True,
            min_score=similarity_threshold
        )
        
        if not similar_entities:
            logger.info(f"No similar entities found for {entity.id}")
            return []
        
        logger.info(f"Found {len(similar_entities)} similar entities for {entity.id}")
        
        # ステップ2: LLMで同一性を判断
        same_entities = []
        
        for candidate_entity, similarity_score in similar_entities:
            logger.debug(f"Checking candidate: {candidate_entity.id} (similarity: {similarity_score:.3f})")
            
            # LLMに同一性を判断させるプロンプト
            prompt = f"""
You are an entity matching expert. Your task is to determine if two entities represent the SAME real-world entity.

QUERY ENTITY:
- Description (short): {entity.description_short}
- Description (long): {entity.description or 'N/A'}

CANDIDATE ENTITY:
- Description (short): {candidate_entity.description_short}
- Description (long): {candidate_entity.description or 'N/A'}

INSTRUCTIONS:
1. Compare ONLY the descriptions carefully (IDs and labels have no semantic meaning)
2. Determine if they refer to the SAME real-world entity (not just similar entities)
3. Consider:
   - Do they describe the exact same person/place/thing/concept?
   - Are the descriptions consistent with each other?
   - Could they be different entities with similar descriptions?

IMPORTANT: 
- IDs and labels are just identifiers and have NO semantic meaning - ignore them completely
- Return "YES" ONLY if you are highly confident they are the SAME entity based on descriptions
- Return "NO" if they are different entities, even if similar
- When in doubt, return "NO"

OUTPUT: Return a JSON object with this exact format:
{{
    "is_same": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of your decision"
}}

Return only valid JSON, no additional text.
"""
            
            try:
                # LLMを呼び出し
                response = self.llm.invoke(prompt)
                response_text = response.content
                
                logger.debug(f"LLM response: {response_text[:200]}...")
                
                # JSONをパース
                import json
                import re
                
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                    result = json.loads(json_str)
                    
                    is_same = result.get('is_same', False)
                    confidence = result.get('confidence', 0.0)
                    reasoning = result.get('reasoning', 'N/A')
                    
                    logger.info(f"LLM judgment for {candidate_entity.id}: is_same={is_same}, confidence={confidence:.2f}")
                    logger.debug(f"Reasoning: {reasoning}")
                    
                    if is_same:
                        same_entities.append(candidate_entity)
                        logger.info(f"Entity {candidate_entity.id} identified as same as {entity.id}")
                else:
                    logger.warning(f"No valid JSON found in LLM response for {candidate_entity.id}")
                    
            except Exception as e:
                logger.error(f"Error during LLM judgment for {candidate_entity.id}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        logger.info(f"Found {len(same_entities)} same entities for {entity.id}")
        return same_entities