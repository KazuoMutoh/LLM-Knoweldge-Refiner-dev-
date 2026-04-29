# Web取得トリプル事例まとめ（nationality / head_incident_v1）

- 作成日: 2026-02-01
- 実験出力: `/app/experiments/20260201_llm_policy_web_nationality_head_incident_v1_from_rerun1/arm_run_web_iter25`
- dataset_dir（参照用）: `/app/experiments/test_data_for_nationality_head_incident_v1_kgfit`

## 概要

- **raw取得** = `web_retrieved_triples.tsv`（Webから抽出された候補。フィルタ前）
- **採用（追加）** = `accepted_added_triples.tsv`（フィルタ後。実際に追加候補として採用）

- raw取得 合計: **2321 triples**
- 採用（追加）合計: **87 triples**

## relation分布（上位）

### 採用（追加） top10

| rank | relation | count |
|---:|---|---:|
| 1 | /people/person/places_lived./people/place_lived/location | 52 |
| 2 | /people/person/place_of_birth | 22 |
| 3 | /base/biblioness/bibs_location/country | 5 |
| 4 | /film/actor/film./film/performance/film | 2 |
| 5 | /award/award_nominee/award_nominations./award/award_nomination/award_nominee | 2 |
| 6 | /award/award_nominee/award_nominations./award/award_nomination/nominated_for | 2 |
| 7 | /award/award_winner/awards_won./award/award_honor/award_winner | 2 |

### raw取得 top10

| rank | relation | count |
|---:|---|---:|
| 1 | /base/biblioness/bibs_location/country | 860 |
| 2 | /people/person/places_lived./people/place_lived/location | 703 |
| 3 | /people/person/place_of_birth | 269 |
| 4 | /location/location/contains | 247 |
| 5 | /award/award_nominee/award_nominations./award/award_nomination/award_nominee | 63 |
| 6 | /film/film/country | 59 |
| 7 | /award/award_nominee/award_nominations./award/award_nomination/nominated_for | 48 |
| 8 | /award/award_winner/awards_won./award/award_honor/award_winner | 48 |
| 9 | /film/actor/film./film/performance/film | 24 |

## 事例（採用トリプル + provenance URL）

以下は `accepted_added_triples.tsv` から、`web_provenance.json` にURLが存在するものを中心に抜粋しています。

### /people/person/places_lived./people/place_lived/location

| iter | triple (text) | triple (id) | url | arm_id |
|---:|---|---|---|---|
| 1 | Bridget Fonda — /people/person/places_lived./people/place_lived/location — New York City | /m/01yd8v /people/person/places_lived./people/place_lived/location /m/02_286 | https://en.wikipedia.org/wiki/New_York_City | arm_268fbd2e5a65 |
| 1 | Bridget Fonda — /people/person/places_lived./people/place_lived/location — Los Angeles | /m/01yd8v /people/person/places_lived./people/place_lived/location /m/030qb3t | https://en.wikipedia.org/wiki/Los_Angeles | arm_268fbd2e5a65 |
| 1 | Betty White — /people/person/places_lived./people/place_lived/location — New York City | /m/025mb_ /people/person/places_lived./people/place_lived/location /m/02_286 | https://en.wikipedia.org/wiki/New_York_City | arm_268fbd2e5a65 |
| 1 | Betty White — /people/person/places_lived./people/place_lived/location — Los Angeles | /m/025mb_ /people/person/places_lived./people/place_lived/location /m/030qb3t | https://en.wikipedia.org/wiki/Los_Angeles | arm_268fbd2e5a65 |
| 1 | Walter Murch — /people/person/places_lived./people/place_lived/location — Los Angeles | /m/02lp3c /people/person/places_lived./people/place_lived/location /m/030qb3t | https://en.wikipedia.org/wiki/Ben_Affleck | arm_268fbd2e5a65 |
| 1 | Steven Blum — /people/person/places_lived./people/place_lived/location — Chicago | /m/044_7j /people/person/places_lived./people/place_lived/location /m/01_d4 | https://en.wikipedia.org/wiki/Chicago | arm_6a06bdebf12f |

### /people/person/place_of_birth

| iter | triple (text) | triple (id) | url | arm_id |
|---:|---|---|---|---|
| 1 | Victor Garber — /people/person/place_of_birth — Saint-Denis, Réunion (web:398acc4e5c6b4d48) | /m/01y665 /people/person/place_of_birth web:398acc4e5c6b4d48 | https://en.wikipedia.org/wiki/Saint-Denis,_R%C3%A9union | arm_9cafad4e317c |
| 1 | Victor Garber — /people/person/place_of_birth — Saint-Étienne (web:4401c1b861048bd3) | /m/01y665 /people/person/place_of_birth web:4401c1b861048bd3 | https://en.wikipedia.org/wiki/Saint-%C3%89tienne | arm_9cafad4e317c |
| 1 | Irwin Winkler — /people/person/place_of_birth — Hollywood | /m/04t38b /people/person/place_of_birth /m/0f2wj | https://en.wikipedia.org/wiki/Hollywood | arm_9cafad4e317c |
| 6 | Michael Crichton — /people/person/place_of_birth — Chicago | /m/056wb /people/person/place_of_birth /m/01_d4 | https://en.wikipedia.org/wiki/Chicago | arm_9cafad4e317c |
| 6 | Manny Coto — /people/person/place_of_birth — Chicago | /m/059j4x /people/person/place_of_birth /m/01_d4 | https://en.wikipedia.org/wiki/Chicago | arm_9cafad4e317c |
| 6 | Ruth Gordon — /people/person/place_of_birth — Chicago | /m/06g4l /people/person/place_of_birth /m/01_d4 | https://en.wikipedia.org/wiki/Chicago | arm_9cafad4e317c |

### /base/biblioness/bibs_location/country

| iter | triple (text) | triple (id) | url | arm_id |
|---:|---|---|---|---|
| 1 | Chicago — /base/biblioness/bibs_location/country — United Kingdom | /m/01_d4 /base/biblioness/bibs_location/country /m/07ssc | https://en.wikipedia.org/wiki/United_Kingdom | arm_6a06bdebf12f |
| 1 | Honolulu — /base/biblioness/bibs_location/country — United Kingdom | /m/02hrh0_ /base/biblioness/bibs_location/country /m/07ssc | https://en.wikipedia.org/wiki/United_Kingdom | arm_6a06bdebf12f |
| 1 | Saint-Denis, Réunion (web:398acc4e5c6b4d48) — /base/biblioness/bibs_location/country — Canada | web:398acc4e5c6b4d48 /base/biblioness/bibs_location/country /m/0d060g | https://en.wikipedia.org/wiki/Canada | arm_9cafad4e317c |
| 1 | Saint-Étienne (web:4401c1b861048bd3) — /base/biblioness/bibs_location/country — Canada | web:4401c1b861048bd3 /base/biblioness/bibs_location/country /m/0d060g | https://en.wikipedia.org/wiki/Canada | arm_9cafad4e317c |
| 6 | Marin County — /base/biblioness/bibs_location/country — Canada | /m/0l2hf /base/biblioness/bibs_location/country /m/0d060g | https://en.wikipedia.org/wiki/Canada | arm_6a06bdebf12f |

## 事例（raw取得トリプル + provenance URL）

raw側は量が多いので、上の主要relationから少数のみ抜粋します。

### /people/person/places_lived./people/place_lived/location

| iter | triple (text) | url |
|---:|---|---|
| 1 | Betty White — /people/person/places_lived./people/place_lived/location — New York City | https://en.wikipedia.org/wiki/New_York_City |
| 1 | Betty White — /people/person/places_lived./people/place_lived/location — Los Angeles | https://en.wikipedia.org/wiki/Los_Angeles |
| 1 | Betty White — /people/person/places_lived./people/place_lived/location — Chicago | https://en.wikipedia.org/wiki/Chicago |
| 1 | Steven Blum — /people/person/places_lived./people/place_lived/location — New York City | https://en.wikipedia.org/wiki/New_York_City |

### /people/person/place_of_birth

| iter | triple (text) | url |
|---:|---|---|
| 1 | John Ottman — /people/person/place_of_birth — Mobile | https://en.wikipedia.org/wiki/Mobile,_Alabama |
| 1 | Victor Garber — /people/person/place_of_birth — Saint-Étienne (web:4401c1b861048bd3) | https://en.wikipedia.org/wiki/Saint-%C3%89tienne |
| 1 | Victor Garber — /people/person/place_of_birth — Saint-Denis, Réunion (web:398acc4e5c6b4d48) | https://en.wikipedia.org/wiki/Saint-Denis,_R%C3%A9union |
| 1 | Victor Garber — /people/person/place_of_birth — Saint-Germain-en-Laye (web:6a4395d9fbacf303) | https://en.wikipedia.org/wiki/Saint-Germain-en-Laye |

### /base/biblioness/bibs_location/country

| iter | triple (text) | url |
|---:|---|---|
| 1 | Chicago — /base/biblioness/bibs_location/country — United States (web:fec8a026aed983a9) | https://en.wikipedia.org/wiki/United_States |
| 1 | New York City — /base/biblioness/bibs_location/country — United States (web:fec8a026aed983a9) | https://en.wikipedia.org/wiki/United_States |
| 1 | Mobile — /base/biblioness/bibs_location/country — United States (web:fec8a026aed983a9) | https://en.wikipedia.org/wiki/United_States |
| 1 | Mobile — /base/biblioness/bibs_location/country — United Kingdom | https://en.wikipedia.org/wiki/United_Kingdom |

### /location/location/contains

| iter | triple (text) | url |
|---:|---|---|
| 1 | Cook County — /location/location/contains — Chicago | https://en.wikipedia.org/wiki/Cook_County,_Illinois |
| 1 | United States (web:fec8a026aed983a9) — /location/location/contains — Chicago | https://en.wikipedia.org/wiki/United_States |
| 1 | United States (web:fec8a026aed983a9) — /location/location/contains — New York City | https://en.wikipedia.org/wiki/United_States |
| 1 | California — /location/location/contains — Los Angeles | https://en.wikipedia.org/wiki/California |

## 補足

- `web:*` のようなエンティティIDはWeb由来の新規エンティティ（候補）です。
- URLは `iter_k/web_provenance.json` に記録されているものを表示しています（多くはWikipedia）。

