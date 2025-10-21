🎥 YouTube データ分析（YouTube Data Analysis）

このリポジトリは、YouTube Analytics API と YouTube Data API を活用し、
YouTubeチャンネルおよび動画データを分析するためのツールを提供します。

主な目的は、チャンネルのパフォーマンスや動画統計、エンゲージメント指標を可視化し、
コンテンツクリエイターがデータに基づいた意思決定を行えるように支援することです。

⸻

🧠 開発の背景

実はこのプロジェクトは、8歳の子どもが運営しているYouTubeチャンネルから生まれました。
「どんなタイトルをつけたら見てもらえるの？」という素朴な疑問をきっかけに、
データサイエンスの力でその答えを探るために開発を始めました。

YouTube APIから取得したタイトル・視聴回数を解析し、
タイトル表現と視聴パフォーマンスの関係を学習・可視化することで、
「タイトルの付け方」や「効果的なワード選定」を定量的に理解できる仕組みを構築しています。

⸻

⚙️ 主な機能
	•	YouTube Data API / Analytics API を用いた動画データ取得
	•	タイトル・説明文・タグなどの自然言語処理（SentenceTransformer + Word2Vec）
	•	回帰モデル（AutoGluon）による視聴数予測
	•	Looker Studioによる可視化ダッシュボード
	•	タイトル類似度に基づく改善提案機能

⸻

🚀 使用技術
	•	Language: Python
	•	Libraries: pandas, matplotlib, SentenceTransformers, AutoGluon
	•	Visualization: Looker Studio
	•	APIs: YouTube Data API, YouTube Analytics API

⸻

🌱 成果
	•	チャンネル登録者数：8名 → 74名（2025年3月時点）
	•	ライブ視聴平均：10回 → 30回（最大130回） に成長
