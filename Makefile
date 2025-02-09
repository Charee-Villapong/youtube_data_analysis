#cd youtube
#trainファイル実行後にpredictionファイルを実行 -- make run
run:
	python ml_task/src/train.py
	python ml_task/src/prediction.py
train:
	python ml_task/src/train.py
pred:
	python ml_task/src/prediction.py
bi:
	python bi_project/youtube_api.py
all:
	python bi_project/youtube_api.py
	python ml_task/src/train.py
	python ml_task/src/prediction.py

web:
	streamlit run /Users/yoshitakanishikawa/Downloads/youtube/app/src/youtube_predictor.py