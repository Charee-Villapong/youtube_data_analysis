#cd youtube
#trainファイル実行後にpredictionファイルを実行 -- make run
run:
	python ml_task/src/train.py
	python ml_task/src/prediction.py
training:
	python ml_task/src/train.py
prediction:
	python ml_task/src/prediction.py
bi:
	python bi_project/youtube_api.py
all:
	python bi_project/youtube_api.py
	python ml_task/src/train.py
	python ml_task/src/prediction.py
