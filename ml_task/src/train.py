import os
import sys
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from googleapiclient.discovery import build
import pandas as pd

from sentence_transformers import SentenceTransformer, util

from autogluon.tabular import TabularDataset, TabularPredictor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../credentials')))
from utils import *
from config import *

import warnings


_SUFFIX = datetime.today().strftime("%Y%m%d")

"""
API経由でYoutubeデータ情報をインポートしてDataFrameに格納
"""
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
df = YOUTUBE_DATA_API(API_KEY, YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, CHANNEL_ID)[['Title','View Count']] # type: ignore
print(df.info())

output_folder = os.path.join('..', 'YOUTUBE', 'ml_task', 'data', 'datasets')
# 出力ファイル名を設定
output_filename = f'dataset_{_SUFFIX}.csv'
# フルパスを作成
output_path = os.path.join(output_folder, output_filename)

# ファイルが存在しない場合のみ、CSVファイルを出力
if not os.path.exists(output_path):
    df.to_csv(output_path, index=False)
    print(f"CSVファイルが {output_path} に出力されました。")
else:
    print(f"ファイル {output_filename} は既に存在します。上書きされませんでした。")

"""
モデルの学習
"""
model_name = f'STF_{_SUFFIX}'
output_folder = os.path.join('..', 'YOUTUBE', 'ml_task', 'models', 'cos_sim', f'cos_sim_{_SUFFIX}')
output_path = os.path.join(output_folder, model_name)

# 出力先フォルダが存在しない場合に作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  # cos_sim フォルダが存在しない場合は作成

# モデルがすでに存在する場合は読み込み、存在しない場合は新たに学習して保存
if os.path.exists(output_path):
    print(f"モデルが既に存在します。既存のモデルを読み込みます: {model_name}")
    model = SentenceTransformer(output_path)  # 既存のモデルを読み込む
else:
    print(f"モデルが存在しません。新しいモデルを学習して保存します: {model_name}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 新しいモデルを学習
    model.save(output_path)  # モデルを保存
print(f"モデルの処理が完了しました: {output_path}")

"""
タイトルのエンベッティング処理
"""
colums_name = df.columns
titles = df['Title'].to_list()
vals = df['View Count'].values

# タイトルのエンベッティング
title_embeddings = model.encode(titles, convert_to_tensor=True)

# テンソルをNumPy配列に変換
title_embeddings_np = title_embeddings.cpu().numpy()

# 各エンベディングをリストに変換
embeddings_list = [embedding.tolist() for embedding in title_embeddings_np]

# 新しいDataFrameを作成 (Embedding列の各要素を個別の列に変換)
embedding_df = pd.DataFrame(embeddings_list)

# 元のデータフレームにタイトルと閲覧数を結合
df_embeddings = pd.concat([pd.DataFrame({'Title': titles, 'View Count': vals}), embedding_df], axis=1)
file_name_pkl = f'df_pkl_{_SUFFIX}'

# 保存先のパスを設定
output_folder = os.path.join('..', 'YOUTUBE', 'ml_task', 'data', 'datasets', 'df_pkl')
# フルパスを作成
output_path = os.path.join(output_folder, file_name_pkl)

if not os.path.exists(output_path):
    df_embeddings.to_pickle(output_path)
else:
    print(f"ファイル {file_name_pkl} は既に存在します。上書きされませんでした。")


"""
分析用データの分割
y = df_embeddings['View Count']
X = df_embeddings.drop(['Title','View Count'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""

"""
Autogluonの学習
"""

def autoglon_train(mode:str='train'):
    """
    mode = 'test' 軽量での実行/実行時間短縮用
    mode = 'train' 本番実行用
    """
    model_name = f'autogluon_{_SUFFIX}'
    output_folder = os.path.join('..', 'YOUTUBE', 'ml_task', 'models', 'auto_ML', 'autogluon', f'autogluon_{_SUFFIX}')
    output_path = os.path.join(output_folder, model_name)
    ag_test_params =  {
        "presets": "fastest_train",  # 学習を高速化
        "time_limit": 300,  # 5分以内で学習終了
        "num_bag_folds": 0,  # バギングなし
        "num_stack_levels": 0,  # スタッキングなし
        "excluded_model_types": ["KNN", "RF", "XT", "NN_TORCH", "CAT"],  # 除外するモデル
        "ag_args_fit": {"use_gpu": True}  # GPU を使用
    }
    ag_params =  {
        "presets": "best_quality",
        "fit_strategy": "sequential",
        "memory_limit": 2048,
        "ag_args_fit": {"use_gpu": True},
    }
    # フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print('対象のフォルダを作成します')

    # フォルダが空の場合（モデルがない場合）は学習する
    if not os.listdir(output_folder):
        print(f"モデルが存在しません。新しいモデルを学習して保存します: {output_folder}")

        target = 'View Count'
        train_data = df_embeddings.drop('Title', axis=1)  # 入力データの準備
        if mode in ['train', 'test']:
            params = ag_test_params if mode == 'train' else ag_params  # モードに応じてパラメータを選択

            predictor = TabularPredictor(
                label=target,
                problem_type='regression',
                eval_metric='rmse',
                path=output_folder
            ).fit(
                train_data,
                **params  # ここでパラメータを適用
            )

            print(f"モデルの学習が完了しました: {model_name}")
            return predictor
    else:
        print(f"モデルは既に存在しています: {model_name}")

path = 'ml_task/models/auto_ML/autogluon/autogluon_20250207/ds_sub_fit/sub_fit_ho'
predictor = TabularPredictor.load(path)

"""
エラー原因を突き止めるために推論も実装
"""

new_title = '【小学生　スプラ配信】今日こそXパワーを上げる'
new_embedding = model.encode(new_title, convert_to_tensor=True)

new_embedding_df = pd.DataFrame(new_embedding.cpu().numpy().reshape(1, -1), columns=[f"{i}" for i in range(new_embedding.shape[0])])
new_data = TabularDataset(new_embedding_df)

pred = predictor.predict(new_data) # type: ignore
model_names = predictor.model_names() # type: ignore
pred_multi = predictor.predict_multi(new_data,as_pandas=True)
leaderboard = predictor.leaderboard()


print(f'predictions:{pred}')
print(f'model_names:{model_names}')
print(f'model_multi:{pred_multi}')