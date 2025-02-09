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

from autogluon.text import TextPredictor 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../credentials')))
from utils import *
from config import *

import warnings
warnings.filterwarnings('ignore')

_SUFFIX = datetime.today().strftime("%Y%m%d")

def youtube_api(API_KEY, CHANNEL_ID):
    """
    API経由でYoutubeデータ情報をインポートしてDataFrameに格納
    """
    YOUTUBE_API_SERVICE_NAME = 'youtube'
    YOUTUBE_API_VERSION = 'v3'
    df = YOUTUBE_DATA_API(API_KEY, YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, CHANNEL_ID)[['Title','View Count']] # type: ignore

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
        return df


def autogluon_train(df, mode:str="train"):
    """
    Autogluonの学習
    """
    model_name = f'autogluon_{_SUFFIX}'
    output_folder = os.path.join('..', 'YOUTUBE', 'ml_task', 'models', 'auto_ML', 'autogluon', f'autogluon_{_SUFFIX}')
    output_path = os.path.join(output_folder, model_name)
    target = 'View Count'
    # フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print('対象のフォルダを作成します')

    # フォルダが空の場合（モデルがない場合）は学習する
    if not os.listdir(output_folder):
        print(f"モデルが存在しません。新しいモデルを学習して保存します: {output_folder}")
    if mode not in {'train', 'test', 'staging'}:
         raise ValueError("引数modeにはtrainかtestを必ず入力してください")
    if mode == 'train':
            predictor = TextPredictor(problem_type='regression',label=target, eval_metric='root_mean_squared_error',verbosity=1)
            predictor.fit(df
                        ,save_path=output_folder
                        ,presets='best_quality'
                        ,num_gpus=2
                        ,holdout_frac=0.2
                        )
    if mode == 'staging':
            predictor = TextPredictor(problem_type='regression',label=target, eval_metric='root_mean_squared_error',verbosity=1)
            predictor.fit(df
                        ,save_path=output_folder
                        ,presets='medium_quality_faster_train'
                        ,num_gpus=2
                        ,holdout_frac=0.2
                        )
    if mode == 'test':
            predictor = TextPredictor(problem_type='regression',label=target, eval_metric='root_mean_squared_error',verbosity=1)
            predictor.fit(df
                        ,save_path=output_folder
                        ,presets='lower_quality_fast_train'
                        ,num_gpus=2
                        ,holdout_frac=0.2
                        )
    print(f"モデルの学習プロセスが完了しました: {model_name}")


"""
Transformerを用いたEmbedding
"""
def st_train(df,_SUFFIX):
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

    #タイトルのエンベッティング処理

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

if __name__ == "__main__":
    df = youtube_api(API_KEY, CHANNEL_ID)
    autogluon_train(df, mode="staging")
    st_train(df, _SUFFIX)