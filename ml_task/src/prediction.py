import os
import sys

from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timedelta
import numpy as np
import torch
import pandas as pd
from autogluon.text import TextPredictor
import os

import warnings
warnings.filterwarnings('ignore')

def autogluon_pred(new_title:str,_SUFFIX:str):
    """
    autoML : Autogluonの推論実施
    """
    path = f'ml_task/models/auto_ML/autogluon/autogluon_{_SUFFIX}'
    predictor = TextPredictor.load(path)

    pred = predictor.predict({'Title': [new_title]}, as_pandas=False)
    print(pred)
    return pred

def get_cos_sim(new_title:str,_SUFFIX:str):
    """
    分散表現のコサイン類似度を算出
    """
    model_name = f'STF_{_SUFFIX}'
    input_folder = os.path.join('..', 'YOUTUBE', 'ml_task', 'models', 'cos_sim', f'cos_sim_{_SUFFIX}')
    input_path = os.path.join(input_folder, model_name)
    # モデルの読み込み
    model = SentenceTransformer(input_path)
    # タイトルのエンベッティング
    new_embedding = model.encode(new_title, convert_to_tensor=True)

    # 保存先のパスを設定
    input_folder = os.path.join('..', 'YOUTUBE', 'ml_task', 'data', 'datasets', 'df_pkl')
    file_name_pkl = f'df_pkl_{_SUFFIX}'
    # フルパスを作成
    input_path = os.path.join(input_folder, file_name_pkl)
    df_embeddings = pd.read_pickle(input_path)
    vals = df_embeddings['View Count'].values
    title_embeddings = torch.tensor(df_embeddings.drop(['Title', 'View Count'], axis=1).to_numpy(), dtype=torch.float32)

    # コサイン類似度の計算
    similarities = util.pytorch_cos_sim(new_embedding.cpu(), title_embeddings.cpu())[0]

    # 最も類似度の高いインデックスを取得
    best_match_idx = similarities.argmax().item()
    top_k = 3
    top_match_indices = similarities.topk(top_k).indices.tolist()

    # 類似度の高い動画の閲覧数を予測値とする
    predicted_views = vals[best_match_idx] # type: ignore

    print(f"予測閲覧数(top): {predicted_views}")
    predicted_views_mean = sum(vals[i] for i in top_match_indices) / len(top_match_indices)
    print(f"予測閲覧数(Top3の平均): {predicted_views_mean}")
    return predicted_views, predicted_views_mean

"""
main execution
"""
if __name__ == "__main__":
    new_title = '【小学生　スプラ配信】今日こそXパワーを上げる'
    _SUFFIX = datetime.today().strftime("%Y%m%d")
    get_cos_sim(new_title, _SUFFIX)
    autogluon_pred(new_title, _SUFFIX)
