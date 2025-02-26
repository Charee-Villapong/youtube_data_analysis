import streamlit as st
import os
import sys
from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timedelta
import numpy as np
import torch
import pandas as pd
from autogluon.text import TextPredictor
import warnings

#streamlit run /Users/yoshitakanishikawa/Downloads/youtube/app/src/youtube_predictor.py

warnings.filterwarnings('ignore')

def autogluon_pred(new_title:str,_SUFFIX:str) -> float:
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

def main():
    st.title("YouTube View予測アプリ")
    st.text('このアプリはChatGPTの技術的基盤であるTransformerを採用しているよ')
    st.text('君が考えたタイトルがどのくらいの視聴数を取れるか高性能な予測をしてあげる')

    # ユーザー入力
    new_title = st.text_input("動画のタイトルを入力してください：")

    if st.button("予測する"):
        if new_title:
            _SUFFIX = datetime.today().strftime("%Y%m%d")

            # コサイン類似度による予測
            predicted_views, predicted_views_mean = get_cos_sim(new_title, _SUFFIX)

            # AutoGluonによる予測
            autogluon_result = autogluon_pred(new_title, _SUFFIX)

            # 結果の表示
            st.subheader("予測結果")
            st.write(f"AutoGluonというAutoMLによる予測: {autogluon_result} 回")
            st.write(f"コサイン類似度というアルゴリズムから算出(top1)）: {predicted_views} 回")
            st.write(f"コサイン類似度というアルゴリズムから算出（Top3の平均値）: {predicted_views_mean} 回")
            result = (autogluon_result + predicted_views + predicted_views_mean) / 3

            st.write(f"三つのアルゴリズムの平均値: {result} 回")

        else:
            st.warning("タイトルを入力してください。")

if __name__ == "__main__":
    main()