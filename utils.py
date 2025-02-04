import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import gspread
from google.oauth2.service_account import Credentials

from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from googleapiclient.discovery import build
import pandas as pd
import jpholiday

from credentials.config import client


def YOUTUBE_DATA_API(API_KEY:str, YOUTUBE_API_SERVICE_NAME:str, YOUTUBE_API_VERSION:str, CHANNEL_ID:str):
    youtube = build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        developerKey=API_KEY
    )

    # データを格納するリスト
    video_data = []

    # Step 1: チャンネルのアップロードプレイリストIDを取得
    response = youtube.channels().list(part='contentDetails', id=CHANNEL_ID).execute()
    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # Step 2: プレイリストから動画情報を取得
    next_page_token = None
    while True:
        playlist_response = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_playlist_id,
            maxResults=50,  # 1回のリクエストで最大50件取得可能
            pageToken=next_page_token
        ).execute()

        # 動画情報をリストに追加
        for video in playlist_response['items']:
            video_id = video['snippet']['resourceId']['videoId']
            title = video['snippet']['title']
            published_at = video['snippet']['publishedAt']
            description = video['snippet']['description']
            video_data.append({'Video ID': video_id, 'Title': title,'Description':description,'published_date':published_at})

        # 次のページがある場合は取得
        next_page_token = playlist_response.get('nextPageToken')
        if not next_page_token:
            break

    # Step 3: データをDataFrameに変換
    df = pd.DataFrame(video_data).sort_values('published_date', ascending=True)
    return df


def YOUTUBE_ANALYTICS_API(SCOPES:str, end_date:str, start_date:str, dimensions, metrics, filters=None, sort=None):
    """
    SCOPES:OAuth2認証用のスコープ
    end_date:分析用終了日
    start_date:分析用開始日
    VIDEO_ID:ビデオID
    dimensions:
    metrics:
    """
    # 認証フローを設定
    flow = InstalledAppFlow.from_client_secrets_file(
        client,  # クライアントIDファイルのパス
        SCOPES
    )

    # ローカルサーバーで認証を実行
    credentials = flow.run_local_server(port=0)

    # APIクライアントを作成
    youtube_analytics = build('youtubeAnalytics', 'v2', credentials=credentials)
    youtube = build('youtube', 'v3', credentials=credentials)

    # 認証後の確認メッセージ
    print("Authentication successful!")

    dimensions = dimensions
    metrics = metrics

    # YouTube Analytics APIでデータを取得
    response = youtube_analytics.reports().query(
        ids='channel==MINE',
        #ids=f'channel=={channel_id}',
        startDate=start_date,
        endDate=end_date,
        metrics=metrics,
        dimensions=dimensions,
        sort=sort,
        filters = filters
    ).execute()

    #取得データをDataFrameに格納
    df = pd.DataFrame(response['rows'], columns=[col['name'] for col in response['columnHeaders']])
    return df

def get_previous(df, target=None):
    """
    前日との差分取得関数
    """
    if target is None:
        return df
    else:
        df[f"{target}_previous_val"] = df[target].shift(1).fillna(0)
        df[f"{target}_diff"] = df[target] - df[f"{target}_previous_val"]
        df.drop(columns=[f"{target}_previous_val"], inplace=True)
        return df

def get_rolling_week_sum(df,target=None):
    if target is None:
        return df
    else:
        df[f'{target}_week_rolling'] = df[target].rolling(window=7).sum()
        df[f'{target}_week_rolling'].fillna(0, inplace=True)
        return df

def get_dayname(df, target=None):
    """
    曜日に対応する番号/曜日(EN)/曜日(JP)列の付与
    """
    # 英語 → 日本語 の対応辞書
    day_translation = {
        "Monday": "月",
        "Tuesday": "火",
        "Wednesday": "水",
        "Thursday": "木",
        "Friday": "金",
        "Saturday": "土",
        "Sunday": "日"
    }
    df[target] = pd.to_datetime(df[target], errors="coerce")
    df["day_number"] = df[target].dt.weekday
    df["day_name"] = df[target].dt.day_name()
    df["day_name_jp"] = df["day_name"].map(day_translation) # 日本語に変換
    return df


def get_holiday(df, target=None):
    # 日付を変換して、祝日名を取得
    df[target] = pd.to_datetime(df[target], errors="coerce")
    df["holiday_name"] = df[target].apply(jpholiday.is_holiday_name)
    # 祝日かどうかを判定し、holiday_statusカラムに結果を格納
    # 祝日 または 土日 の場合 "holiday" にする
    df["holiday_status"] = np.where(
    df["holiday_name"].notna() | df[target].dt.weekday.isin([5, 6]), "holiday", "")
    return df


# Google Sheets API認証
def authenticate_google_sheets(credentials_file):
    # サービスアカウントの認証情報をロード
    creds = Credentials.from_service_account_file(
        credentials_file, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return creds


def write_dataframe_to_sheet(df, spreadsheet_id, sheet_name, credentials_file):
    # 認証
    creds = authenticate_google_sheets(credentials_file)

    # Google Sheets APIのクライアントを作成
    client = gspread.authorize(creds)

    # スプレッドシートを開く
    try:
        sheet = client.open_by_key(spreadsheet_id)
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Spreadsheet with ID {spreadsheet_id} not found.")
        return

    # 既存のシートにアクセス（シートが存在しない場合は作成）
    try:
        worksheet = sheet.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=sheet_name, rows="100", cols="20")

    # 日付型の列を文字列に変換
    datetime_columns = list(df.select_dtypes(include=['datetime64[ns]']).columns)
    if datetime_columns:
        df[datetime_columns] = df[datetime_columns].astype(str)

    # datetime_columns を最初に持ってきて並べ替える
    sorted_columns = datetime_columns + [col for col in df.columns if col not in datetime_columns]
    df = df[sorted_columns]

    # DataFrame をリスト形式に変換してシートに書き込む
    data = [df.columns.values.tolist()] + df.values.tolist()

    # シートにデータを書き込む
    worksheet.update('A1', data)