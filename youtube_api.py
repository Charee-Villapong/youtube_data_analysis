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

from utils import *
from credentials.config import *

import warnings

# 警告を無視する
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    """
    YOUTUBE_DATA_APIの実行
    """
    YOUTUBE_API_SERVICE_NAME = 'youtube'
    YOUTUBE_API_VERSION = 'v3'
    df_data = YOUTUBE_DATA_API(API_KEY, YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, CHANNEL_ID) # type: ignore

    """
    YOUTUBE_ANALYTICS_APIの実行
    """
    #start_date = (datetime.today() - timedelta(days=90)).date().isoformat()
    start_date = "2024-12-27"
    end_date = datetime.today().date().isoformat()
    dimensions = "day"
    metrics = (
        "views,likes,dislikes,comments,shares,subscribersGained,subscribersLost,estimatedMinutesWatched,averageViewPercentage,averageViewDuration"
    )

    targets = ["views","averageViewDuration", "estimatedMinutesWatched"]
    # OAuth2認証用のスコープを設定
    SCOPES = ["https://www.googleapis.com/auth/yt-analytics.readonly"]

    # YOUTUBE_ANALYTICS_API関数の実行
    df = YOUTUBE_ANALYTICS_API(SCOPES, end_date, start_date, dimensions, metrics,sort=dimensions) # type: ignore

    df = get_dayname(df,"day") #曜日etcの取得
    df = get_holiday(df,"day") #祝日/休日の取得

    for target in targets:
        df = get_rolling_week_sum(df, target=target) #一週間のRolling SUM取得
        df = get_previous(df, target=target) #前日との差分取得

    """
    DataframeをSpreadsheetへ書き込み
    """

    # Google SheetsのスプレッドシートID（URLの中に含まれるID部分）
    sheet_name_1 = 'data'  # 書き込みたいシート名
    sheet_name_2 = 'data_2'

    # DataFrameを書き込む
    write_dataframe_to_sheet(df, spreadsheet_id, sheet_name_1, credentials_file)
    write_dataframe_to_sheet(df_data, spreadsheet_id, sheet_name_2, credentials_file)

    print("DataFrame has been written to the Google Sheets!")
