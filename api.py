'''
api это просто микросервис, в котором мы можем получить вероятность дефолта определенной сделки (если она есть в файле dataset.csv)
Методом get мы получаем id сделки. С помощью функции print_info отправляем вероятность дефолта.
'''
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from loguru import logger
from joblib import load

app = FastAPI()
df = pd.read_csv('dataset.csv')

class GetProba(BaseModel):
    probability: str
    class Config:
        orm_mode = True

@app.get('/get_info/{id}', response_model=GetProba)
def print_info(id:int):
    result = df[df['Deal_id']==id]
    result = result.drop('Default', axis=1, errors='ignore')
    cols = ['Secret_info_1', 'Secret_info_2', 'Secret_info_3']
    for col in cols:
        temp_df = result.copy()
        temp_df[col] = temp_df[col].fillna('NaN')
    result['Secret_info_1'] = result['Secret_info_1'].fillna(0).astype('category')
    result['Secret_info_2'] = result['Secret_info_2'].fillna('NaN').astype('category')
    result['Secret_info_3'] = result['Secret_info_3'].fillna('NaN').astype('category')
    result = result.drop('Deal_id', axis=1)
    result['year'] = result['Deal_date'].astype('datetime64[ns]').dt.year
    result['month'] = result['Deal_date'].astype('datetime64[ns]').dt.month
    result['day'] = result['Deal_date'].astype('datetime64[ns]').dt.day
    result['days_from_1st_deal'] = result['Deal_date'].astype('datetime64[ns]') - result['First_deal_date'].astype('datetime64[ns]')
    result['days_from_1st_deal'] = result['days_from_1st_deal'].dt.days
    result = result.drop('First_deal_date', axis=1)
    result['days_after_default'] = result['Deal_date'].astype('datetime64[ns]') - result['First_default_date'].astype('datetime64[ns]')
    result['days_after_default'] = result['days_after_default'].dt.days
    result = result.drop(['First_default_date'], axis=1)
    result['days_after_default'] = result['days_after_default'].fillna(0)
    result['Region'] = result['Region'].fillna(result['Region'].value_counts().index[0])
    result = result.drop('Hashed_deal_detail_6', axis=1)
    result['Deal_date'] = result['Deal_date'].astype('datetime64[ns]')
    result = result.sort_values(by='Deal_date')
    result = result.drop('Deal_date', axis=1)
    model = load('model.joblib')
    result['preds'] = model.predict_proba(result)[:, 1]

    logger.info(result['preds'].item())
    return {"probability":f"Вероятность дефолта {round(result['preds'].item() * 100 ,2)} %"}

