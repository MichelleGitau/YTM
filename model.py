import streamlit as st
import pandas as pd
import pickle


st.set_page_config(layout="wide", initial_sidebar_state="auto", page_icon="⚡", page_title='YTM'
                                                                                          ' prediction')

load_model = (open('./model.pkl', 'rb'))
model = pickle.load(load_model)

Bondtype = pd.DataFrame(['IFB', 'FXD', 'SDB'])
st.title('YTM Prediction Model')

Tenor = pd.to_numeric(st.text_input('Tenor[in days]', ''))
Loan = pd.to_numeric(st.text_input('Loan Amount', ''))
Coupon = pd.to_numeric(st.text_input('Coupon Rate[in decimals]', ''))
Clean = pd.to_numeric(st.text_input('Clean Price', ''))
Accrued = pd.to_numeric(st.text_input('Accrued Interest', ''))
Dirty = pd.to_numeric(st.text_input('Dirty Price', ''))
CBR = pd.to_numeric(st.text_input('CBR rate', ''))
Inflation = pd.to_numeric(st.text_input('Inflation Rate', ''))
Bond = st.selectbox('Bond', Bondtype)

sel_bond = pd.DataFrame()
sel_bond['Bondtype'] = Bondtype
sel_bond = sel_bond[sel_bond == Bond]
sel_bond = pd.get_dummies(sel_bond)
selected_bond = sel_bond.T

selected_bond['IFB'] = selected_bond[0].astype(int)
selected_bond['FXD'] = selected_bond[1].astype(int)
selected_bond['SDB'] = selected_bond[2].astype(int)
selected_bond = selected_bond.drop([0, 1, 2, 'SDB'], axis=1).reset_index()

selected_bond = selected_bond.drop(selected_bond.columns[0], axis=1)

dt = {'Tenor[in days]': Tenor,
      'Loan Amount': Loan,
      'Coupon Rate[in decimals]': Coupon,
      'Clean Price': Clean,
      'Accrued Interest': Accrued,
      'Dirty Price': Dirty,
      'CBR rate': CBR,
      'Inflation Rate': Inflation}

data1 = pd.DataFrame(pd.Series(dt))
data_frame = pd.DataFrame(data1.T)
data = pd.concat((data_frame, selected_bond), axis=1)


def prediction(dataset):
    df = model.predict(dataset)
    return df


if st.button("Predict"):
    result = prediction(dataset=data)
    st.success(result)
