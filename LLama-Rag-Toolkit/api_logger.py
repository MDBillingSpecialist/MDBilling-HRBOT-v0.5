import streamlit as st

def add_api_call(request, response):
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = []
    call_id = len(st.session_state.api_calls)
    st.session_state.api_calls.append({
        'id': call_id,
        'request': request,
        'response': response
    })