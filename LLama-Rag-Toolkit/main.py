import streamlit as st
from utils import initialize_session_state
from ui import render_ui

def main():
    initialize_session_state()
    render_ui()

if __name__ == "__main__":
    main()
