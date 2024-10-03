import streamlit as st
from src.ui.main_ui import render_ui
from config.settings import initialize_settings

def main():
    initialize_settings()
    render_ui()

if __name__ == "__main__":
    main()
