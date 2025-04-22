#!/usr/bin/env python3
import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Launch the Streamlit app
    sys.argv = ["streamlit", "run", "app/Homepage.py"]
    sys.exit(stcli.main()) 