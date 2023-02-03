#!/bin/bash
nginx -t &&
service nginx start &&
streamlit run index-app.py --theme.base "dark"