import streamlit as st
import pandas as pd
from sqlalchemy.future import select
from openduck_py.db import connection_string
from openduck_py.models import DBChatRecord

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)
db = Session()

def get_chat_records(session_id):
    stmt = select(DBChatRecord).where(DBChatRecord.session_id == session_id)
    records = db.execute(stmt).scalars().all()
    return records

def display_chat_interface(records):
    for record in records:
        meta_json = record.meta_json or {}
        st.text(f"{record.event_name} {record.timestamp}")
        if "audio_url" in meta_json:
            st.audio(meta_json.get("audio_url"), format="audio/wav", start_time=0)
        elif "text" in meta_json:
            st.text(meta_json.get("text"))



stmt = select(DBChatRecord.session_id).distinct().order_by(DBChatRecord.timestamp.desc())
unique_session_ids = db.execute(stmt).all()
session_id_options = [session_id[0] for session_id in unique_session_ids]  # Unpack the tuples
session_id_input = st.selectbox("Select Session ID to display records:", session_id_options)
if session_id_input:
    records = get_chat_records(session_id_input)
    display_chat_interface(records)
