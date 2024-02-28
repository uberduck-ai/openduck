import streamlit as st
import pandas as pd
from typing import Dict
from sqlalchemy.future import select
from openduck_py.db import connection_string
from openduck_py.models import DBChatRecord

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)
db = Session()

def get_chat_records(session_id):
    stmt = select(DBChatRecord).where(DBChatRecord.session_id == session_id).order_by(DBChatRecord.timestamp)
    records = db.execute(stmt).scalars().all()
    return records

def display_chat_interface(records, show_events: Dict[str, bool]):
    for record in records:
        if not show_events[record.event_name]:
            continue
        role = "user"
        if record.event_name in ["generated_completion", "generated_tts", "normalized_text"]:
            role = "assistant"
        meta_json = record.meta_json or {}
        avatar = None
        if record.event_name in ["started_session", "detected_start_of_speech", "detected_end_of_speech", "normalized_text", "interrupted_response"]:
            avatar = "⚙️" 
        with st.chat_message(role, avatar=avatar):
            st.write(f"{record.event_name} {record.timestamp}")
            if "audio_url" in meta_json:
                st.audio(meta_json.get("audio_url"), format="audio/wav", start_time=0)
            elif "text" in meta_json:
                st.markdown("##### "+meta_json.get("text"))


stmt = select(DBChatRecord.session_id).distinct().order_by(DBChatRecord.timestamp.desc())
unique_session_ids = db.execute(stmt).all()
session_id_options = [session_id[0] for session_id in unique_session_ids]
session_id_input = st.selectbox("Select Session ID to display records:", session_id_options)

unique_event_names = db.execute(select(DBChatRecord.event_name).distinct()).all()
event_name_options = [event_name[0] for event_name in unique_event_names]
st.sidebar.write("Select events to show:")
show_events = {}
for event in event_name_options:
    value = False
    if event in ["generated_tts", "generated_completion", "transcribed_audio", "started_response"]:
        value = True
    show_events[event] = st.sidebar.checkbox(event, value=value)

if session_id_input:
    records = get_chat_records(session_id_input)
    display_chat_interface(records, show_events)
