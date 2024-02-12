from uuid import uuid4
from datetime import datetime
from typing import Optional, List
import math
import tempfile
import os
import random

from sqlalchemy import (
    BigInteger,
    cast,
    Column,
    ForeignKey,
    Text,
    select,
    func,
    Boolean,
    update,
    DateTime,
)
from sqlalchemy_searchable import search
from sqlalchemy_utils.types import TSVectorType
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql.expression import nullslast
from sqlalchemy.orm import relationship, Session
from torch.utils.data import DataLoader
from tqdm import tqdm

from uberduck_ml_dev.data.data import Data
from uberduck_ml_dev.data.collate import Collate

from openduck_py.db import Base
from openduck_py.models import DBDataSource, DBDatasetAudioFile

# from openduck_py.utils.s3 import (
#     upload_file,
#     AUDIO_FILE_BUCKET_CLEANED,
#     AUDIO_FILE_BUCKET,
# )
# from openduck_py.db.utils import all_scalars


class DBDataset(Base):
    __tablename__ = "dataset"
    id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    uuid = Column(Text, nullable=False, default=uuid4, unique=True)
    clean_path = Column(Text)
    upload_path = Column(Text)
    team_id = Column(BigInteger, ForeignKey("team.id", ondelete="SET NULL"), index=True)
    meta_json = Column(MutableDict.as_mutable(JSONB))
    is_private = Column(Boolean, default=True)
    data_source_id = Column(
        BigInteger, ForeignKey("data_source.id", ondelete="SET NULL"), index=True
    )
    voiced_entity_id = Column(
        BigInteger, ForeignKey("voiced_entity.id", ondelete="SET NULL"), index=True
    )
    marketplace_task = relationship(
        "DBMarketplaceTask", backref="datasets"
    )  # NOTE (Sam): allows one dataset per marketplace task, but multiple marketplace tasks per dataset. We might want a task_data table for many to many functionality.
    data_source = relationship("DBDataSource", backref="datasets")
    voiced_entity = relationship("DBEntity", backref="datasets")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=True)
    is_internal = Column(Boolean, default=False, nullable=True)
    is_customer = Column(Boolean, default=False, server_default="false", nullable=False)

    search_vector = Column(TSVectorType("name"))

    def __str__(self):
        return f"Name: {self.name}, ID: {self.id}"

    def url(self, clean=False):
        assert not clean, "url on iplmeneted for dirty"
        return f"https://uberduck-datasets-dirty.s3.amazonaws.com/{self.upload_path}"

    @classmethod
    def create(
        cls,
        name,
        upload_path: Optional[str] = None,
        team_id: Optional[int] = None,
        is_private: bool = True,
        meta_json: Optional[dict] = None,
        data_source_id: Optional[int] = None,
        voiced_entity_id: Optional[int] = None,
        is_internal: bool = False,
        is_customer: bool = False,
    ):
        query = (
            datasets.insert()
            .values(
                name=name,
                is_private=is_private,
                meta_json=meta_json,
                upload_path=upload_path,
                team_id=team_id,
                uuid=str(uuid4()),
                data_source_id=data_source_id,
                voiced_entity_id=voiced_entity_id,
                is_internal=is_internal,
                is_customer=is_customer,
            )
            .returning(*datasets.c)
        )
        return query

    @classmethod
    def get_by_id(cls, id: int):
        return select(cls).where(cls.id == id)

    @classmethod
    def get_by_uuid(cls, uuid: str, include_data_source: bool = True):
        if include_data_source:
            return (
                select(cls, DBDataSource.uuid.label("data_source_uuid"))
                .join(DBDataSource, cls.data_source_id == DBDataSource.id, isouter=True)
                .where(cls.uuid == uuid)
            )
        else:
            return select(cls).where(cls.uuid == uuid)

    @classmethod
    def get_by_data_source_id(cls, data_source_id: str):
        return (
            select(cls, func.count(DBDatasetAudioFile.id).label("audio_file_count"))
            .join(
                DBDatasetAudioFile,
                cls.id == DBDatasetAudioFile.dataset_id,
                isouter=True,
            )
            .group_by(cls.id)
            .where(cls.data_source_id == data_source_id)
        )

    @classmethod
    def update_voiced_entity_by_id(cls, id: int, voiced_entity_id: int):
        return (
            update(cls)
            .where(cls.id == id)
            .values(voiced_entity_id=voiced_entity_id)
            .returning(cls)
        )

    @classmethod
    def update_by_uuid(cls, uuid: str, **values):
        allowed_keys = [
            "voiced_entity_id",
            "name",
        ]
        updates = {}
        for k in allowed_keys:
            if k in values:
                updates[k] = values[k]
        return update(cls).where(cls.uuid == uuid).values(**updates).returning(cls)

    @classmethod
    def update_voiced_entity_by_uuid(cls, uuid: str, voiced_entity_id: int):
        return (
            update(cls)
            .where(cls.uuid == uuid)
            .values(voiced_entity_id=voiced_entity_id)
            .returning(cls)
        )

    # NOTE (Sam): this will recompute the pitches even if they already exist.
    # TODO (Sam): operate on resampled data (can save resampled in meta_json)
    @classmethod
    def compute_pitches(
        cls,
        dataset_id,
        session: Session,
        download_size=1024,
        batch_size=16,
        cleaned=False,
        device="cpu",
        method="parselmouth",
    ):
        f0_cache = tempfile.TemporaryDirectory()
        f0_cache_path = f0_cache.name
        bucket = AUDIO_FILE_BUCKET_CLEANED if cleaned else AUDIO_FILE_BUCKET
        db_dataset_audio_files = (
            session.execute(
                DBDatasetAudioFile.get_by_dataset_id(dataset_id, is_cleaned=cleaned)
            )
            .scalars()
            .all()
        )
        nfiles = len(db_dataset_audio_files)
        niter = math.ceil(nfiles / download_size)
        for j in tqdm(range(niter)):
            print("Downloading new data subset")
            file_start_idx = j * download_size
            file_stop_idx = min(nfiles, (j + 1) * download_size)
            paths = []
            db_dataset_audio_file_uuids = []
            for i, db_dataset_audio_file in tqdm(
                enumerate(db_dataset_audio_files[file_start_idx:file_stop_idx])
            ):
                path = db_dataset_audio_file.download(resample=True, normalize=True)
                paths.append(path)
                db_dataset_audio_file_uuids.append(db_dataset_audio_file.uuid)

            data = Data(
                audiopaths=paths,
                return_f0s=True,
                f0_cache_path=f0_cache_path,
                return_texts=False,
                return_mels=False,
                return_speaker_ids=False,
                sampling_rate=22050,
            )

            collate_fn = Collate(
                cudnn_enabled=device == "cuda",
            )
            data_loader = DataLoader(
                data,
                batch_size=batch_size,
                collate_fn=collate_fn,
            )

            print("Computing pitches")
            for batch in data_loader:
                pass  # computes pitches in loader.

            print("Writing to db")
            for i, db_dataset_audio_file in tqdm(
                enumerate(db_dataset_audio_files[file_start_idx:file_stop_idx])
            ):
                pitch_identification_tag = (
                    "_f0_sr{}_fl{}_hl{}_f0min{}_f0max{}_log{}.pt".format(
                        data.sampling_rate,
                        data.filter_length,
                        data.hop_length,
                        data.f0_min,
                        data.f0_max,
                        data.use_log_f0,
                    )
                )
                f0_save_path = (
                    f"{db_dataset_audio_file.uuid}/{pitch_identification_tag}"
                )
                f0_rel_path = (
                    # NOTE (Sam): must be same as https://github.com/uberduck-ai/uberduck-ml-dev/blob/master/uberduck_ml_dev/data/data.py#L166
                    "_".join(paths[i].split("/")).split(".wav")[0]
                    + pitch_identification_tag
                )

                upload_file(
                    f0_save_path, bucket, path=os.path.join(f0_cache_path, f0_rel_path)
                )
                if db_dataset_audio_file.meta_json is None:
                    db_dataset_audio_file.meta_json = {}
                db_dataset_audio_file.meta_json["pitch_file_path"] = f0_save_path

                session.add(db_dataset_audio_file)

            session.commit()

    # TODO (Sam): parallelize by moving to the dataloader.
    @classmethod
    def modify(
        cls,
        dataset_id,
        session: Session,
        is_cleaned: bool = False,
        resample: bool = True,
        normalize: bool = True,
        skip_existing: bool = True,
    ):
        db_dataset_audio_files = (
            session.execute(DBDatasetAudioFile.get_by_dataset_id(dataset_id))
            .scalars()
            .all()
        )

        if not db_dataset_audio_files:
            raise Exception("Training audio files not found.")
        bucket = AUDIO_FILE_BUCKET_CLEANED if is_cleaned else AUDIO_FILE_BUCKET
        for idx, db_dataset_audio_file in enumerate(tqdm(db_dataset_audio_files)):
            norm_string = "normalized" if normalize else "unnormalized"
            meta_json_key = f"22k_{norm_string}_path"
            if db_dataset_audio_file.meta_json is None:
                db_dataset_audio_file.meta_json = {}
            if skip_existing and meta_json_key in db_dataset_audio_file.meta_json:
                continue

            try:
                local_22k_normalized_path = db_dataset_audio_file.download(
                    resample=resample, normalize=normalize
                )
            except Exception:
                print(
                    f"Failed to download file with ID: {db_dataset_audio_file.id}, UUID: {db_dataset_audio_file.uuid}"
                )
                raise
            identification_tag = (
                ("resampled" if resample else "unresampled")
                + "_"
                + norm_string
                + ".wav"
            )
            bucket_path = f"{db_dataset_audio_file.uuid}/{identification_tag}"
            upload_file(bucket_path, bucket, path=local_22k_normalized_path)
            db_dataset_audio_file.meta_json[meta_json_key] = bucket_path
            session.add(db_dataset_audio_file)
            if idx % 50 == 0:
                session.commit()

        session.commit()

    @classmethod
    def download(
        cls,
        session: Session,
        dataset_id: int,
        is_cleaned: bool,
        sample_fraction: float = 1.0,
        speaker_id: int = 0,
        resample: bool = True,
        normalize: bool = True,
    ):
        paths, speaker_ids, transcripts = [], [], []
        db_dataset_audio_files = all_scalars(
            session,
            DBDatasetAudioFile.get_by_dataset_id(
                dataset_id,
                is_cleaned=is_cleaned,
            ),
        )
        random.seed(1234)
        n_samples = int(sample_fraction * len(db_dataset_audio_files))
        db_dataset_audio_files = random.sample(db_dataset_audio_files, n_samples)
        if not db_dataset_audio_files:
            raise Exception("Training audio files not found.")
        for db_dataset_audio_file in tqdm(db_dataset_audio_files):
            # NOTE (Sam): currently the data is not being kept normalized in s3.
            path = db_dataset_audio_file.download(
                resample=resample, normalize=normalize
            )
            paths.append(path)
            speaker_ids.append(speaker_id)
            if db_dataset_audio_file.transcript is not None:
                transcripts.append(
                    db_dataset_audio_file.transcript.replace('"', "").strip()
                )
            else:
                transcripts.append("")
        return paths, speaker_ids, transcripts


datasets = DBDataset.__table__
