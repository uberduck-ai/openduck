import os
from uuid import uuid4
from datetime import datetime
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
from scipy.io.wavfile import read, write
from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Text,
    select,
    update,
    Boolean,
    DateTime,
    and_,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship

# from openduck_py.settings import MAX_WAV_VALUE
# from openduck_py.utils.s3 import (
#     AUDIO_FILE_BUCKET_CLEANED,
#     AUDIO_FILE_BUCKET,
#     download_file,
# )
from openduck_py.db import Base


class DBDatasetAudioFile(Base):
    """An audio file associated with a dataset.

    meta_json can contain the following keys:
        "22k_normalized_path": If set, the S3 path to the file, sampled at 22050 Hz and max value normalized to 32768 / 2.
        "22k_unnormalized_path": If set, the S3 path to the file, sampled at 22050 Hz.
        "coqui_resnet_512_emb": If set, the embedding of the audio file, computed with the Coqui pretrained audio embedding.
        "f0": If set, the fundamental frequency of the audio file computed with
            librosa.pyin(y, fmin=80, fmax=640, hop_length=256, frame_length=1024, win_length=512)
        "hubert_16k": If set, the embedding of the audio file, computed with the Hubert pretrained audio embedding. Computed on 16kHz audio. Dimensons (256, T).
        "metrics": Assorted audio metrics.
        "old_files": List of paths in S3 to previous versions of this audio file.
    """

    __tablename__ = "dataset_audio_file"
    id = Column(BigInteger, primary_key=True)
    uuid = Column(Text, nullable=False, default=uuid4, unique=True)
    user_id = Column(BigInteger, ForeignKey("user.id", ondelete="SET NULL"), index=True)
    dataset_id = Column(
        BigInteger,
        ForeignKey("dataset.id", ondelete="SET NULL"),
        index=True,
    )
    speaker_display_name = Column(Text)
    transcript = Column(Text)
    upload_path = Column(Text, nullable=False, index=True)
    meta_json = Column(MutableDict.as_mutable(JSONB))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_cleaned = Column(Boolean, nullable=False, index=True)
    voiced_entity_id = Column(
        BigInteger, ForeignKey("voiced_entity.id", ondelete="SET NULL"), index=True
    )
    deleted_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(upload_path, is_cleaned, name="upload_path_is_cleaned_uc"),
    )
    dataset = relationship("DBDataset", backref="dataset_audio_files")
    voiced_entity = relationship("DBEntity", backref="dataset_audio_files")

    def __str__(self):
        return f"uuid: {self.uuid}, speaker: {self.speaker_display_name}, is_cleaned: {self.is_cleaned}"

    @classmethod
    def create(
        cls,
        upload_path,
        user_id=None,
        dataset_id=None,
        speaker_display_name=None,
        transcript=None,
        meta_json=None,
        is_cleaned=False,
    ):
        query = (
            dataset_audio_file.insert()
            .values(
                upload_path=upload_path,
                user_id=user_id,
                dataset_id=dataset_id,
                speaker_display_name=speaker_display_name,
                transcript=transcript,
                meta_json=meta_json,
                uuid=str(uuid4()),
                created_at=datetime.utcnow(),
                is_cleaned=is_cleaned,
            )
            .returning(*dataset_audio_file.c)
        )
        return query

    @classmethod
    def get_by_dataset_id(
        cls,
        dataset_id: int,
        is_cleaned: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ):
        filters = [
            cls.dataset_id == dataset_id,
            cls.deleted_at == None,
        ]
        if is_cleaned is not None:
            filters += [cls.is_cleaned == is_cleaned]

        query = (
            select(cls, func.count(cls.id).over().label("total"))
            .where(*filters)
            .order_by(cls.id)
            .group_by(cls.id)
        )
        if limit is not None:
            query = query.limit(limit).offset(offset)

        return query

    @classmethod
    def get_by_data_source_id(cls, data_source_id: int, is_cleaned: bool = False):
        from openduck_py.models.dataset import DBDataset
        from openduck_py.models.data_source import DBDataSource

        query = (
            select(cls)
            .join(DBDataset, DBDataset.id == cls.dataset_id)
            .join(DBDataSource, DBDataSource.id == DBDataset.data_source_id)
            .where(DBDataSource.id == data_source_id)
        )
        return query

    @classmethod
    def get_by_user_id(cls, user_id: int):
        return select(cls).where(cls.user_id == user_id)

    @classmethod
    def get_by_uuid(cls, uuid: str):
        return select(cls).where(cls.uuid == uuid)

    @classmethod
    def get_by_uuid_and_user_id(cls, uuid: str, user_id: int):
        return select(cls).where(cls.uuid == uuid, cls.user_id == user_id)

    @classmethod
    def get_by_upload_path_cleaned(cls, upload_path: str, is_cleaned: bool):
        return select(cls).where(
            cls.upload_path == upload_path, cls.is_cleaned == is_cleaned
        )

    @classmethod
    def update_by_uuid(cls, uuid: str, **kwargs):
        supported_kwargs = {"upload_path", "meta_json", "transcript", "deleted_at"}
        updates = {k: v for k, v in kwargs.items() if k in supported_kwargs}
        return (
            update(cls)
            .where(cls.uuid == uuid)
            .values(**updates)
            .returning(*dataset_audio_file.c)
        )

    @classmethod
    def bulk_delete_by_dataset_id(cls, dataset_id: int, is_cleaned: bool):
        return (
            update(cls)
            .where(cls.dataset_id == dataset_id, cls.is_cleaned == is_cleaned)
            .values(deleted_at=datetime.utcnow())
            .returning(*dataset_audio_file.c)
        )

    @property
    def bucket(self):
        return AUDIO_FILE_BUCKET_CLEANED if self.is_cleaned else AUDIO_FILE_BUCKET

    # NOTE (Sam): to-do - simplify code in the dataset router using this function.
    def download(self, resample=False, normalize=False, root_path="/tmp"):
        uuid_ = self.upload_path.split(".wav")[0]
        output_dir = str(Path(root_path) / Path(uuid_))
        os.makedirs(Path(output_dir), exist_ok=True)
        # NOTE (Sam): just using the first letter to make subpath trunctation easier.
        output_path = f"{output_dir}/audio_resample{str(resample)[:1]}_normalize{str(normalize)[:1]}.wav"
        # NOTE (Sam): doesn't handle edge case when data has been downloaded but not manipulated and now we want to manipulate it
        if not Path(output_path).exists():
            tf = tempfile.NamedTemporaryFile(suffix=".wav")
            tf.close()
            if resample and normalize:
                if (
                    self.meta_json is not None
                    and "22k_normalized_path" in self.meta_json
                ):
                    download_file(
                        self.meta_json["22k_normalized_path"],
                        self.bucket,
                        tf.name,
                    )
                # TODO (Sam): handle other cases.
                else:
                    download_file(self.upload_path, self.bucket, tf.name)
                    resampled_audio, sr = librosa.load(tf.name)
                    assert sr == 22050
                    # NOTE (Sam): to-do - enable audacity-like volume normalization
                    # NOTE (Sam): integer wav file is not usable by spectrogram encoder and must first be transformed to within (-1,1)
                    # NOTE (Sam): somehow the factor of 2 makes an audible difference as well (even without spectrogram encoding)
                    normalized_resampled_audio = np.asarray(
                        resampled_audio
                        / np.abs(resampled_audio).max()
                        * MAX_WAV_VALUE
                        / 2,
                        dtype=np.int16,
                    )
                    write(
                        data=normalized_resampled_audio,
                        rate=sr,
                        filename=output_path,
                    )
            elif resample:
                download_file(self.upload_path, self.bucket, tf.name)
                resampled_audio, sr = librosa.load(tf.name)
                if np.abs(resampled_audio.max()) > 1.0:
                    resampled_audio = resampled_audio / np.abs(resampled_audio).max()
                assert sr == 22050
                resampled_audio = (resampled_audio * MAX_WAV_VALUE).astype(np.int16)
                write(
                    data=resampled_audio,
                    rate=sr,
                    filename=output_path,
                )
            elif normalize:
                download_file(self.upload_path, self.bucket, tf.name)
                rate, audio = read(tf.name)
                normalized_audio = np.asarray(
                    audio / np.abs(audio).max() * MAX_WAV_VALUE / 2,
                    dtype=np.int16,
                )
                write(
                    data=normalized_audio,
                    rate=rate,
                    filename=output_path,
                )
            else:
                download_file(self.upload_path, self.bucket, output_path)
        return output_path


dataset_audio_file = DBDatasetAudioFile.__table__
