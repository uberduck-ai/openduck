"use client";

import { useCallback, useEffect, useState } from "react";
import {
  DailyProvider,
  DailyAudio,
  useParticipantIds,
  useScreenShare,
  useDailyEvent,
  useMeetingState,
  useLocalSessionId,
  useVideoTrack,
  useParticipantProperty,
  useCallObject,
} from "@daily-co/daily-react";
import DailyIframe, { DailyCall } from "@daily-co/daily-js";

const refreshPage = () => {
  console.log(
    "make sure to allow access to your microphone and camera in your browser's permissions"
  );
  window.location.reload();
};

function UserMediaError() {
  return (
    <div className="bg-red-100 p-4 rounded-lg shadow">
      <div className="text-center">
        <h1 className="text-xl font-semibold text-red-700">
          Camera or mic blocked
        </h1>
        <button
          onClick={refreshPage}
          type="button"
          className="mt-4 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Try again
        </button>
        <p className="mt-2">
          <a
            href="https://docs.daily.co/guides/how-daily-works/handling-device-permissions"
            target="_blank"
            rel="noreferrer"
            className="text-blue-600 hover:underline"
          >
            Get help
          </a>
        </p>
      </div>
    </div>
  );
}

function Username({ id, isLocal }) {
  const username = useParticipantProperty(id, "user_name");

  return (
    <div className="text-sm font-medium text-gray-700">
      {username || id}{" "}
      {isLocal && <span className="text-green-500">(you)</span>}
    </div>
  );
}

function Tile({
  id,
  isScreenShare,
  isLocal,
  isAlone,
}: {
  id: string;
  isScreenShare?: boolean;
  isLocal?: boolean;
  isAlone?: boolean;
}) {
  const videoState = useVideoTrack(id);

  let containerCssClasses = "rounded-lg overflow-hidden shadow-lg m-2 ";
  containerCssClasses += isScreenShare ? "bg-blue-100" : "bg-gray-100";

  if (isLocal) {
    containerCssClasses += " border-2 border-green-500";
    if (isAlone) {
      containerCssClasses += " opacity-50";
    }
  }

  if (videoState.isOff) {
    containerCssClasses += " bg-gray-300";
  }

  return (
    <div className={containerCssClasses}>
      <div className="p-4">
        {!isScreenShare && <Username id={id} isLocal={isLocal} />}
      </div>
    </div>
  );
}

function Call() {
  const [getUserMediaError, setGetUserMediaError] = useState(false);

  useDailyEvent(
    "camera-error",
    useCallback(() => {
      setGetUserMediaError(true);
    }, [])
  );

  const { screens } = useScreenShare();
  const remoteParticipantIds = useParticipantIds({ filter: "remote" });

  const localSessionId = useLocalSessionId();
  const isAlone = remoteParticipantIds.length < 1 && screens.length < 1;

  const renderCallScreen = () => (
    <div className="flex flex-wrap justify-center items-center p-4">
      {localSessionId && <Tile id={localSessionId} isLocal isAlone={isAlone} />}
      {remoteParticipantIds.map((id) => (
        <Tile key={id} id={id} />
      ))}
      {screens.map((screen) => (
        <Tile key={screen.screenId} id={screen.session_id} isScreenShare />
      ))}
      {isAlone && (
        <div className="text-center p-4 m-4 rounded-lg shadow-lg bg-yellow-100">
          <h1 className="text-lg font-semibold">Waiting for others</h1>
          <p className="mt-2">Invite someone by sharing this link:</p>
          <span className="text-blue-600">{window.location.href}</span>
        </div>
      )}
    </div>
  );

  return getUserMediaError ? <UserMediaError /> : renderCallScreen();
}

const AudioCall = ({ callObject }) => {
  // const [callObject, setCallObject] = useState<DailyCall | null>(null);
  // const callObject = useCallObject({}, () => false);
  const [roomUrl, setRoomUrl] = useState(
    "https://matthewkennedy5.daily.co/Od7ecHzUW4knP6hS5bug"
  );
  const [joinedRoom, setJoinedRoom] = useState(false);
  const [micOn, setMicOn] = useState(true);
  const meetingState = useMeetingState();

  console.log("meeting state", meetingState);

  const toggleMic = () => {
    callObject?.setLocalAudio(!callObject?.localAudio());
    setMicOn(!micOn);
  };

  const leaveCall = useCallback(() => {
    callObject?.leave();
    setJoinedRoom(false);
  }, [callObject]);

  const joinCall = useCallback(
    (userName: string) => {
      console.log("[DAILY] Joining room", roomUrl);
      callObject?.join({ url: roomUrl, userName });
      setJoinedRoom(true);
    },
    [callObject, roomUrl]
  );

  const startHairCheck = useCallback(
    async (url: string) => {
      // const newCallObject = DailyIframe.createCallObject();
      setRoomUrl(url);
      if (!callObject) {
        console.log("No call object");
        return;
      }
      // setCallObject(newCallObject);
      await callObject.preAuth({ url });
      await callObject.startCamera();
    },
    [callObject]
  );

  useEffect(() => {
    startHairCheck(roomUrl);
  }, []);

  return (
    <div className="flex flex-col space-y-4 p-4">
      <input
        type="text"
        className="form-input px-4 py-2 border rounded"
        value={roomUrl}
        onChange={(e) => setRoomUrl(e.target.value)}
        placeholder="Enter room URL"
      />
      <div className="flex space-x-2">
        {joinedRoom ? (
          <button
            className="px-4 py-2 rounded font-bold bg-red-500 hover:bg-red-600 text-white"
            onClick={leaveCall}
          >
            Leave Room
          </button>
        ) : (
          <button
            className="px-4 py-2 rounded font-bold bg-blue-500 hover:bg-blue-600 text-white"
            onClick={() => joinCall("User")}
          >
            Join Room
          </button>
        )}
        <button
          className="px-4 py-2 bg-gray-300 hover:bg-gray-400 text-black rounded"
          onClick={toggleMic}
        >
          {micOn ? "Mic On" : "Mic Off"}
        </button>
      </div>
      <div>
        <Call />
        <DailyAudio />
      </div>
    </div>
  );
};

export default function Home() {
  const callObject = useCallObject({});
  return (
    <DailyProvider callObject={callObject}>
      <main className="min-h-screen bg-gray-50 flex flex-col items-center justify-center">
        <AudioCall callObject={callObject} />
      </main>
    </DailyProvider>
  );
}
