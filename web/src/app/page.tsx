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

const apiHost = process.env.NEXT_PUBLIC_API_URL;

console.log("API HOST: ", apiHost);

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

function Username({ id, isLocal }: { id: string; isLocal: boolean }) {
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
  isLocal = false,
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

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant: "success" | "danger" | "primary" | "secondary";
  children: React.ReactNode;
}

function Button({ variant, children, ...rest }: ButtonProps) {
  const baseStyle = "px-4 py-2 rounded font-bold text-white ";
  let variantStyle = "";
  let additionalClasses = rest.className ? ` ${rest.className}` : "";

  switch (variant) {
    case "success":
      variantStyle = "bg-green-500 hover:bg-green-600";
      break;
    case "danger":
      variantStyle = "bg-red-500 hover:bg-red-600";
      break;
    case "primary":
      variantStyle = rest.disabled
        ? "bg-blue-300"
        : "bg-blue-500 hover:bg-blue-600";
      break;
    case "secondary":
      variantStyle = "bg-gray-500 hover:bg-gray-600 text-black";
      break;
    default:
      variantStyle = "bg-gray-200"; // Fallback style
  }

  // Exclude className from rest since it's already applied
  const { className, ...restProps } = rest;

  return (
    <button
      className={`${baseStyle} ${variantStyle}${additionalClasses}`}
      {...restProps}
    >
      {children}
    </button>
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
        </div>
      )}
    </div>
  );

  return getUserMediaError ? <UserMediaError /> : renderCallScreen();
}

const AudioCall = ({ callObject }: { callObject: DailyCall | null }) => {
  const [roomUrl, setRoomUrl] = useState<string>("");
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
      if (!url) return;
      if (!callObject) {
        console.log("No call object");
        return;
      }
      await callObject.preAuth({ url });
      await callObject.startCamera();
    },
    [callObject]
  );

  const createRoom = useCallback(async () => {
    console.log("creatre rom");
    try {
      const response = await fetch(`${apiHost}/audio/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const room = await response.json();
      if (room.url) {
        console.log("Room created:", room.url);
        setRoomUrl(room.url);
      } else {
        console.error("Failed to create room");
      }
    } catch (error) {
      console.error("Error creating room:", error);
    }
  }, []);

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
        <Button variant="success" onClick={createRoom}>
          Create Room
        </Button>
        {joinedRoom ? (
          <Button variant="danger" onClick={leaveCall}>
            Leave Room
          </Button>
        ) : (
          <Button
            className={`${
              roomUrl &&
              roomUrl.match(
                /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/
              )
                ? "bg-blue-500 hover:bg-blue-600"
                : "bg-blue-300"
            } text-white`}
            onClick={() => joinCall("User")}
            disabled={
              !roomUrl ||
              !roomUrl.match(
                /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/
              )
            }
            variant="primary"
          >
            Join Room
          </Button>
        )}
        <Button variant="secondary" onClick={toggleMic}>
          {micOn ? "Mic On" : "Mic Off"}
        </Button>
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
