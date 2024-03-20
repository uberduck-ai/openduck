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
  toggleMic,
  micOn,
}: {
  id: string;
  isScreenShare?: boolean;
  isLocal?: boolean;
  isAlone?: boolean;
  toggleMic?: () => void;
  micOn?: boolean;
}) {
  const videoState = useVideoTrack(id);

  let containerCssClasses =
    "rounded-lg overflow-hidden shadow-lg m-2 border-2 ";
  containerCssClasses += isScreenShare ? "bg-blue-100" : "bg-gray-100";

  if (isLocal) {
    containerCssClasses += " border-green-500 ";
    if (isAlone) {
      containerCssClasses += " opacity-50";
    }
  } else {
    containerCssClasses += " border-gray-300 ";
  }

  if (videoState.isOff) {
    containerCssClasses += " bg-gray-300";
  }

  let micButtonClasses = "absolute bottom-4 right-4 px-2 py-1 ";
  micButtonClasses += micOn
    ? "bg-green-500 text-white"
    : "bg-red-500 text-white";

  return (
    <div className={containerCssClasses}>
      <div className="flex h-16 p-4 flex-row items-center">
        {!isScreenShare && <Username id={id} isLocal={isLocal} />}
        {isLocal && (
          <Button
            variant={micOn ? "success" : "danger"}
            className={"text-xs ml-4"}
            // className={micButtonClasses}
            onClick={toggleMic}
          >
            {micOn ? "unmute" : "mute"}
          </Button>
        )}
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin -ml-1 mr-3 h-5 w-5 text-black"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      ></circle>
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      ></path>
    </svg>
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

function Call({ toggleMic, micOn }: { toggleMic: () => void; micOn: boolean }) {
  const [getUserMediaError, setGetUserMediaError] = useState(false);
  const meetingState = useMeetingState();

  console.log("Meeting State: ", meetingState);

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
      {localSessionId && (
        <Tile
          id={localSessionId}
          isLocal
          isAlone={isAlone}
          toggleMic={toggleMic}
          micOn={micOn}
        />
      )}
      {remoteParticipantIds.map((id) => (
        <Tile key={id} id={id} />
      ))}
      {screens.map((screen) => (
        <Tile key={screen.screenId} id={screen.session_id} isScreenShare />
      ))}
      {isAlone && meetingState === "joined-meeting" && (
        <div className="text-center p-4 m-4 rounded-lg shadow-lg bg-yellow-100 flex flex-col items-center">
          <h1 className="text-lg font-semibold">Waiting for others</h1>
          <Spinner />
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
  const [userName, setUserName] = useState<string>("");

  const toggleMic = () => {
    callObject?.setLocalAudio(!callObject?.localAudio());
    setMicOn(!micOn);
  };

  const leaveCall = useCallback(() => {
    callObject?.leave();
    setJoinedRoom(false);
  }, [callObject]);

  const handleOrbClick = useCallback(async () => {
    if (joinedRoom) {
      leaveCall();
    } else {
      try {
        const response = await fetch(`${apiHost}/audio/start`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });
        const room = await response.json();
        if (room.url) {
          console.log("Room created and joining:", room.url);
          setRoomUrl(room.url);
          callObject?.join({ url: room.url, userName: userName });
          setJoinedRoom(true);
        } else {
          console.error("Failed to create room");
        }
      } catch (error) {
        console.error("Error creating room:", error);
      }
    }
  }, [callObject, roomUrl, userName]);

  return (
    <div className="flex flex-col items-center space-y-4 p-4">
      <input
        type="text"
        placeholder="Your Name"
        value={userName}
        onChange={(e) => setUserName(e.target.value)}
        className="text-center p-2 rounded-lg border-2 border-gray-300 mb-4"
        disabled={joinedRoom}
      />
      <button
        className={`orb-button min-w-32 bg-blue-500 hover:bg-blue-600 text-white rounded-full p-4 cursor-pointer shadow-lg transform transition-transform duration-300 ease-in-out ${
          !userName && "opacity-50 cursor-not-allowed"
        }`}
        onClick={handleOrbClick}
        onMouseOver={(e) => e.currentTarget.classList.add("hover:shadow-xl")}
        onMouseOut={(e) => e.currentTarget.classList.remove("hover:shadow-xl")}
        disabled={!userName}
      >
        {joinedRoom ? "Leave Room" : "Start"}
      </button>
      {roomUrl && false && <div className="text-sm">Room URL: {roomUrl}</div>}
      <div>
        <Call toggleMic={toggleMic} micOn={micOn} />
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
        <h1 className="text-2xl font-bold text-center mb-4">
          An AI podcast featuring you and an infinite cast of AI friends
        </h1>
        <AudioCall callObject={callObject} />
      </main>
    </DailyProvider>
  );
}
