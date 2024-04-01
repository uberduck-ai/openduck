"use client";

import { useCallback, useEffect, useState } from "react";
import {
  DailyProvider,
  DailyAudio,
  useActiveSpeakerId,
  useParticipantIds,
  useScreenShare,
  useDailyEvent,
  useMeetingState,
  useLocalSessionId,
  useVideoTrack,
  useParticipantProperty,
  useCallObject,
  useDevices,
} from "@daily-co/daily-react";
import { DailyCall } from "@daily-co/daily-js";

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
  isActiveSpeaker,
}: {
  id: string;
  isScreenShare?: boolean;
  isLocal?: boolean;
  isAlone?: boolean;
  toggleMic?: () => void;
  micOn?: boolean;
  isActiveSpeaker?: boolean;
}) {
  const videoState = useVideoTrack(id);

  let containerCssClasses =
    "rounded-lg overflow-hidden shadow-lg m-2 border-2 transition-all duration-500 ";
  containerCssClasses += isScreenShare ? "bg-blue-100 " : "bg-gray-100 ";

  if (isLocal) {
    containerCssClasses += " border-green-500 ";
    if (isAlone) {
      containerCssClasses += " opacity-50 ";
    }
  } else {
    containerCssClasses += " border-gray-300 ";
  }

  if (videoState.isOff) {
    containerCssClasses += " bg-gray-300 ";
  }

  if (isActiveSpeaker) {
    containerCssClasses += " ring-4 ring-yellow-500 ring-opacity-50 ";
  }

  let micButtonClasses = "absolute bottom-4 right-4 px-2 py-1 ";
  micButtonClasses += micOn
    ? "bg-green-500 text-white "
    : "bg-red-500 text-white ";

  return (
    <div className={containerCssClasses}>
      <div className="flex h-16 p-4 flex-row items-center">
        {!isScreenShare && <Username id={id} isLocal={isLocal} />}
        {isLocal && (
          <Button
            variant={micOn ? "success" : "danger"}
            className={"text-xs ml-4"}
            onClick={toggleMic}
          >
            {micOn ? "unmute" : "mute"}
          </Button>
        )}
      </div>
    </div>
  );
}

function Spinner({ color = "text-black" }) {
  return (
    <svg
      className={`animate-spin -ml-1 mr-3 h-5 w-5 ${color}`}
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

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label: string;
  options: { value: string; label: string }[];
  id: string;
}

const Select = ({ label, options, id, ...rest }: SelectProps) => {
  return (
    <div className="w-full">
      <label htmlFor={id} className="block text-sm font-medium text-gray-700">
        {label}
      </label>
      <select
        id={id}
        {...rest}
        className="mt-1 p-2 rounded-lg border-2 border-gray-300 block w-full"
      >
        {options.map((option, index) => (
          <option key={index} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
};

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
  const activeSpeakerId = useActiveSpeakerId();

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
          isActiveSpeaker={activeSpeakerId === localSessionId}
        />
      )}
      {remoteParticipantIds.map((id) => (
        <Tile key={id} id={id} isActiveSpeaker={activeSpeakerId === id} />
      ))}
      {screens.map((screen) => (
        <Tile
          key={screen.screenId}
          id={screen.session_id}
          isScreenShare
          isActiveSpeaker={activeSpeakerId === screen.session_id}
        />
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
  const meetingState = useMeetingState();
  const [isJoining, setIsJoining] = useState(false);
  const { currentMic, microphones, setMicrophone } = useDevices();
  const [conversationType, setConversationType] = useState("podcast");

  const toggleMic = () => {
    callObject?.setLocalAudio(!callObject?.localAudio());
    setMicOn(!micOn);
  };

  const leaveCall = useCallback(async () => {
    await callObject?.leave();
    setJoinedRoom(false);
  }, [callObject]);

  const handleJoinClick = useCallback(async () => {
    console.log("hi", joinedRoom);
    if (joinedRoom) {
      leaveCall();
    } else {
      setIsJoining(true);
      try {
        const response = await fetch(`${apiHost}/audio/start`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt: conversationType,
            context: {
              local_time: new Date().toLocaleTimeString([], {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
              }),
              local_date: new Date().toLocaleDateString(),
              name: userName,
              num_prev_conversations: 0,
              is_public: false,
              topics: [],
            },
          }),
        });
        const room = await response.json();
        if (room.url) {
          console.log("Room created and joining:", room.url);
          setRoomUrl(room.url);
          try {
            await callObject?.join({ url: room.url, userName: userName });
          } catch (error) {
            await new Promise((resolve) => setTimeout(resolve, 1000)); // Sleep for 1 second
            await callObject?.join({ url: room.url, userName: userName });
          }
          setIsJoining(false);
          setJoinedRoom(true);
        } else {
          console.error("Failed to create room");
          alert("couldn't join room");
        }
      } catch (error) {
        console.error("Error creating room:", error);
      } finally {
        setIsJoining(false);
      }
    }
  }, [callObject, roomUrl, userName, joinedRoom, conversationType]);

  return (
    <div className="flex flex-col items-center space-y-4 p-4 w-full md:w-96">
      <button
        className={`orb-button w-48 h-48 bg-blue-500 enabled:hover:bg-blue-600 text-white rounded-full p-4 shadow-lg transform transition-transform duration-300 ease-in-out flex items-center justify-center text-2xl ${
          isJoining && "opacity-50 cursor-not-allowed"
        }`}
        onClick={handleJoinClick}
        onMouseOver={(e) => e.currentTarget.classList.add("hover:shadow-xl")}
        onMouseOut={(e) => e.currentTarget.classList.remove("hover:shadow-xl")}
        disabled={isJoining}
      >
        <div>{isJoining && <Spinner color="text-white" />}</div>
        <div>{joinedRoom ? "Leave Room" : "Try It!"}</div>
      </button>

      <Select
        label="Conversation Type"
        value={conversationType}
        options={[
          {
            value: "podcast",
            label: "Business Podcast - For content creation",
          },
          { value: "todo", label: "TODO List - For organizing thoughts" },
          { value: "comedy", label: "Joking around - Just for fun" },
        ]}
        onChange={(e) => setConversationType(e.target.value)}
        id="conversation-type"
      />
      {currentMic && (
        <Select
          label="Select Microphone"
          options={microphones.map((mic) => ({
            value: mic.device.deviceId,
            label: mic.device.label,
          }))}
          id="microphone-select"
          onChange={(e) => setMicrophone(e.target.value)}
        />
      )}
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
  callObject?.updateInputSettings({
    audio: {
      processor: {
        type: "noise-cancellation",
      },
    },
  });
  return (
    <DailyProvider callObject={callObject}>
      <main className="min-h-screen bg-gray-50 flex flex-col items-center">
        <h1 className="text-2xl font-bold text-center mt-8 mb-4 mx-4">
          Recorded, shareable voice chats with AI.
        </h1>
        <div className="flex items-start">
          <AudioCall callObject={callObject} />
        </div>
        <div className="text-left mt-4 sm:mt-4 ml-2 mr-2">
          <h2 className="text-lg font-bold mb-4">Why?</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li className="text-sm">
              <span className="font-semibold">Create authentic content:</span>{" "}
              Just have a conversation. We&apos;ll turn it into video content.
            </li>
            <li className="text-sm">
              <span className="font-semibold">Think out loud:</span> Talk
              Through your thoughts, daily TODO list, or ideas.
            </li>
            <li className="text-sm">
              <span className="font-semibold">Experiment with AI:</span>{" "}
              Understanding AI better by interacting with it over voice.
            </li>
            <li className="text-sm">
              <span className="font-semibold">Just have fun:</span> Say absurd
              things. Share the results.
            </li>
          </ul>
        </div>
      </main>
    </DailyProvider>
  );
}
