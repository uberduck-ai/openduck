"use client";

import { useCallback, useRef, useState } from "react";
import Image from "next/image";
import { DailyProvider, useCallObject } from "@daily-co/daily-react";

export default function Home() {
  const callObject = useCallObject({});
  const callRef = useRef(null);

  console.log(callObject);
  return (
    <DailyProvider callObject={callObject}>
      <main className="flex min-h-screen flex-col items-center justify-between p-24">
        <div>hi</div>
        <div ref={callRef}></div>
      </main>
    </DailyProvider>
  );
}
