/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as posenet_module from "@tensorflow-models/posenet";
import * as facemesh_module from "@tensorflow-models/facemesh";
import * as tf from "@tensorflow/tfjs";
import { PaperScope } from "paper";
import "babel-polyfill";

import {
  drawKeypoints,
  drawPoint,
  drawSkeleton,
  isMobile,
  toggleLoadingUI,
  setStatusText,
} from "./utils/demoUtils";
import { SVGUtils } from "./utils/svgUtils";
import { PoseIllustration } from "./illustrationGen/illustration";
import { Skeleton, facePartName2Index } from "./illustrationGen/skeleton";
import { FileUtils } from "./utils/fileUtils";

import * as girlSVG from "./resources/illustration/girl.svg";
import * as boySVG from "./resources/illustration/boy.svg";
import * as abstractSVG from "./resources/illustration/abstract.svg";
import * as blathersSVG from "./resources/illustration/blathers.svg";
import * as tomNookSVG from "./resources/illustration/tom-nook.svg";

// Camera stream video element
let video;
let videoWidth = 256;
let videoHeight = 256;

// Canvas
let canvasWidth = 200;
let canvasHeight = 200;

// ML models
let minPoseConfidence = 0.15;
let minPartConfidence = 0.1;
let nmsRadius = 30.0;

// Misc
let mobile = false;
const avatarSvgs = {
  girl: girlSVG.default,
  boy: boySVG.default,
  abstract: abstractSVG.default,
  blathers: blathersSVG.default,
  "tom-nook": tomNookSVG.default,
};

const canvasContainer: HTMLElement =
  document.querySelector(".canvas-container")!;

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera(): Promise<HTMLVideoElement> {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Browser API navigator.mediaDevices.getUserMedia not available"
    );
  }

  const video = document.getElementById("video") as HTMLVideoElement;
  video.width = videoWidth;
  video.height = videoHeight;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: videoWidth,
      height: videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo(): Promise<HTMLVideoElement> {
  const video = await setupCamera();
  video.play();

  return video;
}

const defaultPoseNetArchitecture = "MobileNetV1";
const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 200;

const guiState = {
  avatarSVG: Object.keys(avatarSvgs)[0],
  debug: {
    showDetectionDebug: true,
    showIllustrationDebug: false,
  },
};

async function detectPose({
  facemesh,
  posenet,
  video,
}: {
  video: HTMLVideoElement;
  facemesh: facemesh_module.FaceMesh;
  posenet: posenet_module.PoseNet;
}) {
  // Creates a tensor from an image
  const input = tf.browser.fromPixels(video);
  const faceDetections = await facemesh.estimateFaces(input, false, false);
  const poses = await posenet.estimatePoses(video, {
    flipHorizontal: false,
    decodingMethod: "multi-person",
    maxDetections: 1,
    scoreThreshold: minPartConfidence,
    nmsRadius: nmsRadius,
  });
  input.dispose();

  return {
    poses,
    faceDetections,
  };
}

type Context = {
  video: HTMLVideoElement;
  facemesh: facemesh_module.FaceMesh;
  posenet: posenet_module.PoseNet;
  userIllustrations: { [id: string]: UserIllustration };
  socket: WebSocket;
  id: string;
  landmarks: Landmarks;
};

async function onAnimationFrame(context: Context) {
  const { facemesh, posenet, video, landmarks, socket } = context;

  const { poses, faceDetections } = await detectPose({
    facemesh,
    posenet,
    video,
  });

  const myLandmark = getLandmark(poses, faceDetections);
  if (myLandmark) {
    sendLandmark(socket, myLandmark);
    landmarks[context.id] = myLandmark;
  }

  clearUnusedUserIllustrations(context, Object.keys(landmarks));

  const promises = Object.entries(landmarks).map(async ([id, landmark]) => {
    const userIllustration = await getUserIllustration(context, id);

    renderLandmark(landmark, userIllustration);
  });
  await Promise.all(promises);

  requestAnimationFrame(() => onAnimationFrame(context));
}

function renderLandmark(
  landmark: Landmark,
  userIllustration: UserIllustration
) {
  const { illustration, scope } = userIllustration;

  scope.project.clear();

  illustration.updateSkeleton(landmark.pose, landmark.face);
  illustration.draw();

  scope.project.activeLayer.scale(
    canvasWidth / videoWidth,
    canvasHeight / videoHeight,
    new scope.Point(0, 0)
  );
}

function getLandmark(
  poses: posenet_module.Pose[],
  faceDetections: facemesh_module.AnnotatedPrediction[]
): Landmark | undefined {
  const pose = poses[0];
  const faceDetection = faceDetections[0];
  if (!pose || !faceDetection) {
    return;
  }
  return {
    pose,
    face: Skeleton.toFaceFrame(faceDetection),
  };
}
function setupCanvas() {
  mobile = isMobile();
  if (mobile) {
    canvasWidth = Math.min(window.innerWidth, window.innerHeight);
    canvasHeight = canvasWidth;
    videoWidth *= 0.7;
    videoHeight *= 0.7;
  }
}

function initSocket(): Promise<{
  socket: WebSocket;
  id: string;
}> {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(`ws://${location.host}`);
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "setId") {
        resolve({
          socket,
          id: data.id,
        });
      }
    };
    socket.onerror = (err) => {
      reject(err);
    };
  });
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  const { socket, id } = await initSocket();

  toggleLoadingUI(true);
  setStatusText("Loading PoseNet model...");
  const posenet = await posenet_module.load({
    architecture: defaultPoseNetArchitecture,
    outputStride: defaultStride,
    inputResolution: defaultInputResolution,
    multiplier: defaultMultiplier,
    quantBytes: defaultQuantBytes,
  });
  setStatusText("Loading FaceMesh model...");
  const facemesh = await facemesh_module.load();

  setStatusText("Loading Avatar file...");

  setStatusText("Setting up camera...");
  const video = await loadVideo();

  toggleLoadingUI(false);

  const context: Context = {
    facemesh,
    posenet,
    video,
    socket,
    id,
    landmarks: {},
    userIllustrations: {},
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch (data.type) {
      case "landmarks":
        {
          const landmarks: Landmarks = data.landmarks;
          context.landmarks = landmarks;
        }
        break;
    }
  };

  requestAnimationFrame(() => onAnimationFrame(context));
}

type UserIllustration = {
  illustration: PoseIllustration;
  scope: paper.PaperScope;
  canvas: HTMLCanvasElement;
};

async function initIllustration(
  canvas: HTMLCanvasElement
): Promise<UserIllustration> {
  const scope = new PaperScope();
  scope.setup(canvas);

  const svgScope = await SVGUtils.importSVG(boySVG.default);
  const skeleton = new Skeleton(svgScope);
  const illustration = new PoseIllustration(scope);
  illustration.bindSkeleton(skeleton, svgScope);

  return {
    scope,
    illustration,
    canvas,
  };
}

bindPage();

type Face = ReturnType<typeof Skeleton.toFaceFrame>;

type Landmark = {
  pose: posenet_module.Pose;
  face: Face;
};

type Landmarks = { [id: string]: Landmark };

function sendLandmark(socket: WebSocket, landmark: Landmark) {
  socket.send(JSON.stringify(landmark));
}
async function getUserIllustration(
  context: Context,
  id: string
): Promise<UserIllustration> {
  let userIllustration = context.userIllustrations[id];
  if (userIllustration) {
    return userIllustration;
  }

  const canvas = document.createElement("canvas");
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  canvasContainer.appendChild(canvas);

  userIllustration = await initIllustration(canvas);

  context.userIllustrations[id] = userIllustration;

  return userIllustration;
}

function clearUnusedUserIllustrations(context: Context, ids: string[]) {
  for (const id of Object.keys(context.userIllustrations)) {
    if (!ids.includes(id)) {
      const userIllustration = context.userIllustrations[id];
      canvasContainer.removeChild(userIllustration.canvas);
      delete context.userIllustrations[id];
    }
  }
}
