type Landmark = {};

const landmarks: { [id: string]: Landmark } = {};

export function removeLandmark(id: string) {
  delete landmarks[id];
}

export function putLandmark(id: string, landmark: Landmark) {
  landmarks[id] = landmark;
}

export function getLandmarks() {
  return landmarks;
}
